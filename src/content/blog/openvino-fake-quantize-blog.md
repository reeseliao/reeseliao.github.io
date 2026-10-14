---
title: "Implementing fake_quantize in OpenVINO's PyTorch Frontend"
date: 2026-05-14
description: "How I contributed PR #34577 to OpenVINO (merged Oct 2026) – adding support for the learnable per-channel fake quantize op so QAT-trained models can be converted to OpenVINO IR without import failures."
tags: [OpenVINO, PyTorch, Quantization, Edge Inference, Open Source]
---

> **Update (Oct 13, 2026):** This PR has been merged into OpenVINO master (commit [`10864ee`](https://github.com/openvinotoolkit/openvino/commit/10864ee713c0c1d4f967506bd88eb359b86a5b4a), resolves [#29682](https://github.com/openvinotoolkit/openvino/issues/29682)). 

## TL;DR

I contributed PR [#34577](https://github.com/openvinotoolkit/openvino/pull/34577) to OpenVINO, adding support for `aten::_fake_quantize_learnable_per_channel_affine` in the PyTorch frontend. This lets QAT-trained models (ones that learned to be robust to quantization error during training) be converted to OpenVINO IR without import failures. The implementation maps the PyTorch op to OpenVINO's native `v0::FakeQuantize` node through a per-channel helper shared with the existing converter, with the main challenge being the "learnable" zero point: it is a float tensor that PyTorch rounds and clamps before quantizing, and the converter has to do the same.

## Background: What Is Fake Quantization?

Quantization shrinks model weights from FP32 to INT8 (or lower), making inference faster and smaller — critical for edge deployment. But naively quantizing a trained model (post-training quantization) often hurts accuracy because the model never learned to handle rounding errors.

Quantization-Aware Training (QAT) solves this by inserting **fake quantization nodes** during training. These nodes simulate the precision loss of quantization in the forward pass — they clamp, round, and rescale values — so the model gradually learns weights that are robust to that error. By the time you actually quantize for deployment, accuracy barely drops.

The "learnable" variant goes a step further: instead of keeping `scale` and `zero_point` fixed, it makes them **trainable parameters**. The model doesn't just tolerate quantization error — it actively learns the optimal quantization grid. This is particularly effective for handling outliers in weight distributions, because the learned scale can adapt to the actual value range rather than relying on a static min/max calculation.

## The Problem

OpenVINO's PyTorch frontend converts PyTorch models to OpenVINO's Intermediate Representation (IR). It works by walking through the model's ATen-level operations and translating each one to the corresponding OpenVINO op. Think of it as a translator between two languages — PyTorch's op vocabulary and OpenVINO's op vocabulary.

The issue: `aten::_fake_quantize_learnable_per_channel_affine` wasn't in the translator's dictionary. Any QAT-trained model using this op would fail at import with an unrecognized operator error.

The non-learnable `fake_quantize_per_tensor_affine` and `fake_quantize_per_channel_affine` were already supported. The learnable per-channel variant looks almost the same, but differs in two ways:

- An extra `grad_factor` parameter that doesn't exist in the non-learnable version.
- `zero_point` is a **float** tensor (it has to be, to receive gradients), whereas the non-learnable op takes an integer zero point.

## Implementation

The merged change touched four files (+124 / −15).

**1. Shared per-channel helper** — `src/frontends/pytorch/src/op/fake_quantize.cpp`

The existing `translate_fake_quantize_per_channel_affine` already knew how to build a per-channel `v0::FakeQuantize`: broadcast the 1D `scale` and `zero_point` along the channel axis, compute the clamping bounds (`input_low` / `input_high`) from them, and set `levels = quant_max - quant_min + 1`. Rather than copy that logic, I extracted it into a helper, `make_fake_quantize_per_channel`, and made both converters call it. The old converter became a thin wrapper, and the new one only has to prepare its inputs differently.

**2. The learnable converter** — same file

Two things make the learnable variant different from a plain per-channel fake quantize:

- **`grad_factor` is dropped.** It only rescales gradients during backpropagation and has no effect on the forward pass. Since OpenVINO only handles inference, the converter accepts the 7th input and ignores it. Knowing *what to throw away* is as important as knowing what to keep.
- **The zero point is rounded and clamped.** Because `zero_point` is a learnable float, PyTorch doesn't use it as-is: before quantizing, it rounds it to the nearest integer (half-to-even) and clamps it into `[quant_min, quant_max]` (see `_get_rounded_zero_point` in PyTorch). The converter mirrors this with a `v5::Round` (`HALF_TO_EVEN`) followed by a `v0::Clamp`, then hands the result to the shared helper. Without this step, a model whose learned zero point drifted to something like `1.4` or `400.0` would convert fine but produce numerically different outputs.

**3. Op registry** — `src/frontends/pytorch/src/op_table.cpp`

The op is registered twice: once in the TorchScript table as `aten::_fake_quantize_learnable_per_channel_affine`, and once in the FX table as `aten._fake_quantize_learnable_per_channel_affine.default`, so both `torch.jit`-traced and `torch.export` / FX models are covered.

**4. Conformance docs** — `docs/sphinx_setup/_static/conformance_files/pytorch_ops.csv`

A new row lists the op as supported, with the note *"grad_factor is ignored: it only affects gradients."*

## Testing

I added `TestFakeQuantizeLearnablePerChannelAffine` in `tests/layer_tests/pytorch_tests/test_fake_quantize.py`. Rather than running a full neural network, the test wraps the op in a minimal single-layer module and verifies that PyTorch → OpenVINO conversion produces numerically equivalent results. It is parametrized over four cases, each targeting a specific behavior:

| Case | scale | zero_point | axis | range | What it checks |
|---|---|---|---|---|---|
| Baseline | `[0.005, 0.7]` | `[0, 0]` | 1 | `[0, 255]` | Basic per-channel conversion |
| Rounding | `[0.5, 0.25]` | `[1.4, 2.6]` | 1 | `[0, 255]` | Fractional zero point is rounded |
| Clamping | `[0.5, 0.25]` | `[-3.0, 400.0]` | 1 | `[0, 255]` | Out-of-range zero point is clamped |
| Signed | `[1.5, -0.7, -0.1]` | `[1, 0, -1]` | 0 | `[-128, 127]` | Different axis, signed range, negative scale |

The tests run in precommit for the TorchScript, `torch.export` and FX backend paths. Layer tests are the standard approach in OpenVINO's test suite for op-level validation — they isolate the translator logic from any model-level complexity.

## Takeaways

**Frontends are translators.** OpenVINO's PyTorch frontend doesn't "understand" PyTorch models in any deep sense. It pattern-matches op signatures and emits equivalent IR nodes. The interesting engineering is in the edge cases: parameters that exist in one framework but not the other, broadcasting semantics that differ subtly, and knowing which training-only artifacts can be safely dropped.

**Read the reference implementation, not just the signature.** The op signature only hints that `zero_point` is a float; the actual semantics — round half-to-even, then clamp — live in PyTorch's `_get_rounded_zero_point`. Matching the signature gets you a converter that compiles; matching the reference gets you one that's numerically correct.

**Navigating a large C++ codebase is pattern recognition.** 
The OpenVINO repo is massive, and finding where things live (the right `.cpp` file, the two registration tables, the conformance docs, the right test directory) took more time than writing the actual translation logic. But once you understand the pattern — translator function + registry entries + layer test — adding new ops becomes mechanical.

---

*PR: [openvinotoolkit/openvino#34577](https://github.com/openvinotoolkit/openvino/pull/34577) (merged) · Issue: [openvinotoolkit/openvino#29682](https://github.com/openvinotoolkit/openvino/issues/29682)*
