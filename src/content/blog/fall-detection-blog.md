---
title: "Beyond the IMU: Designing a Fault-Tolerant Fall Detection System for Smart Glasses"
date: 2026-06-26
description: How I built an end-to-end fall detection pipeline for Solos smart glasses combining IMU state machines, a lock-free ring buffer, VLM visual confirmation, and a three-tier photo fallback designed around a BLE bandwidth collision.
tags: [Edge AI, Android, BLE, VLM, Systems Design, Kotlin]
---

## TL;DR

Over two weeks of after-hours development, I built an end-to-end fall detection system for Solos smart glasses: IMU-based impact detection, a 30-second cancel window, VLM-based visual confirmation via Gemini, and automated alerting through n8n. The most interesting engineering problem wasn't the AI — it was a BLE bandwidth collision between sensor streaming and photo capture, which forced a redesign around a lock-free ring buffer. The system won highest score at the internal hackathon.

## The Problem

Wearable fall detection is a well-studied problem — the core IMU algorithm (detect impact, detect stillness, confirm a fall) has existed since the early 2010s. What's still unsolved in most consumer products is the false positive rate. A system that cries wolf gets ignored or turned off within weeks, which defeats the entire purpose for the elderly-care use case it's meant to serve.

My architecture treats the IMU as a *trigger*, not a *verdict*. It hands off to a vision-language model for a second opinion before anything gets sent.

## Architecture

```
IMU (phone or glasses)
  → Impact + Stillness Detection (state machine)
  → Photo Capture (three-tier fallback, see below)
  → 30s Cancel Window
  → Gemini VLM Confirmation
  → n8n Alert Workflow
  → NTFY Push Notification
```

The state machine runs: `MONITORING → IMPACT → STILLNESS → FALL_CONFIRMED`. Each phase transition is time-windowed rather than instantaneous — a raw sensor callback fires dozens of times per second, so naive threshold-checking on every callback causes the state to flicker and reset before a real fall sequence can complete. Locking into a state for a bounded window (e.g. 3 seconds waiting for stillness after impact) fixed this.

## The Interesting Bug: BLE Bandwidth Collision

The glasses stream IMU data continuously over BLE. When `FALL_CONFIRMED` triggers, the natural next step is to call the glasses camera's `getPhoto()` — also over BLE. In testing, this request would intermittently fail.

The root cause: a single JPEG photo payload is roughly 80 KB. Under a tight 1-second timeout (kept short deliberately, for responsiveness), that's not enough time to push the photo through a BLE channel that's simultaneously carrying continuous sensor telemetry. The bus was saturated.

The obvious fix — extend the timeout — works, but not cleanly. Pushing it to 10 seconds resolved the failures, but a 10-second wait during a fall-confirmed sequence is unacceptable; the whole point of the system is fast, reliable alerting. Narrowing it to 3–4 seconds struck the right balance between giving the BLE channel breathing room and keeping the pipeline responsive.

## Designing Around Photo Capture Failure

Rather than treating BLE contention as something to eliminate entirely, I designed the photo capture step to degrade gracefully. During `MONITORING`, the phone camera continuously captures a frame into a rolling buffer every 1.5 seconds — a lock-free ring buffer, so writes never block the sensor thread.

When a fall is confirmed, three photo sources are tried **in strict priority order, mutually exclusive** — the first one that succeeds wins, and the pipeline moves on immediately rather than waiting to attempt the others:

1. **Glasses camera** — if it succeeds, this is the best source: a first-person view of the ground from the user's actual fall position.
2. **Ring buffer** — if the glasses camera fails (BLE contention, disconnection), fall back to the most recent buffered frame, wherever the phone happened to be (pocket, table, hand).
3. **Live phone capture** — if the buffer is empty, capture fresh from the phone camera on the spot.

IMU sensing stops once `FALL_CONFIRMED` fires, freeing the BLE channel specifically for the photo attempt, with a 4-second timeout on the whole sequence.

## Testing & Results

I ran structured tests across two fall conditions and one false-positive stress test:

**Detection accuracy:**
- **Hard surface falls:** 87% (13/15) with strict thresholds. The IMU reliably catches the distinct G-force spike of a hard impact.
- **Cushioned falls (bed):** 60% (6/10) with strict thresholds. Soft surfaces absorb much of the impact energy, so the G-force spike is often too small to trigger cleanly on IMU alone. This result is exactly what motivated the VLM layer in the first place — IMU-only detection has a real, measurable blind spot for soft-surface falls.
- **False positive rejection:** 100% (0 false alarms) across rapid daily movements and dropped-glasses events.

**Pipeline performance:**
- **End-to-end latency:** ~16 seconds from physical impact to a delivered, cloud-verified push notification.
- **VLM scene reasoning accuracy:** 100% — whenever a frame was successfully captured, Gemini correctly confirmed the fall from the first-person camera perspective (floor/ceiling/furniture-leg views, not a third-person shot of someone lying down).
- **Ring buffer capture rate:** 100% — the phone-camera fallback always had *something* to hand off, even when the glasses camera failed.
- **Cold start:** the first trigger after app launch showed extra latency (one-time initialization cost); every subsequent trigger held the deterministic ~16s.

## What I'd Do Differently

The cushioned-fall number is the most honest result in this whole project. It would have been easy to tune the IMU threshold aggressively low to inflate that number, but that trades away the false-positive rate — and a system elderly users stop trusting is worse than one that occasionally misses a soft fall and lets the VLM layer catch it downstream. I kept the threshold strict and let the architecture, not the sensor, absorb the hard case.

The BLE collision also taught me something about designing for degraded conditions rather than assuming the ideal path always works. The three-tier photo fallback wasn't in the original design — it emerged directly from watching the glasses camera fail under real sensor load. In hindsight, any system pulling from a shared, bandwidth-constrained radio channel while also trying to do something time-sensitive should probably be designed with a fallback from day one.
