---
title: "Beyond the IMU: Designing a Fault-Tolerant Fall Detection System for Smart Glasses"
date: 2026-06-26
description: How I built — and later re-architected — an end-to-end fall detection pipeline for Solos smart glasses combining IMU state machines, a lock-free ring buffer, VLM visual confirmation, and a three-tier photo fallback, eventually replacing an early BLE bandwidth workaround with a proper GATT queue contention fix that cut latency from 16s to under 5s.
tags: [Edge AI, Android, BLE, VLM, Systems Design, Kotlin]
---

## TL;DR

Over two weeks of after-hours development, I built an end-to-end fall detection system for Solos smart glasses: IMU-based impact detection, a 30-second cancel window, VLM-based visual confirmation via Gemini, and automated alerting through n8n. The most interesting engineering problem wasn't the AI — it was a BLE bandwidth collision between sensor streaming and photo capture, later traced to GATT queue contention and fixed by stopping the sensors before requesting a photo, cutting end-to-end latency from ~16s to under 5s. The system won highest score at the internal hackathon.

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

The glasses stream IMU data continuously over BLE. When `FALL_CONFIRMED` triggers, the natural next step is to call the glasses camera's `getPhoto()` — also over BLE. In testing, this request would intermittently fail or take far too long to complete.

The root cause: the accelerometer stream keeps the BLE GATT queue constantly busy. `getPhoto()`'s write command has to queue up behind that continuous sensor traffic — even with a stable WiFi connection on the other end, the photo request itself couldn't get a word in on the BLE side. A colleague's initial workaround was extending the timeout to 10 seconds, which "fixed" it, but that was really just waiting long enough for the BLE queue to drain naturally — not a real fix.

The actual fix was smaller and more obvious in hindsight: **stop the sensors first.** The moment `FALL_CONFIRMED` fires, the pipeline calls `accelerometer.stop()` and `absoluteOrientation.stop()` before attempting the photo. That instantly clears the BLE queue, so `getPhoto()`'s GATT write goes out immediately instead of queuing behind sensor telemetry. Lowering the photo resolution (960×720 instead of the original) shaved off a little more time on the WiFi transfer side, but that was a minor contribution compared to clearing the queue.

The result: end-to-end latency dropped from ~16 seconds to consistently under 5 seconds (measured 4.3–4.7s across repeated trials) — with the glasses camera now succeeding directly, no ring-buffer fallback needed, as long as WiFi is stable.

## Designing Around Photo Capture Failure

The three-tier fallback below was built *before* the BLE queue fix, when glasses-camera photo capture was unreliable under sensor load. It's now mostly a safety net for WiFi instability rather than the primary path — but it's worth documenting because it's a good example of designing for degradation rather than assuming the ideal path always works.

During `MONITORING`, the phone camera continuously captures a frame into a rolling buffer every 1.5 seconds — a lock-free ring buffer, so writes never block the sensor thread.

When a fall is confirmed, three photo sources are tried **in strict priority order, mutually exclusive** — the first one that succeeds wins, and the pipeline moves on immediately rather than waiting to attempt the others:

1. **Glasses camera** — now the reliable default post-fix: a first-person view of the ground from the user's actual fall position, delivered in under 5 seconds.
2. **Ring buffer** — if the glasses camera fails (e.g. WiFi drop), fall back to the most recent buffered frame, wherever the phone happened to be (pocket, table, hand).
3. **Live phone capture** — if the buffer is empty, capture fresh from the phone camera on the spot.

IMU sensing stops once `FALL_CONFIRMED` fires, freeing the BLE channel specifically for the photo attempt, with a 4-second timeout on the whole sequence.

## Testing & Results

I ran structured tests across two fall conditions and one false-positive stress test:

**Detection accuracy:**
- **Hard surface falls:** 87% (13/15) with strict thresholds. The IMU reliably catches the distinct G-force spike of a hard impact.
- **Cushioned falls (bed):** 60% (6/10) with strict thresholds. Soft surfaces absorb much of the impact energy, so the G-force spike is often too small to trigger cleanly on IMU alone. This result is exactly what motivated the VLM layer in the first place — IMU-only detection has a real, measurable blind spot for soft-surface falls.
- **False positive rejection:** 100% (0 false alarms) across rapid daily movements and dropped-glasses events.

**Pipeline performance:**
- **End-to-end latency:** originally ~16 seconds; after identifying and fixing the BLE queue contention (stopping sensors before requesting a photo), this dropped to consistently under 5 seconds (4.3–4.7s across repeated trials).
- **VLM scene reasoning accuracy:** 100% — whenever a frame was successfully captured, Gemini correctly confirmed the fall from the first-person camera perspective (floor/ceiling/furniture-leg views, not a third-person shot of someone lying down).
- **Ring buffer capture rate:** 100% as a fallback — always had *something* to hand off even in early testing when the glasses camera path was unreliable.
- **Cold start:** the first trigger after app launch showed extra latency (one-time initialization cost); every subsequent trigger held the deterministic latency.

## What I'd Do Differently

The cushioned-fall number is the most honest result in this whole project. It would have been easy to tune the IMU threshold aggressively low to inflate that number, but that trades away the false-positive rate — and a system elderly users stop trusting is worse than one that occasionally misses a soft fall and lets the VLM layer catch it downstream. I kept the threshold strict and let the architecture, not the sensor, absorb the hard case.

The BLE latency fix taught me a more general lesson about distinguishing a *workaround* from a *fix*. Extending the timeout to 10 seconds made the symptom go away, but it was treating the queue drain time as a fixed cost to wait out rather than asking why the queue was full in the first place. Once I stopped the sensors before requesting the photo, the "10-second problem" turned out to not really exist — it was a side effect of two BLE consumers competing for the same channel, not a fundamental limit on how fast a photo could be requested. The three-tier fallback design (built to survive the original unreliable path) is still in the system, but it's now a safety net for WiFi issues rather than the thing doing the heavy lifting every time.
