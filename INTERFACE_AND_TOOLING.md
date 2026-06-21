# Interface & Tooling Requirements — Edge Deployment

Runtime engines, API/interchange definitions, custom toolchains, compiler requirements,
and integration frameworks for getting this project's trained models onto the edge.

**Target stack (committed):** NVIDIA **Jetson Orin** (companion, ARM64) + Intel **Loihi 2**
(neuromorphic), with **RealSense** depth and **Prophesee** event cameras.

---

## 1. Executive summary

This project produces **two model artifacts with fundamentally different execution semantics**,
so it needs **two separate runtime/toolchain stacks** — there is no single edge runtime that
covers both.

| Artifact | Nature | Where it runs | Runtime path |
|---|---|---|---|
| **Depth VAE encoder** (`vae_depth/model.py`) | Conventional **CNN**, static feed-forward graph | Jetson Orin **GPU** | PyTorch → **ONNX** → **TensorRT** (or ONNX Runtime) |
| **PopSAN policy** (`navigation_with_obstacles/networks/snn/`) | **Spiking**, stateful, multi-timestep CUBA-LIF | **Loihi 2** | snntorch → **NIR** → **Lava** → Loihi |

> **The key finding for "popular edge runtimes":** TensorFlow Lite Micro, ONNX Runtime Mobile,
> and TFLite handle the **VAE** fine, but **none of them can represent the spiking PopSAN** —
> they are static-tensor, feed-forward graph runtimes with no concept of membrane state,
> firing thresholds, reset dynamics, or a discrete time loop. The SNN requires a
> **neuromorphic toolchain (NIR + Lava)**, which is a different software universe.

---

## 2. The two model artifacts (what we actually have to ship)

### 2a. Depth VAE encoder — `vae_depth/model.py : DepthEncoder`
- **Input:** `[B, 1, 180, 320]` single-channel depth.
- **Ops:** 4× (`Conv2d` strided + `BatchNorm2d` + `ELU`), 1×1 conv channel-reduce, flatten, 2× `Linear`.
- **Output:** `[B, 2*latent_dim]`; **inference uses `mu` only, deterministic** (`vae_image_encoder.py`), so the decoder and the sampling head are dropped on-device.
- **Verdict:** 100% standard ops → exports cleanly to ONNX; ideal for TensorRT/ONNX Runtime.

### 2b. PopSAN spiking policy — `pop_spiking_actor.py : PopulationSpikingActorNetwork`
- **Encoder:** population spike encoder (per-dim Gaussian receptive fields → spike trains, `num_steps` long).
- **Body:** 3× (`nn.Linear` → `snn.Synaptic`). **`snn.Synaptic` = 2nd-order CUBA-LIF** (`alpha`=current/synaptic decay, `beta`=membrane decay, threshold, reset) — this **is Loihi 2's native neuron model**.
- **Temporal:** explicit `for t in range(num_steps)` loop (5 steps), stateful (`syn`,`mem`) carried across steps, output spikes accumulated then averaged → `SpikeDecoder`.
- **Training-only bits to strip:** surrogate gradients (`atan`/`sigmoid`), the critic head, rl_games normalization.
- **Verdict:** **not expressible in ONNX/TFLite** — needs a spiking IR (NIR) and a neuromorphic runtime (Lava/Loihi).

---

## 3. Runtime engines

| Engine | Vendor | Runs on | Use in this project | Notes |
|---|---|---|---|---|
| **TensorRT** | NVIDIA | Jetson Orin GPU (aarch64) | **Primary VAE runtime** | Best Jetson perf; consumes ONNX; produces a device/version-specific `.engine`/`.plan` |
| **ONNX Runtime** (+ CUDA/TensorRT Execution Providers) | MS/community | Jetson (aarch64 build) | Alt VAE runtime / quick bring-up | More portable than raw TensorRT; can delegate to TensorRT EP |
| **Lava runtime** (`lava`, magma) | Intel | Host CPU (drives Loihi) + Loihi NeuroCores | **PopSAN runtime** | Process/channel model; compiles graph to NeuroCores; **runtime host historically x86_64 — verify aarch64/Jetson support** |
| **LibTorch / TorchScript** | Meta | Jetson | Fallback for VAE *and* SNN (CPU/GPU, no Loihi) | Lets the SNN run as a plain PyTorch loop if Loihi integration slips — useful for HIL baseline |
| **TFLite Micro** | Google | MCUs (Cortex-M, bare metal) | **Not applicable** | For KB-RAM microcontrollers with no OS; Jetson runs full Linux+CUDA. Wrong tier. |
| **ONNX Runtime Mobile / TFLite** | — | phones / mobile CPU/NPU | **Not the right fit** | Aimed at ARM CPU/mobile-NPU; on Jetson, TensorRT beats them and neither runs the SNN |

---

## 4. API definitions / interchange formats

| Format | Role | This project |
|---|---|---|
| **ONNX** (opset ≥ 13) | NN interchange (CNN/MLP) | **VAE export format.** `torch.onnx.export(DepthEncoder, dummy[1,1,180,320])`. Bridge to TensorRT/ONNX RT |
| **NIR** (Neuromorphic Intermediate Representation) | "ONNX for spiking nets" — defines CubaLIF, LIF, Linear, etc. | **The SNN interchange.** `snntorch` ↔ NIR; NIR → Lava import. The principled `snntorch`→Loihi bridge |
| **Lava `netx` / process graph** | Lava-native model description | Target the SNN compiles into for Loihi |
| **TensorRT engine (`.plan`)** | Compiled, device-locked VAE binary | Build artifact on Jetson (not portable across GPU arch / TRT version) |
| **TorchScript (`.pt`)** | Serialized PyTorch graph | Fallback packaging for VAE/SNN if running under LibTorch |
| **MAVLink messages** | FC command/telemetry API | Policy action → setpoint contract with PX4/ArduPilot |

---

## 5. Custom toolchains (export pipelines)

### 5a. VAE → Jetson (standard, low-risk)
```
DepthEncoder (PyTorch, mu-only)
   └─ torch.onnx.export ──► model.onnx (opset≥13)
        └─ trtexec / TensorRT builder (ON the Jetson) ──► depth_vae.plan
             └─ load via TensorRT C++/Python runtime on Orin GPU
   (alt) model.onnx ──► ONNX Runtime + TensorRT EP
```
Quantization optional: FP16 is the easy Jetson win; INT8 needs a depth-image calibration set.

### 5b. PopSAN → Loihi 2 (custom, the hard part — not in repo)
```
PopulationSpikingActorNetwork (snntorch, snn.Synaptic CUBA-LIF)
   └─ strip critic + surrogate grad + rl_games norm  (build inference-only module)
        └─ export to NIR  (snntorch NIR export; map Synaptic→CubaLIF node)
             └─ NIR → Lava import (lava-dl / nir-lava)
                  └─ Lava compiler ──► fixed-point quantized NeuroCore mapping
                       └─ deploy to Loihi 2 (Lava runtime, INRC)
```
**Risks (all on this path):** Synaptic(2nd-order)↔NIR-CubaLIF parameter fidelity; **fixed-point
weight/threshold quantization** vs. float training; discrete-timestep alignment (`num_steps`);
population-encoder reproduction on/around the chip. **No exporter exists in the repo today.**

### 5c. Fallback (de-risk Loihi): run PopSAN on Jetson GPU
Keep the snntorch loop, run via LibTorch/PyTorch on the Orin GPU. Loses the energy benefit but
unblocks flight tests while the Lava bridge matures. Recommended as an interim milestone.

---

## 6. Compiler / build requirements

| Component | Compiler / builder | Requirement |
|---|---|---|
| **TensorRT engine** | TensorRT builder (`trtexec`) + CUDA toolkit (`nvcc`) | **Build on-target** (Jetson) or on a host with matching TRT + GPU arch — engines are **not portable** across versions/arch |
| **Jetson software** | **JetPack** = L4T (Ubuntu BSP) + CUDA + cuDNN + TensorRT + VPI | aarch64; pin a JetPack version (ties CUDA/TRT versions together) |
| **Loihi mapping** | **Lava compiler** (graph → NeuroCores) + fixed-point quantizer | Runs on the Loihi host; needs INRC SDK access |
| **aarch64 builds** | cross-compile or build-on-device | Every on-drone dep (librealsense2, Metavision, ONNX RT, LibTorch) must have an **aarch64** build |
| **Training image** | NGC CUDA toolchain (`nvcc`, build-essential, cmake, ninja) | Already in `Dockerfile.base`; **x86_64 only** |

> Architecture split to plan around: **train = x86_64**, **deploy = aarch64**. The training
> Docker image cannot produce a runnable Jetson binary directly — only portable artifacts
> (ONNX, NIR, weights) cross over; final compilation (TensorRT engine, Lava mapping) happens
> on the target side.

---

## 7. Integration frameworks (on-drone glue)

| Framework | Role | Interface |
|---|---|---|
| **ROS 2** (rclpy/rclcpp) | Node graph: cameras → VAE → policy → FC bridge | Topics for depth/events/latent/action; aarch64 supported |
| **librealsense2 / pyrealsense2** | RealSense capture node | Depth frames → VAE input tensor |
| **Metavision SDK / OpenEB** | Prophesee capture | Event stream → tensor / direct SNN input |
| **Lava process/channel** | Host↔Loihi data movement | Inject VAE latent + events as input spikes; read action spikes |
| **MAVSDK / pymavlink** | Policy → flight controller | Action tensor → MAVLink setpoints (PX4/ArduPilot) |
| **TensorRT runtime API** | Load/execute VAE engine | C++/Python on Orin |

**Cross-engine boundary to design:** VAE latent `z` is produced by **TensorRT on the Jetson GPU**
but consumed by the **PopSAN on Loihi** — a host→Loihi transfer each control step. Latency of
this hop (plus event preprocessing) is the critical real-time budget.

---

## 8. Edge-runtime suitability matrix

| Runtime | VAE (CNN) | PopSAN (SNN) | Verdict for this project |
|---|---|---|---|
| **TensorRT** | ✅ best on Jetson | ❌ no spiking | **Use for VAE** |
| **ONNX Runtime (+TRT EP)** | ✅ | ❌ | Fine VAE alt / bring-up |
| **ONNX Runtime Mobile** | ⚠️ works but TRT better on Jetson | ❌ | Not needed (mobile-CPU oriented) |
| **TFLite** | ⚠️ possible | ❌ | Not needed |
| **TFLite Micro** | ❌ wrong tier (MCU/bare-metal) | ❌ | **Not applicable** — Jetson is a full Linux SoC |
| **Apache TVM** | ✅ (optional) | ❌ | Only if multi-backend codegen is wanted later |
| **Lava + Loihi** | ❌ overkill | ✅ native CUBA-LIF | **Use for PopSAN** |
| **LibTorch/TorchScript** | ✅ | ✅ (as a loop, no Loihi) | Good HIL fallback for both |

---

## 9. Gaps & recommendations

1. **Build the SNN export bridge first** (snntorch → NIR → Lava). It's the only novel,
   high-risk piece; everything VAE-side is well-trodden.
2. **Confirm Lava/Loihi host support on aarch64 Jetson** (§3, §6). If unsupported, plan an
   x86 co-host or revise topology — this gates the whole deployment.
3. **Adopt ONNX (VAE) + NIR (SNN) as the artifact contract** between training (x86_64) and
   deployment (aarch64). Keep final compilation on-target.
4. **Stand up the LibTorch fallback** (§5c) so flight testing isn't blocked on Loihi integration.
5. **Budget the cross-engine hop** (TensorRT latent → Loihi input) explicitly in the real-time loop.
6. **Pin JetPack + TensorRT + Lava versions** the way the repo already pins `snntorch==0.9.4`
   and `opencv==4.5.5.64` — engine/IR compatibility is version-sensitive.
