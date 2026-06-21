# Hardware Dependency Map — Aerial Gym (SNN drone)

**Committed deployment stack:** spiking policy on **Intel Loihi 2**, orchestrated by an
**NVIDIA Jetson Orin** companion computer, with an **Intel RealSense** depth camera and a
**Prophesee** event camera for perception / data collection.

The project has **two architecturally distinct phases**:

- **Phase A — Train & collect data** → `x86_64` host + **NVIDIA CUDA GPU** (Isaac Gym sim, VAE + SNN training).
- **Phase B — Deploy** → **ARM64** drone compute (Jetson Orin) + **Loihi 2** neuromorphic chip.

> The two phases run on **different CPU architectures and different vendors' silicon**.
> Nothing from the NVIDIA *simulation* stack flies; only trained weights cross the boundary.

---

## 1. Hardware dependency map

### 1a. Phase A — Training & data collection (ground / lab, x86_64)

| Hardware | Role | Key constraint |
|---|---|---|
| **x86_64 workstation/server** | Runs Docker image, Isaac Gym sim, VAE + SNN training | Isaac Gym bindings are `linux-x86_64` only — **no ARM/Apple Silicon** |
| **NVIDIA CUDA GPU** (render-capable, ≥8 GB, Pascal+) | GPU physics (PhysX), depth rendering, NN training | VRAM caps parallel envs: 8 GB → `num_envs≈256`; needs `graphics,display` + `/dev/dri` (Vulkan/EGL) |
| **Intel RealSense D435/D435i** | Collect *real* depth images to supplement the simulated VAE dataset | USB 3.0; specs already mirrored in `data_generation/config/camera_config.py` (1280×720, 87° HFOV, 0.105–10 m) |
| **Prophesee event camera** (EVK4 / IMX636) | Collect/validate event streams for the spiking front-end | USB 3.0; output is CD events (x, y, polarity, µs timestamp), not frames |
| **Intel Loihi 2** *(optional in Phase A)* | Hardware-in-the-loop SNN validation before flight | Via **INRC cloud** or a **Kapoho Point** board; access gated by Intel INRC |

### 1b. Phase B — Deployment (onboard the drone)

| Hardware | Role | Interface |
|---|---|---|
| **NVIDIA Jetson Orin** (Nano/NX/AGX, ARM64) | Companion computer: sensor drivers, **VAE inference (TensorRT)**, orchestration, FC bridge, host for Loihi | Carrier board; USB3 + UART/Ethernet/PCIe |
| **Intel Loihi 2** (Kapoho Point / Oheo Gulch) | Runs the spiking **PopSAN** policy at low power | Connects to host over **Ethernet/USB**; runtime hosted from Jetson (⚠ see §5) |
| **Intel RealSense D435** | Onboard depth → Depth VAE encoder | USB 3.0 → Jetson |
| **Prophesee event camera** | Onboard events → SNN input (native spiking modality) | USB 3.0 → Jetson |
| **Flight controller** (PX4 / ArduPilot) | Receives Lee/attitude setpoints from the policy | UART / MAVLink (or ROS 2) ← Jetson |
| **Drone airframe + power** | Carries SWaP of Jetson + Loihi board + 2 cameras | Power/weight budget is a real constraint (Loihi boards are dev hardware) |

---

## 2. SDKs & toolchains (per hardware)

| Hardware / layer | SDK / toolchain | License / access | Arch | Purpose |
|---|---|---|---|---|
| **Intel Loihi 2** | **Lava** + **Lava-DL** (SLAYER/Bootstrap, `netx`); legacy **NxSDK** | Lava: open (BSD-3 / LGPL-2.1); **chip + NxSDK gated via INRC** | host CPU | Convert `snntorch` net → Lava process, deploy/run on chip |
| **Jetson Orin** | **NVIDIA JetPack** = L4T (Ubuntu BSP) + **CUDA** + **cuDNN** + **TensorRT** + VPI | **NVIDIA proprietary** (free, EULA) | **aarch64** | OS + GPU-accelerated VAE inference |
| **Intel RealSense** | **Intel RealSense SDK 2.0** = `librealsense2` + `pyrealsense2` | **Apache-2.0** (open) | x86_64 + aarch64 | Capture/stream depth (dataset + onboard) |
| **Prophesee event cam** | **Metavision SDK** (core = **OpenEB**) | OpenEB **Apache-2.0**; some Pro/ML modules **commercial-licensed** | x86_64 + aarch64 | Decode/stream events; event→tensor preprocessing |
| **Flight controller** | **MAVSDK** / `pymavlink`, optionally **ROS 2** | open-source | — | Convert policy output → flight setpoints |
| **Training / sim** | **Isaac Gym Preview 4**, PyTorch (NGC), `snntorch==0.9.4`, `rl_games` | Isaac Gym **proprietary**; rest open | x86_64 | Sim, dataset, VAE + SNN training |

---

## 3. Proprietary & gated tools

| Tool | Vendor | Gate / restriction | Impact |
|---|---|---|---|
| **Isaac Gym Preview 4** | NVIDIA | Account-gated download, proprietary EULA, **not redistributable**, x86_64-only, **deprecated** | Hard lock-in for Phase A; caps project at **Python <3.9** |
| **Loihi 2 silicon + INRC access** | Intel | **Requires Intel Neuromorphic Research Community membership**; hardware not on open market | Phase B blocked without INRC; programmatic dependency on Intel's research program |
| **NxSDK** (if used over Lava) | Intel | Gated under INRC | Avoid if possible — prefer open Lava |
| **JetPack / L4T / TensorRT / CUDA** | NVIDIA | Proprietary license (free), Jetson-locked | Standard NVIDIA edge lock-in; ties inference to NVIDIA edge GPU |
| **NGC base image** `nvcr.io/nvidia/pytorch:22.12-py3` | NVIDIA | Registry-gated | Pins CUDA/torch for training |
| **Metavision Pro / ML modules** | Prophesee | Commercial license for advanced features (core OpenEB is free) | Fine if you stay within OpenEB |
| **Device firmware** (Loihi 2, RealSense, Prophesee, Jetson) | resp. vendors | Closed firmware | SDKs are open, silicon/firmware are not |

---

## 4. Data & model flow (train → export → deploy)

```
PHASE A — TRAIN (x86_64 + NVIDIA CUDA GPU)
  Isaac Gym sim  ─► simulated depth ─┐
  RealSense D435 ─► real depth      ─┼─► Depth VAE (PyTorch)  ─► latent z
  Prophesee EVK  ─► event streams   ─┘                              │
                                                                    ▼
        state + latent z + events ─► PopSAN SNN (snntorch, CUBA-LIF, multi-timestep)
                                                                    │
                                          export (lava-dl / netx, + quantization)
                                                                    ▼
                                                        Loihi 2 executable

PHASE B — DEPLOY (drone: Jetson Orin ARM64 + Loihi 2)
  RealSense ─► librealsense2 ─► VAE encoder (TensorRT on Jetson GPU) ─► z ┐
  Prophesee ─► Metavision ───► event tensor ─────────────────────────────┼─► Loihi 2 (Lava) PopSAN ─► action
  FC state estimate ─────────────────────────────────────────────────────┘                          │
                                                                                                      ▼
                                          action ─► Jetson ─► MAVLink ─► PX4/ArduPilot ─► motors
```

**What crosses the sim→real boundary:** only the **trained VAE weights** and the
**trained SNN parameters** (→ Loihi). All NVIDIA *simulation* binaries (Isaac Gym, PhysX,
FleX, Carbonite) stay on the training host.

---

## 5. Integration risks & open questions

1. **Loihi 2 is research hardware.** Kapoho Point / Oheo Gulch are dev boards, **not
   flight-qualified** — SWaP, vibration, and thermal on an airframe are unproven. Plan a
   tethered / hardware-in-the-loop stage before free flight.
2. **INRC dependency.** Loihi 2 access requires Intel INRC membership; this is a
   programmatic/legal gate, not just a purchase.
3. **⚠ Host-arch question for Loihi 2.** The Loihi runtime host tooling has historically
   been **x86_64 Linux**. Whether the Lava/Loihi runtime is supported with an **aarch64
   Jetson Orin host** must be **verified with Intel** — if not, an x86 single-board host
   (or a different topology) is needed between Jetson and the chip.
4. **`snntorch` → Lava export gap.** No exporter exists in-repo. CUBA-LIF parameter mapping
   (`alpha`/`beta`/thresholds), **fixed-point weight quantization**, and discrete-timestep
   alignment are the main accuracy risks.
5. **Two perception modalities.** Frame depth (RealSense→VAE, on Jetson GPU) and events
   (Prophesee→SNN) need **time synchronization** and a clear split of which signal drives
   the policy — two pipelines to build and calibrate.
6. **Architecture split.** Train = x86_64/NVIDIA; deploy = aarch64/Intel-Loihi. Every deploy
   SDK must have an **aarch64 build** (librealsense2, Metavision, Lava ✓; verify Loihi runtime per #3).

---

## Appendix — Phase A software stack (for completeness)

| Package | Pin | Notes |
|---|---|---|
| `aerial_gym_simulator` | `main` (ntnu-arl) | Drone task/env layer over Isaac Gym |
| `torch`/`torchvision` | from NGC image | Untouched; extras installed `--no-deps` |
| `snntorch` | `==0.9.4` (`--ignore-requires-python --no-deps`) | CUBA-LIF fix; bypasses Py3.9 gate on Py3.8 |
| `rl_games` | aerial_gym dep | PPO runner |
| `opencv-python` | `==4.5.5.64` | Avoids `cv2.dnn.DictValue` crash |
| `wandb`, `loguru` | unpinned | Tracking / logging |

Repo modules: `data_generation/` (depth dataset) → `vae_depth/` (Depth VAE) →
`navigation_with_obstacles/` (PopSAN + VAE, active branch). Cluster training via
`slurm/` (Pyxis/enroot `.sqsh`, 1 GPU + 8 CPU).
