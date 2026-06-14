# Aerial Gym Simulator — Drone RL Tasks

GPU-accelerated reinforcement-learning experiments for aerial robotics, built on
[NVIDIA Isaac Gym Preview 4](https://developer.nvidia.com/isaac-gym) and the
[Aerial Gym Simulator](https://ntnu-arl.github.io/aerial_gym_simulator/), packaged in Docker.

This repository trains a family of quadrotor control policies that grow in difficulty —
from stabilizing a hover with only inertial sensing, up to vision-based navigation through
cluttered environments — and explores both conventional artificial neural networks (ANN/MLP/GRU)
and **spiking neural networks (SNN)** for neuromorphic, energy-efficient control.

---

## Training Tasks

Each task is a self-contained package (`config/`, `task/`, `training/`, optional `networks/`)
with its own README covering theory, training, and inference. They are ordered roughly by
increasing difficulty.

| Task | Folder | What it learns | Sensing | Controller | Policy |
|------|--------|----------------|---------|------------|--------|
| **Simple Hover** | [`simple_hover/`](simple_hover/) | Stabilize and hover in place | IMU + state (no position) | Attitude (Lee) | MLP |
| **Simple Hover (SNN)** | [`simple_hover_snn/`](simple_hover_snn/) | Same hover task, neuromorphic policy | IMU + state | Attitude (Lee) | SNN / PopSAN / MLP baseline |
| **Simple Obstacle Avoidance** | [`simple_obstacle_avoidance/`](simple_obstacle_avoidance/) | Fly to a waypoint past a few fixed obstacles | Depth camera (VAE) + state | Attitude (Lee) | MLP |
| **Navigation with Obstacles** | [`navigation_with_obstacles/`](navigation_with_obstacles/) | Curriculum navigation through dense clutter | Depth camera (DepthVAE) + state | Attitude (Lee) | PopSAN (SNN) / MLP / GRU |

### Supporting components

| Component | Folder | Purpose |
|-----------|--------|---------|
| **Depth VAE (DCE)** | [`vae_depth/`](vae_depth/) | Variational autoencoder that compresses depth images into a 32-D latent with implicit collision encoding, used by the navigation task. |
| **Dataset Generation** | [`data_generation/`](data_generation/) | Generates the depth-image dataset (Intel RealSense D435 simulation) used to train the Depth VAE. |

> All tasks use [`rl_games`](https://github.com/Denys88/rl_games) PPO and the Aerial Gym
> physics/rendering stack. Each task folder's README documents its observation/action spaces,
> reward function, network architecture, and exact training/inference commands.

---

## Quick Start (training a task)

Once the Docker image is built and you are inside the container (see the setup guide below):

```bash
cd /workspaces/aerial_gym_docker

# Example: train the simple hover task
python simple_hover/training/runner.py \
    --file=simple_hover/training/ppo_hover.yaml \
    --train --num_envs=1024 --headless=True
```

See each task's README for the full set of configs and inference commands.

---

# Docker Setup Guide

A complete step-by-step guide to running Aerial Gym Simulator in Docker (without conda/anaconda).

---

## Prerequisites

## Step 1: Install NVIDIA Container Toolkit

```bash
# Add NVIDIA repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify GPU access in Docker
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

---

## Step 2: Download Isaac Gym (Manual Step - Required)

Isaac Gym cannot be downloaded automatically. You must:

1. Go to: https://developer.nvidia.com/isaac-gym
2. Sign in with NVIDIA account (free)
3. Click "Join now" to get access
4. Download **Isaac Gym Preview 4** (`IsaacGym_Preview_4_Package.tar.gz`)
5. Save it to a known location, e.g., `~/Downloads/`

---

## Step 3: Create Project Directory Structure

```bash
# Create project directory
mkdir -p ~/aerial_gym_docker
cd ~/aerial_gym_docker

# Copy Isaac Gym package here
cp ~/Downloads/IsaacGym_Preview_4_Package.tar.gz .

# Extract it
tar -xzf IsaacGym_Preview_4_Package.tar.gz
```

---

## Step 4: Create the Dockerfile

The Dockerfile is already provided in the repository. It includes critical fixes for IsaacGym rendering:

**Key features:**
- **Mesa EGL workaround**: Fixes segmentation fault during rendering by removing Mesa's buggy EGL implementation
- **NVIDIA vendor configs**: Ensures NVIDIA's stable EGL and Vulkan are used
- **OpenCV fix**: Uses compatible OpenCV version (4.5.5.64) to avoid `cv2.dnn.DictValue` errors
- **Multi-stage build**: Optimized for both IsaacGym and Aerial Gym


**Important fixes included:**
1. Mesa EGL removal prevents "Segmentation fault (core dumped)"
2. NVIDIA vendor configs ensure proper GPU rendering
3. OpenCV 4.5.5.64 fixes import errors

---

## Step 5: Docker Compose Configuration

The `docker-compose.yml` is already provided with two services:

**1. aerial-gym** (headless - for training)
```yaml
aerial-gym:
  build:
    context: .
    dockerfile: Dockerfile
  image: aerial-gym:latest
  runtime: nvidia
  environment:
    - NVIDIA_VISIBLE_DEVICES=all
    - NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
  volumes:
    - ./outputs:/workspace/aerial_gym_simulator/outputs
    - ./logs:/workspace/aerial_gym_simulator/logs
```

**2. aerial-gym-gui** (with GUI support - for visualization)
```yaml
aerial-gym-gui:
  build:
    context: .
    dockerfile: Dockerfile
  image: aerial-gym:latest
  runtime: nvidia
  environment:
    - NVIDIA_VISIBLE_DEVICES=all
    - NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display  # Note: includes 'display'
    - DISPLAY=${DISPLAY:-:0}
    - QT_X11_NO_MITSHM=1  # Prevents Qt shared memory issues
  volumes:
    - /tmp/.X11-unix:/tmp/.X11-unix:rw
    - ${HOME}/.Xauthority:/root/.Xauthority:ro  # Note: read-only for security
  devices:
    - /dev/dri:/dev/dri  # Direct rendering device
  network_mode: host
```

**Key differences from basic setup:**
- `display` capability added to NVIDIA_DRIVER_CAPABILITIES
- QT_X11_NO_MITSHM prevents Docker Qt issues
- .Xauthority mounted as read-only (security best practice)
- /dev/dri device for GPU rendering access

---

## Step 6: Build the Docker Image

```bash
cd ~/aerial_gym_docker

# Using Docker directly
docker build -t aerial-gym:latest .

# OR using Docker Compose
docker compose build
```

**Note**: The build process takes 10-20 minutes depending on your internet speed.

---

## Step 6: Run the Container

### Option A: Headless Mode (Recommended for Training)

```bash
# Using Docker directly
docker run --gpus all -it --rm \
    -v $(pwd)/outputs:/workspace/aerial_gym_simulator/outputs \
    -v $(pwd)/logs:/workspace/aerial_gym_simulator/logs \
    aerial-gym:latest

# OR using Docker Compose
docker compose run aerial-gym
```

### Option B: With GUI Support (for Visualization)

```bash
# Allow X server connections
xhost +local:docker

# Run with display
docker run --gpus all -it --rm \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v $HOME/.Xauthority:/root/.Xauthority:rw \
    -v $(pwd)/outputs:/workspace/aerial_gym_simulator/outputs \
    aerial-gym:latest
```

---

## Step 7: Test the Installation

Once inside the container:

```bash
# Test Isaac Gym
cd /opt/isaacgym/python/examples
python joint_monkey.py --headless

# Test Aerial Gym - Position control task
cd /workspace/aerial_gym_simulator
python examples/position_control_example.py --headless

# Run RL training example
python aerial_gym/rl_training/rl_games/runner.py \
    --file=./ppo_aerial_quad_position.yaml \
    --num_envs=512 \
    --headless=True
```

---

## Step 8: Training with Depth Images for Navigation

For your specific use case (depth-based navigation):

```bash
# Inside the container
cd /workspace/aerial_gym_simulator

# Train navigation policy with depth
python aerial_gym/rl_training/rl_games/runner.py \
    --file=./ppo_aerial_quad_navigation.yaml \
    --num_envs=256 \
    --headless=True
```

## Useful Commands

### Enter a Running Container
```bash
docker exec -it aerial-gym /bin/bash
```

### Copy Files Out of Container
```bash
docker cp aerial-gym:/workspace/aerial_gym_simulator/outputs/model.pth ./
```

### Stop Container
```bash
docker stop aerial-gym
```

### View Logs
```bash
docker logs aerial-gym
```

### Rebuild After Changes
```bash
docker compose build --no-cache
```


## Troubleshooting

### GPU Not Detected
```bash
# Check NVIDIA runtime
docker info | grep -i runtime

# Ensure nvidia-container-toolkit is installed
nvidia-ctk --version
```

### Display/GUI Issues
```bash
# On host
xhost +local:docker
export DISPLAY=:0
```

### Out of Memory (RTX 3070 - 8GB VRAM)
Reduce the number of parallel environments:
```bash
python runner.py --num_envs=256 --headless=True
```

### Permission Denied on Volumes
```bash
# Fix ownership
sudo chown -R $USER:$USER ~/aerial_gym_docker/outputs
```
