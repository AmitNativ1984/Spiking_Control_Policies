# Cluster Training Setup

Run navigation VAE training on a SLURM cluster with Pyxis (DGX A100).

## 1. Build & prepare the Docker image

On a machine with Docker:

```bash
cd aerial_gym_docker
docker build -f Dockerfile.base -t aerial-gym:latest .
```

**Option A — Push to a registry** (if your cluster can pull):

```bash
docker tag aerial-gym:latest <registry>/aerial-gym:latest
docker push <registry>/aerial-gym:latest
```

Then set `CONTAINER_IMAGE=<registry>/aerial-gym:latest` when submitting.

**Option B — Build a .sqsh on the cluster** (if no registry):

```bash
# On the cluster (requires enroot)
enroot import dockerd://aerial-gym:latest
# Creates aerial-gym+latest.sqsh
```

Then set `CONTAINER_IMAGE=./aerial-gym+latest.sqsh` when submitting.

## 2. Set up Weights & Biases

```bash
# Get your API key from https://wandb.ai/authorize
echo "WANDB_API_KEY=<your-key>" > slurm/.env
```

The sbatch script sources this file automatically. Do not commit it to git.

## 3. Interactive test

Verify the container works on the cluster before submitting a job:

```bash
srun --partition=dlc --gres=gpu:1 --mem=32G \
    --container-image=aerial-gym:latest \
    --container-mounts=$HOME/aerial_gym_docker:/workspaces/aerial_gym_docker \
    --container-workdir=/workspaces/aerial_gym_docker \
    --pty bash
```

Inside the container:

```bash
python -c "from isaacgym import gymapi; print('Isaac Gym OK')"
python -c "import aerial_gym; print('Aerial Gym OK')"
```

## 4. Submit training

```bash
cd aerial_gym_docker
sbatch slurm/train_navigation.sbatch
```

Override defaults:

```bash
NUM_ENVS=4096 MAX_EPOCHS=1000 sbatch slurm/train_navigation.sbatch
```

## 5. Monitor

**W&B dashboard:** Real-time metrics at https://wandb.ai (project: `aerial_gym`)

**SLURM logs:**

```bash
tail -f slurm_logs/nav-vae_<JOBID>.out
```

**Checkpoints:**

```bash
ls navigation_with_obstacles/runs/*/nn/*.pth
```

## Cluster config vs laptop config

| Parameter | Laptop (`ppo_navigation.yaml`) | Cluster (`ppo_navigation_cluster.yaml`) |
|-----------|------|---------|
| num_envs | 1024 | 8192 |
| horizon_length | 64 | 128 |
| minibatch_size | 4096 | 32768 |
| learning_rate | 1e-4 | 3e-4 |
| max_epochs | 1000 | 2000 |
| save_frequency | 50 | 100 |

Physics/simulation parameters are identical between both configs.
