# Cluster Training Setup

Run F450 navigation training (`rl_training.rl_games.runner`) on a SLURM cluster
with Pyxis (DGX A100).

## 1. Build & prepare the Docker image

On a machine with Docker:

```bash
cd aerial_gym_docker
docker build -f Dockerfile.base -t aerial-gym:latest .
# Pull from Docker Hub and build the squashfs image
enroot import -o aerial-gym.sqsh docker://YOUR_DOCKERHUB_USER/aerial-gym:latest
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

Submit from the repo root: `SLURM_SUBMIT_DIR` is what the job bind-mounts into the container.

Override defaults (env count and epoch budget live in the YAML — pick a `_cluster` config
rather than passing them here):

```bash
CONFIG_FILE=rl_training/rl_games/cfg/popsan_cluster.yaml \
EXPERIMENT=f450_nav_snn \
CURRICULUM_LEVEL=25 \
sbatch slurm/train_navigation.sbatch
```

Full list of variables: `CONFIG_FILE`, `TASK`, `EXPERIMENT`, `WANDB_PROJECT`, `CHECKPOINT`,
`NUM_ENVS`, `SEED`, `CURRICULUM_LEVEL`, `CONTAINER_IMAGE`, `EXTRA_ARGS`. Config matrix and
runner flags: `rl_training/README.md`.

The legacy `navigation_with_obstacles` tree has its own scripts under
`navigation_with_obstacles/slurm/`; both read `slurm/.env`.

## 5. Monitor

**W&B dashboard:** Real-time metrics at https://wandb.ai (project: `f450_navigation_task`,
or whatever `WANDB_PROJECT` is set to)

**SLURM logs:**

```bash
tail -f slurm_logs/f450-nav_<JOBID>.out
```

**Checkpoints:**

```bash
ls runs/<EXPERIMENT>/*/nn/*.pth
```

Each run directory also holds `config.yaml`, the resolved config the run actually used —
play it back with that file, not with `cfg/`.
