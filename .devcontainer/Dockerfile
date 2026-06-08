# Devcontainer Dockerfile
# Builds on top of the pre-built aerial-gym image and adds dev dependencies

FROM localhost:5000/aerial-gym:latest

# Install dev dependencies.
# NOTE: loguru and snntorch are already installed in the base image
# (aerial-gym:latest). snntorch in particular is pinned there to ==0.9.4 via
# --ignore-requires-python (the 0.9.4 Synaptic CUBA-LIF fix doesn't ship a
# Python 3.8 wheel tag). Do NOT `pip install snntorch` here unpinned — on 3.8
# that can downgrade it back to the buggy 0.9.1. Keep this layer for any
# dev-only extras instead.
RUN pip install \
    loguru

# Create non-root user matching host UID/GID
ARG USERNAME=devuser
ARG USER_UID=1000
ARG USER_GID=${USER_UID}

RUN groupadd --gid ${USER_GID} ${USERNAME} 2>/dev/null || true \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} \
    && apt-get update && apt-get install -y sudo \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME} \
    && rm -rf /var/lib/apt/lists/*

# Give the user access to key directories
RUN chown -R ${USER_UID}:${USER_GID} /app \
    && chown -R ${USER_UID}:${USER_GID} /opt/isaacgym

USER ${USERNAME}

# Set working directory to the aerial gym simulator
WORKDIR /app/aerial_gym/aerial_gym_simulator

CMD ["bash"]
