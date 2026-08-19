"""Turn a trained checkpoint into the artifacts the flight stack consumes.

THIS IS THE TORCH SIDE OF THE FENCE. The flight repo (sail-uav-core) contains no torch at
all: it builds the observation in numpy and runs an exported ONNX graph. Reading a .pth is
work that belongs here, in the container that already has torch, Isaac Gym and the runs.

The boundary is not tidiness. The two environments cannot run the same torch -- this one is
Python 3.8 with an NVIDIA NGC build that was never released to any index, the flight image is
Python 3.12 whose floor is torch 2.2 -- so a checkpoint interpreted on the vehicle would be
interpreted by a different torch than trained it. An exported graph is a frozen description
of the arithmetic and raises no such question.

WHAT TO RUN
-----------
    python -m deploy.export_onnx   --checkpoint runs/.../last_....pth --output <flight>/hover.onnx
    python -m deploy.record_golden --checkpoint runs/.../last_....pth --output <flight>/hover_golden.npz

which produce the three artifacts sail-uav-core consumes:

    hover.onnx         the network graph
    hover.npz          its frozen input-normalization statistics (written beside the graph)
    hover_golden.npz   observation/action pairs, so the flight container can prove it
                       reproduces this environment's numbers

Both import control_policy_api for the observation layout, so the deployment package and the
training container agree on the vector by construction rather than by transcription. Install
it editable from the monorepo:

    pip install -e <sail-uav-core>/libs/frame-transforms -e <sail-uav-core>/libs/control-policy-api
"""
