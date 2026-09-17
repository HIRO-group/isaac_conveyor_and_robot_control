# syntax=docker/dockerfile:1
# Isaac Sim image running this simulator headless.
#   docker build -t conveyor-sim:<tag> --build-arg SIM_GIT_SHA=$(git rev-parse --short=12 HEAD) .
# Run with a GPU, a scene package mounted at $SIM_SCENE_DIR and ZENOH_ROUTER set.
# Headed by default (needs DISPLAY and the X socket); CONVEYOR_INDEXING_HEADLESS=1 for headless.
ARG ISAAC_SIM_IMAGE=nvcr.io/nvidia/isaac-sim:6.0.1
FROM ${ISAAC_SIM_IMAGE}

ARG SIM_GIT_SHA=unknown
ENV ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ALLOW_ROOT=1 \
    SIM_GIT_SHA=${SIM_GIT_SHA} \
    SIM_HOME=/opt/sim \
    PROTO_OUT=/opt/sim/proto_gen \
    PYTHONPATH=/opt/sim/src:/opt/sim/proto_gen \
    CONVEYOR_INDEXING_DATA_DIR=/data \
    SIM_ASSET_DIR=/assets
USER root
WORKDIR ${SIM_HOME}

COPY requirements.txt ./
RUN /isaac-sim/python.sh -m pip install --no-cache-dir -r requirements.txt

COPY proto ./proto
COPY gen_proto.sh ./
RUN PROTOC="/isaac-sim/python.sh -m grpc_tools.protoc" bash gen_proto.sh

COPY src ./src
COPY scripts ./scripts
COPY robot_configs ./robot_configs

HEALTHCHECK --interval=15s --timeout=10s --start-period=300s --retries=5 \
  CMD ["/isaac-sim/python.sh", "/opt/sim/scripts/wait_for_clock.py", "--timeout", "8"]

ENTRYPOINT ["/isaac-sim/python.sh", "/opt/sim/scripts/run_conveyor_indexing.py"]
