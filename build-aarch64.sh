#!/usr/bin/env bash
# Cross-build the engine for aarch64 handhelds (H700: Cortex-A53) in the ReNFS5 Rust cross image.
#   ./build-aarch64.sh            -> target/aarch64-unknown-linux-gnu/release/open-unleashed
set -euo pipefail
cd "$(dirname "$0")"
image=${CROSS_IMAGE:-nfs5-rust-cross:bookworm}
export DOCKER_CONFIG=${DOCKER_CONFIG:-$HOME/repos/renfs5/work/docker-config}
docker run --rm -u "$(id -u):$(id -g)" \
  -v "$PWD":/src -w /src \
  -v "$HOME/.cargo/registry":/opt/cargo/registry \
  -e CARGO_HOME=/opt/cargo \
  -e RUSTFLAGS="-C target-cpu=${TARGET_CPU:-cortex-a53}" \
  "$image" cargo build --release --locked -p open-unleashed --target aarch64-unknown-linux-gnu "$@"
