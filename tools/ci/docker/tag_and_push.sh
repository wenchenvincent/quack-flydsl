#!/bin/bash
# Tag local docker images and push to Docker Hub.
# Usage:
#   ./tag_and_push.sh [cu129|cu132]   (default: push both)
#
# Auth:
#   - One-time interactive: `docker login -u tridao` (paste access token).
#   - Unattended: set DOCKERHUB_TOKEN env var, this script logs in for you.
set -e

DATE=$(date +%y.%m.%d)
DOCKERHUB_USER="${DOCKERHUB_USER:-tridao}"
REGISTRY_REPO="${REGISTRY_REPO:-${DOCKERHUB_USER}/quack-kernels}"

if [ -n "${DOCKERHUB_TOKEN:-}" ]; then
    echo "Logging in to Docker Hub as $DOCKERHUB_USER (via DOCKERHUB_TOKEN)..."
    echo "$DOCKERHUB_TOKEN" | docker login -u "$DOCKERHUB_USER" --password-stdin
fi

push_image() {
    local tag=$1
    local local_image="quack-kernels:${tag}-${DATE}"
    local remote_image="${REGISTRY_REPO}:${tag}-${DATE}"
    echo "=== Pushing $remote_image ==="
    docker tag "$local_image" "$remote_image"
    docker push "$remote_image"
}

case "${1:-all}" in
    cu129)  push_image cu12.9 ;;
    cu132)  push_image cu13.2 ;;
    all)    push_image cu12.9
            push_image cu13.2 ;;
    *)      echo "Unknown variant: $1 (expected cu129, cu132, or all)" >&2; exit 1 ;;
esac
