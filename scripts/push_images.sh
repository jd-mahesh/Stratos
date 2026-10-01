#!/usr/bin/env bash
# Build the images, push them to ECR, and point existing Lambda functions at the new version.
#
#   ./scripts/push_images.sh              # all three
#   ./scripts/push_images.sh live-trader  # just one: backtester | live-trader | dashboard
#
# Needs: AWS CLI v2 configured (aws configure), Docker running, repos created (see docs/DEPLOY.md).
set -euo pipefail

cd "$(dirname "$0")/.."

REGION="${AWS_REGION:-us-east-1}"
ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
REGISTRY="$ACCOUNT.dkr.ecr.$REGION.amazonaws.com"
# A unique tag per build, so Lambda sees a new image (git commit if available, else a timestamp).
TAG="${TAG:-$(git rev-parse --short HEAD 2>/dev/null || date +%Y%m%d%H%M%S)}"

# Lambda functions are created as arm64 (Graviton: cheaper, and native on Apple Silicon Macs).
# Fargate tasks default to x86_64, so the dashboard is built for amd64 (emulated on an M-series Mac).
LAMBDA_PLATFORM="${LAMBDA_PLATFORM:-linux/arm64}"
DASHBOARD_PLATFORM="${DASHBOARD_PLATFORM:-linux/amd64}"

echo "Logging in to $REGISTRY"
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$REGISTRY"

build_and_push() {
  local name="$1" dockerfile="$2" platform="$3"
  local repo="$REGISTRY/stratos-$name"
  echo "Building $name for $platform -> $repo:$TAG"
  # --provenance=false: Lambda rejects the multi-manifest "image index" that buildx
  # produces by default when it attaches provenance attestations.
  docker build --platform "$platform" --provenance=false \
    -f "$dockerfile" -t "$repo:$TAG" -t "$repo:latest" .
  docker push "$repo:$TAG"
  docker push "$repo:latest"
}

update_lambda() {
  local name="$1" fn="stratos-$1"
  if aws lambda get-function --function-name "$fn" --region "$REGION" >/dev/null 2>&1; then
    aws lambda update-function-code --function-name "$fn" --region "$REGION" \
      --image-uri "$REGISTRY/stratos-$name:$TAG" >/dev/null
    echo "Lambda $fn now runs :$TAG"
  else
    echo "Lambda $fn doesn't exist yet; create it with the command in docs/DEPLOY.md"
  fi
}

# (plain if-statements: macOS ships bash 3.2, which lacks newer case syntax)
target="${1:-all}"
case "$target" in
  backtester|live-trader|dashboard|all) ;;
  *) echo "unknown target: $target (use backtester, live-trader, dashboard or all)"; exit 1 ;;
esac

if [ "$target" = backtester ] || [ "$target" = all ]; then
  build_and_push backtester docker/backtester.Dockerfile "$LAMBDA_PLATFORM"
  update_lambda backtester
fi
if [ "$target" = live-trader ] || [ "$target" = all ]; then
  build_and_push live-trader docker/live_trader.Dockerfile "$LAMBDA_PLATFORM"
  update_lambda live-trader
fi
if [ "$target" = dashboard ] || [ "$target" = all ]; then
  build_and_push dashboard docker/dashboard.Dockerfile "$DASHBOARD_PLATFORM"
  echo "Dashboard pushed as :$TAG. Point the ECS Express service at it (docs/DEPLOY.md, step 8)."
fi
