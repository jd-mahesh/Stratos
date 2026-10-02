#!/usr/bin/env bash
# Run the dashboard on this computer, reading the cloud database (RDS).
# Free: nothing extra runs on AWS. The trading itself keeps running on Lambda either way.
#
#   bash scripts/dashboard.sh        # then open http://localhost:8502, Ctrl+C to stop
#
# Needs: AWS CLI logged in (aws login --profile stratos), Docker running, and the
# dashboard image built (docker compose build dashboard).
#
# The database URL (which contains the password) is read from Secrets Manager into
# this process only: it's never printed, written to a file, or put on a command line.
set -euo pipefail

PROFILE="${AWS_PROFILE:-stratos}"
SECRET_ID="${SECRET_ID:-stratos/prod}"
PORT="${PORT:-8502}"
IMAGE="${IMAGE:-stratos-dashboard}"
export AWS_PAGER=""

if ! aws sts get-caller-identity --profile "$PROFILE" >/dev/null 2>&1; then
  echo "Not logged in to AWS. Run: aws login --profile $PROFILE" >&2
  exit 1
fi

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "Docker image '$IMAGE' not found. Build it with: docker compose build dashboard" >&2
  exit 1
fi

DATABASE_URL="$(aws secretsmanager get-secret-value --profile "$PROFILE" --secret-id "$SECRET_ID" \
  --query SecretString --output text \
  | python3 -c 'import sys, json; print(json.load(sys.stdin)["DATABASE_URL"])')"
export DATABASE_URL

echo "Dashboard: http://localhost:$PORT  (reading the cloud database; Ctrl+C to stop)"
# 127.0.0.1: reachable from this computer only, not from other devices on your network.
# -e DATABASE_URL (no value) passes it from this script's environment.
exec docker run --rm --name stratos-dashboard-local -p "127.0.0.1:$PORT:8080" -e DATABASE_URL "$IMAGE"
