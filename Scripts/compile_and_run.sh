#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
APP_NAME=${APP_NAME:-PhotoLM}

pkill -f "${APP_NAME}.app/Contents/MacOS/${APP_NAME}" 2>/dev/null || true
SIGNING_MODE=adhoc "$ROOT/Scripts/package_app.sh" release
open "$ROOT/${APP_NAME}.app"
