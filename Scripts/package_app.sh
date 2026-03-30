#!/usr/bin/env bash
set -euo pipefail

CONF=${1:-release}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

APP_NAME=${APP_NAME:-PhotoLM}
BUNDLE_ID=${BUNDLE_ID:-com.photolm.desktop}
MACOS_MIN_VERSION=${MACOS_MIN_VERSION:-14.0}
SIGNING_MODE=${SIGNING_MODE:-adhoc}
APP_IDENTITY=${APP_IDENTITY:-}
ARCHES=${ARCHES:-"arm64 x86_64"}
EMBED_PYTHON_EXECUTABLE=${EMBED_PYTHON_EXECUTABLE:-/usr/local/bin/python3}
FORCE_PYTHON_REBUILD=${FORCE_PYTHON_REBUILD:-0}
EMBEDDED_REQUIREMENTS_FILE=${EMBEDDED_REQUIREMENTS_FILE:-$ROOT/requirements.embedded.txt}

if [[ -f "$ROOT/version.env" ]]; then
  source "$ROOT/version.env"
else
  MARKETING_VERSION=1.0.0
  BUILD_NUMBER=1
fi

ARCH_LIST=( ${ARCHES} )
if [[ ${#ARCH_LIST[@]} -eq 0 ]]; then
  echo "ARCHES is empty" >&2
  exit 1
fi

if [[ ! -x "$EMBED_PYTHON_EXECUTABLE" ]]; then
  echo "EMBED_PYTHON_EXECUTABLE not found: $EMBED_PYTHON_EXECUTABLE" >&2
  exit 1
fi

for script_name in ui_remove.py ui_viewer.py requirements.txt requirements.embedded.txt; do
  if [[ ! -f "$ROOT/$script_name" ]]; then
    echo "Required file is missing: $ROOT/$script_name" >&2
    exit 1
  fi
done

if [[ ! -f "$EMBEDDED_REQUIREMENTS_FILE" ]]; then
  echo "Embedded requirements file is missing: $EMBEDDED_REQUIREMENTS_FILE" >&2
  exit 1
fi

for arch in "${ARCH_LIST[@]}"; do
  swift build -c "$CONF" --arch "$arch"
done

APP="$ROOT/${APP_NAME}.app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources" "$APP/Contents/Frameworks"

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>${APP_NAME}</string>
    <key>CFBundleDisplayName</key><string>${APP_NAME}</string>
    <key>CFBundleIdentifier</key><string>${BUNDLE_ID}</string>
    <key>CFBundleExecutable</key><string>${APP_NAME}</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleShortVersionString</key><string>${MARKETING_VERSION}</string>
    <key>CFBundleVersion</key><string>${BUILD_NUMBER}</string>
    <key>LSMinimumSystemVersion</key><string>${MACOS_MIN_VERSION}</string>
</dict>
</plist>
PLIST

build_product_path() {
  local name="$1"
  local arch="$2"
  case "$arch" in
    arm64|x86_64) echo ".build/${arch}-apple-macosx/$CONF/$name" ;;
    *) echo ".build/$CONF/$name" ;;
  esac
}

install_binary() {
  local name="$1"
  local dest="$2"
  local binaries=()
  for arch in "${ARCH_LIST[@]}"; do
    local src
    src=$(build_product_path "$name" "$arch")
    if [[ ! -f "$src" ]]; then
      echo "Missing build for ${arch}: $src" >&2
      exit 1
    fi
    binaries+=("$src")
  done
  if [[ ${#binaries[@]} -gt 1 ]]; then
    lipo -create "${binaries[@]}" -output "$dest"
  else
    cp "${binaries[0]}" "$dest"
  fi
  chmod +x "$dest"
}

install_binary "$APP_NAME" "$APP/Contents/MacOS/$APP_NAME"

PY_BASE_PREFIX=$("$EMBED_PYTHON_EXECUTABLE" - <<'PY'
import sys
print(sys.base_prefix)
PY
)
PY_VERSION=$("$EMBED_PYTHON_EXECUTABLE" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)
PY_FRAMEWORK_ROOT=$(dirname "$(dirname "$PY_BASE_PREFIX")")
if [[ ! -d "$PY_FRAMEWORK_ROOT" ]]; then
  echo "Python framework root not found: $PY_FRAMEWORK_ROOT" >&2
  exit 1
fi

mkdir -p "$APP/Contents/Resources/PythonScripts"
cp "$ROOT/ui_remove.py" "$APP/Contents/Resources/PythonScripts/ui_remove.py"
cp "$ROOT/ui_viewer.py" "$APP/Contents/Resources/PythonScripts/ui_viewer.py"
cp "$ROOT/requirements.txt" "$APP/Contents/Resources/PythonScripts/requirements.txt"
cp "$EMBEDDED_REQUIREMENTS_FILE" "$APP/Contents/Resources/PythonScripts/requirements.embedded.txt"

rsync -a --delete "$PY_FRAMEWORK_ROOT/" "$APP/Contents/Resources/Python.framework/"

run_for_arch() {
  local arch="$1"
  shift
  local host_arch
  host_arch=$(uname -m)
  if [[ "$arch" == "$host_arch" ]]; then
    "$@"
    return
  fi
  if [[ "$host_arch" == "arm64" && "$arch" == "x86_64" ]]; then
    arch -x86_64 "$@"
    return
  fi
  if [[ "$host_arch" == "x86_64" && "$arch" == "arm64" ]]; then
    arch -arm64 "$@"
    return
  fi
  echo "Unsupported host/target arch combination: host=$host_arch target=$arch" >&2
  exit 1
}

CACHE_ROOT="$ROOT/.build/embedded_python"
mkdir -p "$CACHE_ROOT"

for arch in "${ARCH_LIST[@]}"; do
  SITE_CACHE="$CACHE_ROOT/site-packages/$arch"
  MARKER="$SITE_CACHE/.complete"
  if [[ "$FORCE_PYTHON_REBUILD" == "1" ]]; then
    rm -rf "$SITE_CACHE"
  fi
  if [[ ! -f "$MARKER" ]]; then
    rm -rf "$SITE_CACHE"
    mkdir -p "$SITE_CACHE"
    run_for_arch "$arch" "$EMBED_PYTHON_EXECUTABLE" -m pip install --upgrade pip setuptools wheel
    run_for_arch "$arch" "$EMBED_PYTHON_EXECUTABLE" -m pip install --no-cache-dir --upgrade -r "$EMBEDDED_REQUIREMENTS_FILE" --target "$SITE_CACHE"
    touch "$MARKER"
  fi

  SITE_DEST="$APP/Contents/Resources/PythonSitePackages/$arch"
  mkdir -p "$SITE_DEST"
  rsync -a --delete "$SITE_CACHE/" "$SITE_DEST/"
done

chmod -R u+w "$APP"
xattr -cr "$APP" || true
find "$APP" -name '._*' -delete || true

if [[ "$SIGNING_MODE" == "adhoc" || -z "$APP_IDENTITY" ]]; then
  codesign --force --deep --sign "-" "$APP"
else
  codesign --force --deep --timestamp --options runtime --sign "$APP_IDENTITY" "$APP"
fi

echo "Embedded Python version: $PY_VERSION"
echo "Created $APP"
