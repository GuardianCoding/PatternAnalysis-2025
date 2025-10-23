#!/usr/bin/env bash
# ==============================================================
# install_mambair.sh
# Clone/update MambaIR and make it importable via .pth (no build).
# Downloads latest release assets (model checkpoints).
# Handles detached HEAD by checking out the remote default branch.
# ==============================================================
# Made with the help of ChatGPT5

set -euo pipefail

TARGET_DIR="external/MambaIR"
REPO_URL="https://github.com/csguoh/MambaIR"
CHECKPOINTS_DIR="checkpoints"

echo "[Setup] Preparing MambaIR from ${REPO_URL}"

# Ensure parent directory exists
mkdir -p "$(dirname "$TARGET_DIR")"

# Clone if missing
if [ ! -d "$TARGET_DIR/.git" ]; then
  echo "[Clone] Cloning fresh copy..."
  git clone --quiet "$REPO_URL" "$TARGET_DIR"
fi

# Always fetch latest and prune
git -C "$TARGET_DIR" fetch --all --tags --prune --quiet

# Resolve remote default branch (origin/HEAD); fallback to main/master
DEFAULT_BRANCH="$(git -C "$TARGET_DIR" symbolic-ref --quiet --short refs/remotes/origin/HEAD 2>/dev/null | sed 's#^origin/##' || true)"
if [ -z "${DEFAULT_BRANCH:-}" ]; then
  for b in main master; do
    if git -C "$TARGET_DIR" ls-remote --heads origin "$b" | grep -q "$b"; then
      DEFAULT_BRANCH="$b"
      break
    fi
  done
fi
if [ -z "${DEFAULT_BRANCH:-}" ]; then
  echo "[Error] Could not determine default branch for origin."
  exit 1
fi

# Check out a local branch that tracks origin/<default>, avoiding detached HEAD
git -C "$TARGET_DIR" checkout -B "$DEFAULT_BRANCH" "origin/$DEFAULT_BRANCH" --quiet

# Fast-forward update to latest commit on that branch
git -C "$TARGET_DIR" pull --ff-only --quiet

LATEST_COMMIT="$(git -C "$TARGET_DIR" rev-parse HEAD)"
echo "[Info] Using commit: $LATEST_COMMIT on branch: $DEFAULT_BRANCH"

# Download latest release assets (model checkpoints)
echo "[Release] Fetching latest release information..."
mkdir -p "$CHECKPOINTS_DIR"

# Get latest release info using GitHub API (no auth required for public repos)
RELEASE_INFO=$(curl -sL "https://api.github.com/repos/csguoh/MambaIR/releases/latest" || echo "")

if [ -n "$RELEASE_INFO" ] && echo "$RELEASE_INFO" | grep -q '"tag_name"'; then
  RELEASE_TAG=$(echo "$RELEASE_INFO" | grep '"tag_name"' | sed -E 's/.*"tag_name": "([^"]+)".*/\1/')
  echo "[Release] Found latest release: $RELEASE_TAG"
  
  # Extract download URLs for assets
  ASSET_URLS=$(echo "$RELEASE_INFO" | grep '"browser_download_url"' | sed -E 's/.*"browser_download_url": "([^"]+)".*/\1/')
  
  if [ -n "$ASSET_URLS" ]; then
    echo "[Download] Downloading model checkpoints to ${CHECKPOINTS_DIR}/"
    
    # Download each asset
    while IFS= read -r url; do
      if [ -n "$url" ]; then
        filename=$(basename "$url")
        output_path="${CHECKPOINTS_DIR}/${filename}"
        
        # Skip if already downloaded
        if [ -f "$output_path" ]; then
          echo "[Skip] ${filename} already exists"
        else
          echo "[Downloading] ${filename}..."
          curl -L --progress-bar -o "$output_path" "$url"
          echo "[Downloaded] ${filename}"
        fi
      fi
    done <<< "$ASSET_URLS"
    
    echo "[Release] All checkpoints downloaded to ${CHECKPOINTS_DIR}/"
  else
    echo "[Warning] No assets found in latest release"
  fi
else
  echo "[Warning] Could not fetch release info. Skipping checkpoint download."
  echo "          You may need to download checkpoints manually from:"
  echo "          ${REPO_URL}/releases/latest"
fi

# Patch missing VERSION file (harmless for .pth mode)
if [ ! -f "$TARGET_DIR/VERSION" ]; then
  echo "0.0.0" > "$TARGET_DIR/VERSION"
  echo "[Patch] Created VERSION file"
fi

# Locate site-packages in the *current* Python env (this script should be run via `conda run -n ...`)
SITE_PACKAGES=$(python - <<'PY'
import site, os
cands = []
try:
    cands += site.getsitepackages()
except Exception:
    pass
cands.append(site.getusersitepackages())
for p in cands:
    if os.path.isdir(p):
        print(p)
        break
PY
)

if [ -z "${SITE_PACKAGES:-}" ]; then
  echo "[Error] Could not locate site-packages. Are you running inside the target conda env?"
  exit 1
fi

pip uninstall -y basicsr || true

# Write a .pth file so Python adds the repo root to sys.path
PTH_FILE="${SITE_PACKAGES}/mambair_local.pth"
# Use absolute path
REPO_ABS_PATH="$(cd "$TARGET_DIR" && pwd)"
echo "$REPO_ABS_PATH" > "${PTH_FILE}"
echo "[Link] Wrote ${PTH_FILE} -> ${REPO_ABS_PATH}"

# Optional: disable optional CUDA extensions var (not used in .pth mode)
export BASICSR_EXT=False

echo "[Done] MambaIR is importable via .pth."
echo "       Checkpoints available in: ${CHECKPOINTS_DIR}/"
echo "       Test: conda run -n mamba-colour python -c 'import sys;import mambair,os;print(\"ok\", os.path.exists(\"${REPO_ABS_PATH}/VERSION\"))'"