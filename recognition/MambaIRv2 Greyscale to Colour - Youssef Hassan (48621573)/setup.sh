#!/usr/bin/env bash
# setup.sh at project root
set -euo pipefail

ENV_NAME="mamba-colour"

# Ensure 'conda' is available in this shell
if ! command -v conda >/dev/null 2>&1; then
  echo "[Error] 'conda' not found in PATH. Open a conda-enabled shell first."
  echo "        (e.g., 'conda init' then start a new shell)"
  exit 1
fi

echo "[Setup] Creating or updating conda env '${ENV_NAME}' from env.yml..."
# Create if missing, otherwise update
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda env update -f env.yml --prune
else
  conda env create -f env.yml
fi

echo "[Setup] Installing MambaIR into '${ENV_NAME}'..."
conda run -n "${ENV_NAME}" bash scripts/install_mambair.sh

echo "[Setup] All done."
echo "        To use the environment now:  conda activate ${ENV_NAME}