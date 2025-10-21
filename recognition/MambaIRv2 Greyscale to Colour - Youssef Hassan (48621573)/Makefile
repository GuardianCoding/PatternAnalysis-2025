# Makefile at project root
# Made with the help of ChatGPT

ENV_NAME := mamba-colour

SHELL := /bin/bash

.PHONY: setup reinstall clean veryclean dataset

setup:
	@echo "[Make] Creating/refreshing conda env '$(ENV_NAME)'..."
	conda env create -f env.yml || conda env update -f env.yml --prune
	@echo "[Make] Installing MambaIR into '$(ENV_NAME)'..."
	conda run -n $(ENV_NAME) bash scripts/install_mambair.sh
	@echo "[Make] Done. Activate with: conda activate $(ENV_NAME)"

# Re-run only the MambaIR installer (quick)
reinstall:
	conda run -n $(ENV_NAME) bash scripts/install_mambair.sh

# Remove only the external source checkout
clean:
	rm -rf external/MambaIR

# Remove env + checkout (careful!)
veryclean: clean
	conda env remove -n $(ENV_NAME) || true

dataset:
	@echo "Downloading Dataset..."
	@chmod +x scripts/download_datasets.sh
	bash scripts/download_datasets.sh
	@echo "Datasets downloaded under ./datasets"