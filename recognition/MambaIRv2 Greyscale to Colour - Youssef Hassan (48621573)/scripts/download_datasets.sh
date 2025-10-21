#!/usr/bin/env bash
set -e
mkdir -p datasets && cd datasets

echo "↓ Downloading DFWB_RGB (HQ)..."
gdown --id 1pmyHyyKbgz6J_v5cSVE7R5oS7nhaC6YE -O DFWB_RGB_HQ.zip
unzip -q DFWB_RGB_HQ.zip && rm DFWB_RGB_HQ.zip

echo "↓ Downloading ColorDN test sets..."
gdown --id 1U9sahbwc0_So3kHmyvXocslTT-wG0gyB -O ColorDN.zip
unzip -q ColorDN.zip && rm ColorDN.zip

echo "↓ (optional) Downloading DF2K HR..."
gdown --id 1VGcJpH5H07Fi3jVdEx1A09G7W2Aj5yTb -O DF2K_HR.zip || true
unzip -q DF2K_HR.zip 2>/dev/null && rm -f DF2K_HR.zip

echo "Done!"