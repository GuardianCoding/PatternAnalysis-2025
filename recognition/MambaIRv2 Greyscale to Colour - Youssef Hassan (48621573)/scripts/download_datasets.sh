#!/usr/bin/env bash
set -e
mkdir -p datasets && cd datasets

echo "↓ Downloading DFWB_RGB (HQ)..."
gdown 1jPgG_URDQZ4kyXaMMXJ8AZ8jEErCdKuM -O DFWB_RGB_HQ.zip
unzip -q DFWB_RGB_HQ.zip && rm DFWB_RGB_HQ.zip

echo "↓ Downloading ColorDN test sets..."
gdown 1baLpOjNlTCNbREUDAZf9Lso6YCeUOQER -O ColorDN.zip
unzip -q ColorDN.zip && rm ColorDN.zip

echo "Done!"