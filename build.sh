#!/bin/bash

# Install Git LFS
apt-get update && apt-get install -y git-lfs

# Pull the large files
git lfs pull

# Install Python dependencies
pip install -r requirements.txt