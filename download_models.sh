#!/bin/sh

set -eu

python download_models.py

./install_finnpos.sh
