#!/bin/sh

set -eu

python download_models.py

./install_finnpos.sh
./install-turku-pipeline.sh

(cd models/cg3/ && ./cmake.sh && make -j3)
