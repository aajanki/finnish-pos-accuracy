#!/bin/sh

set -eu

echo "Downloading the Finnish Turku pipeline model"
cd models/Turku-neural-parser-pipeline
python3 fetch_models.py --to .. fi_tdt_dia
