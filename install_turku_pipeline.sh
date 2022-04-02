#!/bin/sh

set -eu

echo "Downloading Turku neural pipeline and the Finnish model"

git clone https://github.com/TurkuNLP/Turku-neural-parser-pipeline.git data/Turku-neural-parser-pipeline
cd data/Turku-neural-parser-pipeline
git checkout 8c9425d

python3 fetch_models.py fi_tdt_dia
