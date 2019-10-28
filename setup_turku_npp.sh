#/bin/sh

# Download Turku neural parser pipeline codes, dependencies and the model
#
# The executables will be saved to data/Turku-neural-parser-pipeline

git clone -- https://github.com/TurkuNLP/Turku-neural-parser-pipeline.git data/Turku-neural-parser-pipeline
(cd data/Turku-neural-parser-pipeline; git checkout f926057)
(cd data/Turku-neural-parser-pipeline; git submodule update --init --recursive)

# Update the tensorflow dependency to a version that supports Python 3.7
sed 's/tensorflow==1.12.2/tensorflow==1.13.2/g' data/Turku-neural-parser-pipeline/requirements-cpu.txt > data/Turku-neural-parser-pipeline/requirements-py37.txt
pip install -r data/Turku-neural-parser-pipeline/requirements-py37.txt

python3 data/Turku-neural-parser-pipeline/fetch_models.py fi_tdt
mkdir -p data/Turku-npp-models
mv models_fi_tdt data/Turku-npp-models/fi_tdt
