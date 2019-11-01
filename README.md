# Evaluating Finnish POS taggers and lemmatizers

## Setup

Install dependencies:
* Docker
* libvoikko with Finnish morphology data files

Create a Python virtual environment and download test data and models by running the following commands:
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

./download_data.sh
./download_models.sh
```

## Run

The Turku neural parser pipeline is run as a docker image. If your
Docker installation requires root priviledges (default on many Linux
distros) define the DOCKER_NEEDS_SUDO environment variable and enter
the sudo password when the evaluation script asks it for running
docker commands.

```
export DOCKER_NEEDS_SUDO=1
python evaluate.py
```

The results will be saved in results/evaluation.csv and POS and lemma
errors made by each model will be saved in results/errorcases.
