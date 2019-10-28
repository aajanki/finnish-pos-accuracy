# Evaluating Finnish POS taggers and lemmatizers

## Setup

Install libvoikko with Finnish morphology data files.

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

./download_data.sh
./download_models.sh
```

## Run

```
python evaluate.py
```
