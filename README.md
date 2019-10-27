# Evaluating Finnish POS taggers and lemmatizers

## Setup

Install libvoikko with Finnish morphology data files.

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

./load_data.sh

python download_models.py
```

## Run

```
python evaluate.py
```
