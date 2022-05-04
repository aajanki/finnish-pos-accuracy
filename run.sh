#!/bin/sh

set -eu

export PATH=$(pwd)/models/cg3/src:$PATH

# Clean up the test data and save it to data/preprocessed
python preprocess_data.py

# Predict lemmas and POS tags using all models.
# Writes results under results/predictions/*/
python predict.py

# Evaluate by comparing the predictions with the gold standard data.
# Writes results to results/evaluation.csv
python evaluate.py

# Plot the evaluations.
# Saves the plots under results/images/
python plot_results.py

# Save lemma error is results/errors/
python print_errors.py
