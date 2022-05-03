import json
import logging
import conll18_ud_eval
import numpy as np
import pandas as pd
import typer
from datasets import gold_path
from pathlib import Path


def main():
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    outputdir = Path('results')
    outputdir.mkdir(exist_ok=True, parents=True)
    predictionsdir = outputdir / 'predictions'
    model_result_dirs = predictionsdir.glob('*')

    evaluation_results = []
    for model_result_dir in model_result_dirs:
        model_name = model_result_dir.stem
        testset_files = model_result_dir.glob('*.conllu')
        for testset_file in testset_files:
            testset_name = testset_file.stem
            metadata_file = model_result_dir / f'{testset_name}.json'

            print()
            print(f'Evaluating {model_name} on {testset_name}')

            metrics = evaluate_model(testset_file, gold_path(testset_name), metadata_file)

            print(f'Lemma F1: {metrics["Lemmatization F1"]:.3f}')
            print(f'UPOS F1: {metrics["UPOS F1"]:.3f}')
            print(f'Duration: {metrics["Duration"]:.1f} s '
                  f'({metrics["Sentences per second"]:.1f} sentences/s)')

            metrics['Model'] = model_name
            metrics['Dataset'] = testset_name

            evaluation_results.append(metrics)

    df = pd.DataFrame(evaluation_results)
    df.to_csv(outputdir / f'evaluation.csv', index=False)


def evaluate_model(predictions_file, gold_file, metadata_file):
    with metadata_file.open() as meta:
        metadata = json.load(meta)

    gold_ud = conll18_ud_eval.load_conllu_file(gold_file)
    system_ud = conll18_ud_eval.load_conllu_file(predictions_file)
    evaluation = conll18_ud_eval.evaluate(gold_ud, system_ud)

    metrics = {}
    metrics.update(ud_evaluation_to_metrics(evaluation['Lemmas'], 'Lemmatization '))
    metrics.update(ud_evaluation_to_metrics(evaluation['UPOS'], 'UPOS '))
    duration = metadata['duration']
    metrics['Duration'] = duration
    metrics['Sentences per second'] = metadata['num_sentences'] / duration
    metrics['Tokens per second'] = metadata['num_tokens'] / duration

    return metrics


def ud_evaluation_to_metrics(evaluation, key_prefix):
    f1 = evaluation.f1
    precision = evaluation.precision
    recall = evaluation.recall
    aligned_accuracy = evaluation.aligned_accuracy
    if f1 == 0 and precision == 0 and recall == 0 and aligned_accuracy == 0:
        f1 = precision = recall = aligned_accuracy = np.NaN

    return {
        key_prefix + 'F1': f1,
        key_prefix + 'precision': precision,
        key_prefix + 'recall': recall,
        key_prefix + 'aligned accuracy': aligned_accuracy,
    }


if __name__ == '__main__':
    typer.run(main)
