import io
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
        testset_files = list(model_result_dir.glob('*.conllu'))
        for testset_file in testset_files:
            testset_name = testset_file.stem
            metadata_file = model_result_dir / f'{testset_name}.json'

            print()
            print(f'Evaluating {model_name} on {testset_name}')

            metrics = evaluate_model_files(gold_path(testset_name), testset_file, metadata_file)

            print(f'Lemma F1: {metrics["Lemmatization F1"]:.3f}')
            print(f'UPOS F1: {metrics["UPOS F1"]:.3f}')
            print(f'Duration: {metrics["Duration"]:.1f} s '
                  f'({metrics["Tokens per second"]:.1f} tokens/s)')

            metrics['Model'] = model_name
            metrics['Dataset'] = testset_name

            evaluation_results.append(metrics)

        print()
        print(f'Evaluating {model_name} on concatenated datasets')

        data_files = [
            (gold_path(tfile.stem), tfile, model_result_dir / f'{tfile.stem}.json')
            for tfile in testset_files
        ]
        metrics = evaluate_concatenated(data_files)

        print(f'Lemma F1: {metrics["Lemmatization F1"]:.3f}')
        print(f'UPOS F1: {metrics["UPOS F1"]:.3f}')
        print(f'Duration: {metrics["Duration"]:.1f} s '
              f'({metrics["Tokens per second"]:.1f} tokens/s)')

        metrics['Model'] = model_name
        metrics['Dataset'] = 'concatenated'

        evaluation_results.append(metrics)

    df = pd.DataFrame(evaluation_results)
    output_file = outputdir / 'evaluation.csv'
    df.to_csv(output_file, index=False)

    print()
    print('Summary of evaluations on the concatenated datasets:')
    df2 = df[df['Dataset'] == 'concatenated']
    print(df2[['Model', 'UPOS F1', 'Lemmatization F1', 'Tokens per second']]
          .sort_values('Model')
          .to_string(index=False))

    print()
    print('Summary on UD_Finnish_TDT:')
    # For comparing with the results published by Stanza, Trankit and Spacy
    df2 = df[df['Dataset'] == 'UD_Finnish_TDT']
    print(df2[['Model', 'UPOS F1', 'Lemmatization F1', 'Tokens per second']]
          .sort_values('Model')
          .to_string(index=False))

    print()
    print(f'Full results written to {output_file}')


def evaluate_model_files(gold_file, predictions_file, metadata_file):
    with metadata_file.open() as meta:
        metadata = json.load(meta)

    gold_ud = conll18_ud_eval.load_conllu_file(gold_file)
    system_ud = conll18_ud_eval.load_conllu_file(predictions_file)

    return evaluate_model(gold_ud, system_ud, metadata)


def evaluate_concatenated(data_files):
    metadata = {
        'duration': 0,
        'num_sentences': 0,
        'num_tokens': 0,
    }

    gold_data = []
    predictions_data = []
    for gold_file, predictions_file, metadata_file in data_files:
        with metadata_file.open() as f:
            meta = json.load(f)
            metadata['duration'] += meta['duration']
            metadata['num_sentences'] += meta['num_sentences']
            metadata['num_tokens'] += meta['num_tokens']

        with gold_file.open('r', encoding='utf-8') as f:
            data = f.read()
            data = ensure_ends_with_empty_line(data)
            gold_data.append(data)

        with predictions_file.open('r', encoding='utf-8') as f:
            data = f.read()
            data = ensure_ends_with_empty_line(data)
            predictions_data.append(data)

    gold_io = io.StringIO(''.join(gold_data))
    gold_ud = conll18_ud_eval.load_conllu(gold_io)

    predictions_io = io.StringIO(''.join(predictions_data))
    system_ud = conll18_ud_eval.load_conllu(predictions_io)

    return evaluate_model(gold_ud, system_ud, metadata)


def evaluate_model(gold_ud, system_ud, metadata):
    gold_ud = lowercase_lemmas(gold_ud)
    system_ud = lowercase_lemmas(system_ud)
    evaluation = conll18_ud_eval.evaluate(gold_ud, system_ud)

    metrics = {}
    metrics.update(ud_evaluation_to_metrics(evaluation['Lemmas'], 'Lemmatization '))
    metrics.update(ud_evaluation_to_metrics(evaluation['UPOS'], 'UPOS '))
    duration = metadata['duration']
    metrics['Duration'] = duration
    metrics['Sentences per second'] = metadata['num_sentences'] / duration
    metrics['Tokens per second'] = metadata['num_tokens'] / duration

    return metrics


def lowercase_lemmas(ud):
    for w in ud.words:
        w.columns[conll18_ud_eval.LEMMA] = w.columns[conll18_ud_eval.LEMMA].lower()
    return ud


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


def ensure_ends_with_empty_line(text):
    if text.endswith('\n\n'):
        return text
    elif text.endswith('\n'):
        return text + '\n'
    else:
        return text + '\n\n'


if __name__ == '__main__':
    typer.run(main)
