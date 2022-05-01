import logging
import re
import tempfile
import time
import conll18_ud_eval
import pandas as pd
import typer
from datasets import Dataset
from nlpmodels import *
from itertools import zip_longest
from pathlib import Path
from typing import Optional

all_models = [
    UDPipe('fi-tdt'),
    Voikko(),
    TurkuNeuralParser(),
    FinnPos(),
    SpacyFiExperimental(),
    Stanza(),
    Trankit('base'),
    Trankit('large'),
    Simplemma(),
    UralicNLP(),
]
all_testsets = [
    Dataset('UD_Finnish_TDT', Path('data/preprocessed/UD_Finnish-TDT/fi_tdt-ud-test.conllu')),
    Dataset('ftb1u', Path('data/preprocessed/ftb1/ftb1u_sample.tsv')),
    Dataset('ftb2-news', Path('data/preprocessed/ftb2/FinnTreeBank_2/news-samples_tab.txt')),
    Dataset('ftb2-sofie', Path('data/preprocessed/ftb2/FinnTreeBank_2/sofie12_tab.txt')),
    Dataset('ftb2-wikipedia', Path('data/preprocessed/ftb2/FinnTreeBank_2/wikipedia-samples_tab.txt')),
]


def main(
        models: Optional[str] = typer.Option(None, help='comma-separated list of models to evaluate'),
        testsets: Optional[str] = typer.Option(None, help='comma-separate list of testsets to evaluate'),
):
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    outputdir = Path('results')
    predictionsdir = outputdir / 'predictions'

    if models:
        models_list = models.split(',')
        selected_models = [m for m in all_models if m.name in models_list]
        selected_names = set(m.name for m in selected_models)

        if set(models_list) - selected_names:
            raise ValueError(f'Unknown model(s): {", ".join(set(models_list) - selected_names)}')
    else:
        selected_models = list(all_models)

    if testsets:
        testsets_list = testsets.split(',')
        selected_testsets = [t for t in all_testsets if t.name in testsets_list]
        selected_names = set(t.name for t in selected_testsets)

        if set(testsets_list) - selected_names:
            raise ValueError(f'Unknown testset(s): {", ".join(set(testsets_list) - selected_names)}')
    else:
        selected_testsets = list(all_testsets)

    for model in selected_models:
        print(f'Initializing {model.name}')
        model.initialize()

    outputdir.mkdir(exist_ok=True, parents=True)
    predictionsdir.mkdir(exist_ok=True, parents=True)
    for testset in selected_testsets:
        sentences = testset.sentences

        print()
        print(f'Test set {testset.name}')
        print(f'Test set size: {count_tokens(sentences)} test tokens '
              f'in {len(sentences)} sentences')

        evaluation_results = []
        for model in selected_models:
            print()
            print(f'Evaluating {model.name} on {testset.name}')

            metrics = evaluate_model(model, testset, predictionsdir)

            print(f'Lemma F1: {metrics["Lemmatization F1"]:.3f}')
            print(f'UPOS F1: {metrics["UPOS F1"]:.3f}')
            print(f'Duration: {metrics["Duration"]:.1f} s '
                  f'({metrics["Sentences per second"]:.1f} sentences/s)')

            evaluation_results.append(metrics)

        df = pd.DataFrame(evaluation_results, index=[m.name for m in selected_models])
        df.index.name = 'Model'
        df.to_csv(outputdir / f'evaluation_{testset.name}.csv')


def evaluate_model(model, testset, outputdir):
    t0 = time.time()

    texts = [s.text() for s in testset.sentences]
    predicted = model.parse(texts)

    duration = time.time() - t0
    sentences_per_s = len(testset.sentences)/duration

    model_output_dir = outputdir / model.name
    model_output_dir.mkdir(exist_ok=True, parents=True)
    prediction_file = model_output_dir / f'{testset.name}.conllu'
    with open(prediction_file, 'w') as conllu_file:
        write_results_conllu(conllu_file, predicted)

    gold_ud = conll18_ud_eval.load_conllu_file(testset.datapath)
    if model.tokenizer_is_destructive:
        # Fix the surface forms so that the CoNLL evaluation can match them
        # with the original text. Write the result to a temporary file and
        # read it back into conll18 data structures.
        updated_predicted = []

        # TODO: Support for arbitrary sentence splits
        assert len(predicted) == len(testset.sentences)
        for system, gold in zip(predicted, testset.sentences):
            system_fixed = dict(system)
            system_fixed['texts'] = model.fix_surface_forms(system['texts'], gold)
            updated_predicted.append(system_fixed)
        with tempfile.NamedTemporaryFile('w+') as conllu_file:
            write_results_conllu(conllu_file, updated_predicted)
            conllu_file.seek(0)
            system_ud = conll18_ud_eval.load_conllu_file(conllu_file.name)
    else:
        system_ud = conll18_ud_eval.load_conllu_file(prediction_file)

    evaluation = conll18_ud_eval.evaluate(gold_ud, system_ud)

    metrics = {}
    metrics.update(ud_evaluation_to_metrics(evaluation['Lemmas'], 'Lemmatization '))
    metrics.update(ud_evaluation_to_metrics(evaluation['UPOS'], 'UPOS '))
    metrics['Duration'] = duration
    metrics['Sentences per second'] = sentences_per_s

    return metrics


def ud_evaluation_to_metrics(evaluation, key_prefix):
    return {
        key_prefix + 'F1': evaluation.f1,
        key_prefix + 'precision': evaluation.precision,
        key_prefix + 'recall': evaluation.recall,
        key_prefix + 'aligned accuracy': evaluation.aligned_accuracy,
    }


def remove_compund_word_boundary_markers(word):
    return re.sub(r'(?<=\w)#(?=\w)', '', word)


def count_tokens(sentences):
    return sum(s.count_tokens() for s in sentences)


def write_results_conllu(f, predicted):
    """Write the predicted lemmas and POS tags to a CoNLL-U file.

    The dependency relations are obviously bogus. There must be some
    dependencies because the evaluation script expects them, but because we
    are not evaluating the dependency scores, it doesn't matter what
    dependencies we have here.
    """
    for pred in predicted:
        observed_words = pred['texts']
        observed_lemmas = pred['lemmas']
        observed_pos = pred['pos']
        if 'id' in pred:
            ids = pred['id']
        else:
            ids = [str(i) for i in range(1, len(observed_words) + 1)]

        it = zip_longest(ids, observed_words, observed_lemmas, observed_pos, fillvalue='')
        for (i, orth, lemma, pos) in it:
            nlemma = remove_compund_word_boundary_markers(lemma)
            if i == '1':
                fake_head = '0'
                fake_rel = 'root'
            else:
                fake_head = '1'
                fake_rel = 'dep'

            f.write(f'{i}\t{orth}\t{nlemma}\t{pos}\t_\t_\t{fake_head}\t{fake_rel}\t_\t_\n')
        f.write('\n')


if __name__ == '__main__':
    typer.run(main)
