import json
import logging
import re
import time
import typer
from datasets import all_testsets
from nlpmodels import all_models
from itertools import zip_longest
from pathlib import Path
from typing import Optional


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

    predictionsdir.mkdir(exist_ok=True, parents=True)
    for testset in selected_testsets:
        sentences = testset.sentences

        print()
        print(f'Test set {testset.name}')
        print(f'Test set size: {testset.count_tokens()} test tokens '
              f'in {len(sentences)} sentences')

        for model in selected_models:
            print()
            print(f'Predicting lemma and POS: {model.name} on {testset.name}')

            duration = predict_lemma_and_pos(model, testset, predictionsdir)

            print(f'Took {duration:.2f} seconds')


def predict_lemma_and_pos(model, testset, outputdir):
    texts = [s.text() for s in testset.sentences]

    t0 = time.time()
    predicted = model.parse(texts)
    duration = time.time() - t0

    if model.tokenizer_is_destructive:
        # If model's tokenizer doesn't preserve the surface forms exactly,
        # fix them to match the original text as required by the CoNLL
        # evaluation script.
        updated_predicted = []

        # TODO: Support for arbitrary sentence splits
        assert len(predicted) == len(testset.sentences)
        for system, gold in zip(predicted, testset.sentences):
            system_fixed = model.fix_surface_forms(system, gold)
            updated_predicted.append(system_fixed)

        predicted = updated_predicted

    model_output_dir = outputdir / model.name
    model_output_dir.mkdir(exist_ok=True, parents=True)
    prediction_file = model_output_dir / f'{testset.name}.conllu'
    with open(prediction_file, 'w') as conllu_file:
        write_results_conllu(conllu_file, predicted)

    metadata_file = model_output_dir / f'{testset.name}.json'
    with open(metadata_file, 'w') as metadata_file:
        data = {
            'num_sentences': len(testset.sentences),
            'num_tokens': testset.count_tokens(),
            'duration': duration,
        }
        json.dump(data, metadata_file, ensure_ascii=False, indent=2)

    return duration


def remove_compund_word_boundary_markers(word):
    return re.sub(r'(?<=[-–\w])#(?=\w)', '', word)


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
            if '-' in i:
                fake_head = '_'
                fake_rel = '_'
            elif i == '1':
                fake_head = '0'
                fake_rel = 'root'
            else:
                fake_head = '1'
                fake_rel = 'dep'

            f.write(f'{i}\t{orth}\t{nlemma}\t{pos}\t_\t_\t{fake_head}\t{fake_rel}\t_\t_\n')
        f.write('\n')


if __name__ == '__main__':
    typer.run(main)
