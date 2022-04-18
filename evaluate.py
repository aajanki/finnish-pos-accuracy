import logging
import re
import time
import numpy as np
import pandas as pd
import typer
from alignment.sequence import Sequence
from alignment.vocabulary import Vocabulary
from alignment.sequencealigner import SimpleScoring, StrictGlobalSequenceAligner
from datasets import parse_conllu
from nlpmodels import *
from itertools import zip_longest
from pathlib import Path
from typing import Optional


def main(
        models: Optional[str] = typer.Option(None, help='comma-separated list of models to evaluate'),
        testsets: Optional[str] = typer.Option(None, help='comma-separate list of testsets to evaluate'),
):
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    outputdir = Path('results')
    errorcasedir = outputdir / 'errorcases'
    predictionsdir = outputdir / 'predictions'

    selected_models = [
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

    if models:
        models_list = models.split(',')
        selected_models = [m for m in selected_models if m.name in models_list]
        selected_names = set(m.name for m in selected_models)

        if set(models_list) - selected_names:
            raise ValueError(f'Unknown model(s): {", ".join(set(models_list) - selected_names)}')

    selected_testsets = [
        { 'name': 'UD_Finnish_TDT', 'load': load_testset_ud_tdt },
        { 'name': 'ftb1u', 'load': load_testset_ftb1u },
        { 'name': 'ftb2-news', 'load': load_testset_ftb2_news },
        { 'name': 'ftb2-sofie', 'load': load_testset_ftb2_sofie },
        { 'name': 'ftb2-wikipedia', 'load': load_testset_ftb2_wikipedia },
    ]

    if testsets:
        testsets_list = testsets.split(',')
        selected_testsets = [t for t in selected_testsets if t['name'] in testsets_list]
        selected_names = set(t['name'] for t in selected_testsets)

        if set(testsets_list) - selected_names:
            raise ValueError(f'Unknown testset(s): {", ".join(set(testsets_list) - selected_names)}')

    for model in selected_models:
        print(f'Initializing {model.name}')
        model.initialize()

    outputdir.mkdir(exist_ok=True, parents=True)
    errorcasedir.mkdir(exist_ok=True, parents=True)
    predictionsdir.mkdir(exist_ok=True, parents=True)
    for testset in selected_testsets:
        sentences = testset['load']()

        print()
        print(f'Test set {testset["name"]}')
        print(f'Test set size: {count_tokens(sentences)} test tokens '
              f'in {len(sentences)} sentences')

        evaluation_results = []
        for model in selected_models:
            print()
            print(f'Evaluating {model.name} on {testset["name"]}')

            metrics, lemma_errors, pos_errors = \
                evaluate_model(model, sentences, predictionsdir, testset['name'])

            print(f'Lemma WER: {metrics["Lemmatization WER"]:.3f}')
            print(f'UPOS WER: {metrics["UPOS WER"]:.3f}')
            print(f'Duration: {metrics["Duration"]:.1f} s '
                  f'({metrics["Sentences per second"]:.1f} sentences/s)')

            evaluation_results.append(metrics)

            le_filename = errorcasedir / f'lemma_erros_{model.name}_{testset["name"]}.txt'
            pe_filename = errorcasedir / f'pos_erros_{model.name}_{testset["name"]}.txt'
            with open(le_filename, 'w') as lemma_errors_file, \
                 open(pe_filename, 'w') as pos_errors_file:
                write_errors(lemma_errors_file, lemma_errors)
                write_errors(pos_errors_file, pos_errors)

        df = pd.DataFrame(evaluation_results, index=[m.name for m in selected_models])
        df.to_csv(outputdir / f'evaluation_{testset["name"]}.csv')


def evaluate_model(model, sentences, outputdir, testset_name):
    t0 = time.time()

    texts = [s.text() for s in sentences]
    predicted = model.parse(texts)
    assert len(predicted) == len(sentences)

    duration = time.time() - t0
    sentences_per_s = len(sentences)/duration

    (outputdir / model.name).mkdir(exist_ok=True, parents=True)
    prediction_file = outputdir / model.name / f'{testset_name}.conllu'
    with open(prediction_file, 'w') as conllu_file:
        write_results_conllu(conllu_file, predicted)

    lemma_matches = []
    lemma_errors = []
    pos_matches = []
    pos_errors = []
    for sent, pred in zip(sentences, predicted):
        expected_lemmas = sent.lemmas()
        observed_lemmas = pred['lemmas']
        matches = count_sequence_matches(
            normalize_lemmas(expected_lemmas),
            normalize_lemmas(observed_lemmas))
        lemma_matches.append(matches)

        if matches['matches'] != matches['gold_length']:
            lemma_errors.append((sent.text(), observed_lemmas, expected_lemmas))

        expected_pos = sent.pos()
        observed_pos = pred['pos']
        matches = count_sequence_matches(expected_pos, observed_pos)
        pos_matches.append(matches)

        if matches['matches'] != matches['gold_length']:
            pos_errors.append((sent.text(), observed_pos, expected_pos))

    df_lemma = pd.DataFrame(lemma_matches)
    df_pos = pd.DataFrame(pos_matches)

    metrics = {}
    metrics.update(calculate_metrics(df_lemma, 'Lemmatization '))
    metrics.update(calculate_metrics(df_pos, 'UPOS '))
    metrics['Duration'] = duration
    metrics['Sentences per second'] = sentences_per_s
    return (metrics, lemma_errors, pos_errors)


def count_sequence_matches(gold_seq, predicted_seq):
    if len(gold_seq) == len(predicted_seq):
        matches = (np.asarray(gold_seq) == np.asarray(predicted_seq)).sum()
        aligned_length = len(gold_seq)
        substitutions = aligned_length - matches
        deletions = insertions = 0
    else:
        encoded = align_sequences(gold_seq, predicted_seq)
        aligned_length = max(len(encoded.first), len(encoded.second))

        substitutions = 0
        deletions = 0
        insertions = 0
        matches = 0
        for a, b in zip(encoded.first, encoded.second):
            if a == encoded.gap:
                insertions += 1
            elif b == encoded.gap:
                deletions += 1
            elif a != b:
                substitutions += 1
            else:
                matches += 1

        reference_len = substitutions + deletions + matches
        assert reference_len == len(gold_seq)

    return {
        'matches': matches,
        'substitutions': substitutions,
        'insertions': insertions,
        'deletions': deletions,
        'gold_length': len(gold_seq),
        'predicted_length': len(predicted_seq),
        'aligned_length': aligned_length
    }


def calculate_metrics(df, key_prefix):
    if len(df) == 0:
        wer = 0
        f1 = 0
        recall = 0
        precision = 0
        aligned_accuracy = 0
    else:
        x = df.sum(axis=0)
        N = x['substitutions'] + x['deletions'] + x['matches']
        wer = (x['substitutions'] + x['deletions'] + x['insertions'])/N
        recall = x['matches']/x['gold_length']
        if x['predicted_length'] > 0:
            precision = x['matches']/x['predicted_length']
        else:
            precision = np.nan
        if recall + precision > 0:
            f1 = 2*recall*precision/(recall + precision)
        else:
            f1 = np.nan
        aligned_accuracy = x['matches']/x['aligned_length']

    return {
        key_prefix + 'WER': wer,
        key_prefix + 'F1': f1,
        key_prefix + 'precision': precision,
        key_prefix + 'recall': recall,
        key_prefix + 'aligned accuracy': aligned_accuracy
    }


def normalize_lemmas(lemmas):
    return [normalize_lemma(lemma) for lemma in lemmas]


def normalize_lemma(lemma):
    if re.match(r'[-#]+$', lemma):
        return lemma

    lemma = lemma.lower().replace('#', '').replace('-', '')
    return normalize_quotes(lemma)


def remove_compund_word_boundary_markers(word):
    return re.sub(r'(?<=\w)#(?=\w)', '', word)


def normalize_quotes(word):
    if word == '”' or word == '“':
        return '"'
    elif word == '’' or word == '‘':
        return "'"
    else:
        return word


def load_testset_ud_tdt():
    return parse_conllu(open('data/preprocessed/UD_Finnish-TDT/fi_tdt-ud-test.conllu'))


def load_testset_ftb1u():
    return parse_conllu(open('data/preprocessed/ftb1/ftb1u_sample.tsv'))


def load_testset_ftb2_wikipedia():
    return parse_conllu(open('data/preprocessed/ftb2/FinnTreeBank_2/wikipedia-samples_tab.txt'))


def load_testset_ftb2_news():
    return parse_conllu(open('data/preprocessed/ftb2/FinnTreeBank_2/news-samples_tab.txt'))


def load_testset_ftb2_sofie():
    return parse_conllu(open('data/preprocessed/ftb2/FinnTreeBank_2/sofie12_tab.txt'))


def count_tokens(sentences):
    return sum(s.count_tokens() for s in sentences)


def align_sequences(seq_a, seq_b):
    # Must escape '-' because alignment library uses it as a gap
    # marker.
    escaped_seq_a = ['\\-' if x == '-' else x for x in seq_a]
    escaped_seq_b = ['\\-' if x == '-' else x for x in seq_b]

    v = Vocabulary()
    encoded_a = v.encodeSequence(Sequence(escaped_seq_a))
    encoded_b = v.encodeSequence(Sequence(escaped_seq_b))

    scoring = SimpleScoring(matchScore=3, mismatchScore=-1)
    aligner = StrictGlobalSequenceAligner(scoring, gapScore=-2)
    _, encodeds = aligner.align(encoded_a, encoded_b, backtrace=True)
    return encodeds[0]


def write_errors(f, errors):
    for error in errors:
        f.write('\t')
        f.write(error[0])
        f.write('\n')

        f.write('exp\t')
        f.write(' '.join(error[2]))
        f.write('\n')

        f.write('obs\t')
        f.write(' '.join(error[1]))
        f.write('\n')

        f.write('-'*80)
        f.write('\n')


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

        it = zip_longest(observed_words, observed_lemmas, observed_pos, fillvalue='')
        for i, (orth, lemma, pos) in enumerate(it):
            nlemma = remove_compund_word_boundary_markers(lemma)
            if i == 0:
                fake_head = '0'
                fake_rel = 'root'
            else:
                fake_head = '1'
                fake_rel = 'dep'

            f.write(f'{i + 1}\t{orth}\t{nlemma}\t{pos}\t_\t_\t{fake_head}\t{fake_rel}\t_\t_\n')
        f.write('\n')


if __name__ == '__main__':
    typer.run(main)
