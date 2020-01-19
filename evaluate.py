import logging
import time
import os
import os.path
import numpy as np
import pandas as pd
from alignment.sequence import Sequence
from alignment.vocabulary import Vocabulary
from alignment.sequencealigner import SimpleScoring, StrictGlobalSequenceAligner
from nlpmodels import *


def main():
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

    outputdir = 'results'
    errorcasedir = 'results/errorcases'

    models = [
        UDPipe('fi-tdt'),
        #UDPipe('fi'),
        StanfordNLP(),
        Voikko(),
        TurkuNeuralParser(),
        FinnPos(),
        SpacyFiExperimental(),
    ]

    testsets = [
        { 'name': 'UD_Finnish_TDT', 'load': load_testset_ud_tdt },
        { 'name': 'ftb1u', 'load': load_testset_ftb1u },
        { 'name': 'ftb2-news', 'load': load_testset_ftb2_news },
        { 'name': 'ftb2-sofie', 'load': load_testset_ftb2_sofie },
        { 'name': 'ftb2-wikipedia', 'load': load_testset_ftb2_wikipedia },
    ]

    try:
        os.makedirs(outputdir, exist_ok=True)
        os.makedirs(errorcasedir, exist_ok=True)
        for testset in testsets:
            sentences = testset['load']()

            print(f'Test set {testset["name"]}')
            print(f'Test set size: {count_tokens(sentences)} test tokens '
                  f'in {len(sentences)} sentences')

            evaluation_results = []
            for model in models:
                print()
                print(f'Evaluating {model.name} on {testset["name"]}')

                metrics, lemma_errors, pos_errors = \
                    evaluate_model(model, sentences)

                print(f'Lemma WER: {metrics["Lemmatization WER"]:.3f}')
                print(f'UPOS WER: {metrics["UPOS WER"]:.3f}')
                print(f'Duration: {metrics["Duration"]:.1f} s '
                      f'({metrics["Sentences per second"]:.1f} sentences/s)')

                evaluation_results.append(metrics)

                le_filename = os.path.join(
                    errorcasedir, f'lemma_erros_{model.name}_{testset["name"]}.txt')
                pe_filename = os.path.join(
                    errorcasedir, f'pos_erros_{model.name}_{testset["name"]}.txt')
                with open(le_filename, 'w') as lemma_errors_file, \
                     open(pe_filename, 'w') as pos_errors_file:
                    write_errors(lemma_errors_file, lemma_errors)
                    write_errors(pos_errors_file, pos_errors)

            df = pd.DataFrame(evaluation_results, index=[m.name for m in models])
            df.to_csv(os.path.join(outputdir, f'evaluation_{testset["name"]}.csv'))

    finally:
        for model in models:
            model.terminate()


def evaluate_model(model, sentences):
    t0 = time.time()

    texts = [x['tokens'] for x in sentences]
    predicted = model.parse(texts)
    assert len(predicted) == len(sentences)
    
    lemma_matches = []
    lemma_errors = []
    pos_matches = []
    pos_errors = []
    for sent, pred in zip(sentences, predicted):
        expected_lemmas = sent['lemmas']
        observed_lemmas = pred['lemmas']
        matches = count_sequence_matches(
            normalize_lemmas(expected_lemmas),
            normalize_lemmas(observed_lemmas))
        lemma_matches.append(matches)

        if matches['matches'] != matches['gold_length']:
            lemma_errors.append((sent['tokens'], observed_lemmas, expected_lemmas))

        expected_pos = sent['pos']
        observed_pos = pred['pos']
        matches = count_sequence_matches(expected_pos, observed_pos)
        pos_matches.append(matches)

        if matches['matches'] != matches['gold_length']:
            pos_errors.append((sent['tokens'], observed_pos, expected_pos))

    duration = time.time() - t0
    sentences_per_s = len(sentences)/duration

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
        precision = x['matches']/x['predicted_length']
        f1 = 2*recall*precision/(recall + precision)
        aligned_accuracy = x['matches']/x['aligned_length']

    return {
        key_prefix + 'WER': wer,
        key_prefix + 'F1': f1,
        key_prefix + 'precision': precision,
        key_prefix + 'recall': recall,
        key_prefix + 'aligned accuracy': aligned_accuracy
    }


def normalize_lemmas(lemmas):
    norm = (w.lower().replace('#', '') for w in lemmas)
    return [normalize_quotes(w) for w in norm]


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


def parse_conllu(f):
    sentences = []
    tokens = []
    lemmas = []
    pos = []
    for linenum, line in enumerate(f.readlines()):
        line = line.strip()
        if line.startswith('#') or line == '':
            continue

        fields = line.split('\t')
        if len(fields) != 10:
            logging.warning(f'Ignoring invalid line {linenum} with {len(fields)} fields')
            continue

        sid = fields[0]

        if sid == '1':
            # sentence boundary
            if tokens:
                sentences.append({'tokens': tokens, 'lemmas': lemmas, 'pos': pos})

            tokens = []
            lemmas = []
            pos = []

        token = fields[1].replace(' ', '')
        lemma = fields[2].replace(' ', '')

        tokens.append(token)
        lemmas.append(lemma)
        pos.append(fields[3])

    if tokens:
        sentences.append({'tokens': tokens, 'lemmas': lemmas, 'pos': pos})

    return sentences


def count_tokens(sentences):
    return np.sum([len(x['tokens']) for x in sentences])


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
        f.write(' '.join(error[0]))
        f.write('\n')

        f.write('exp\t')
        f.write(' '.join(error[2]))
        f.write('\n')

        f.write('obs\t')
        f.write(' '.join(error[1]))
        f.write('\n')

        f.write('-'*80)
        f.write('\n')


if __name__ == '__main__':
    main()
