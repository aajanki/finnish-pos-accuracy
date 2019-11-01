import logging
import time
import os
import os.path
import numpy as np
import pandas as pd
from alignment.sequence import Sequence
from alignment.vocabulary import Vocabulary
from alignment.sequencealigner import SimpleScoring, GlobalSequenceAligner
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
        TurkuNeuralParser()
    ]

    testsets = [
        { 'name': 'UD_Finnish_TDT', 'load': load_testset_ud_tdt },
        { 'name': 'ftb1u', 'load': load_testset_ftb1u },
    ]

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

            t0 = time.time()
            lemma_accuracy, pos_accuracy, lemma_errors, pos_errors = \
                evaluate_model(model, sentences)
            duration = time.time() - t0
            sentences_per_s = len(sentences)/duration

            evaluation_results.append({
                'lemma accuracy': lemma_accuracy,
                'pos accuracy': pos_accuracy,
                'duration': duration,
                'sentences_per_second': sentences_per_s
            })

            print(f'Lemma accuracy: {lemma_accuracy:.3f}')
            print(f'POS accuracy: {pos_accuracy:.3f}')
            print(f'Duration: {duration:.1f} s '
                  f'({sentences_per_s:.1f} sentences/s)')

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

    for model in models:
        model.terminate()


def evaluate_model(model, sentences):
    total_tokens = 0
    num_correct_lemmas = 0
    num_correct_pos = 0
    lemma_errors = []
    pos_errors = []
    for sent in sentences:
        sentence_len = len(sent['tokens'])
        total_tokens += sentence_len

        observed_lemmas, observed_pos = model.parse(sent['tokens'])

        expected_lemmas = sent['lemmas']
        n = count_matches(
            normalize_lemmas(observed_lemmas),
            normalize_lemmas(expected_lemmas))
        correct_count = min(n, sentence_len)
        num_correct_lemmas += correct_count

        if sentence_len != correct_count:
            lemma_errors.append((sent['tokens'], observed_lemmas, expected_lemmas))

        expected_pos = sent['pos']
        n = count_matches(observed_pos, expected_pos)
        correct_count = min(n, sentence_len)
        num_correct_pos += correct_count

        if sentence_len != correct_count:
            pos_errors.append((sent['tokens'], observed_pos, expected_pos))

    if total_tokens <= 0:
        return (0.0, 0.0, lemma_errors, pos_errors)
    else:
        return (num_correct_lemmas/total_tokens,
                num_correct_pos/total_tokens,
                lemma_errors,
                pos_errors)


def normalize_lemmas(lemmas):
    return [w.lower().replace('#', '') for w in lemmas]


def load_testset_ud_tdt():
    return parse_conllu(open('data/test/UD_Finnish-TDT/fi_tdt-ud-test.conllu'))


def load_testset_ftb1u():
    sentences = parse_conllu(open('data/test/ftb1/ftb1u.tsv'))
    return [conj_to_cconj(x) for x in sentences]


def conj_to_cconj(sentence):
    updated_pos = ['CCONJ' if x == 'CONJ' else x for x in sentence['pos']]
    res = sentence.copy()
    res['pos']= updated_pos
    return res


def parse_conllu(f):
    sentences = []
    tokens = []
    lemmas = []
    pos = []
    for line in f.readlines():
        line = line.strip()
        if line.startswith('#') or line == '':
            continue

        fields = line.split('\t')
        assert len(fields) == 10

        sid = fields[0]

        if sid == '1':
            # sentence boundary
            if tokens:
                sentences.append({'tokens': tokens, 'lemmas': lemmas, 'pos': pos})

            tokens = []
            lemmas = []
            pos = []

        tokens.append(fields[1])
        lemmas.append(fields[2])
        pos.append(fields[3])

    if tokens:
        sentences.append({'tokens': tokens, 'lemmas': lemmas, 'pos': pos})

    return sentences


def count_tokens(sentences):
    return np.sum([len(x['tokens']) for x in sentences])


def count_matches(seq_a, seq_b):
    if len(seq_a) == len(seq_b):
        return (np.asarray(seq_a) == np.asarray(seq_b)).sum()
    else:
        v = Vocabulary()
        encoded_a = v.encodeSequence(Sequence(seq_a))
        encoded_b = v.encodeSequence(Sequence(seq_b))

        scoring = SimpleScoring(matchScore=2, mismatchScore=-2)
        aligner = GlobalSequenceAligner(scoring, gapScore=-1)
        _, encodeds = aligner.align(encoded_a, encoded_b, backtrace=True)

        return (encodeds[0].first == encodeds[0].second).sum()


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
