import os
import os.path
import numpy as np
from alignment.sequence import Sequence
from alignment.vocabulary import Vocabulary
from alignment.sequencealigner import SimpleScoring, GlobalSequenceAligner
from models import *


def main():
    outputdir = 'errors'
    sentences = parse_conllu(open('data/UD_Finnish-TDT/fi_tdt-ud-test.conllu'))

    os.makedirs(outputdir, exist_ok=True)

    print(f'Evaluating on {len(sentences)} sentences')

    #evaluated = [UDPipe(), Voikko()]
    evaluated = [Voikko()]
    for model in evaluated:
        print(f'Evaluating {model.name}')

        le_filename = os.path.join(outputdir, f'lemma_erros_{model.name}.txt')
        pe_filename = os.path.join(outputdir, f'pos_erros_{model.name}.txt')
        with open(le_filename, 'w') as lemma_errors_file, \
             open(pe_filename, 'w') as pos_errors_file:
            lemma_accuracy, pos_accuracy, lemma_errors, pos_errors = \
                evaluate_model(model, sentences)

            print(f'Lemma accuracy: {lemma_accuracy}')
            print(f'POS accuracy: {pos_accuracy}')
            print()

            write_errors(lemma_errors_file, lemma_errors)
            write_errors(pos_errors_file, pos_errors)


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
        n = compute_matches(
            normalize_lemmas(observed_lemmas),
            normalize_lemmas(expected_lemmas))
        correct_count = min(n, sentence_len)
        num_correct_lemmas += correct_count

        if sentence_len != correct_count:
            lemma_errors.append((sent['tokens'], observed_lemmas, expected_lemmas))

        expected_pos = sent['pos']
        n = compute_matches(observed_pos, expected_pos)
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


def compute_matches(seq_a, seq_b):
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
