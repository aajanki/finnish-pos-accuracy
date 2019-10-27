import spacy_udpipe
import numpy as np
from alignment.sequence import Sequence
from alignment.vocabulary import Vocabulary
from alignment.sequencealigner import SimpleScoring, GlobalSequenceAligner


def main():
    nlp = spacy_udpipe.load('fi-tdt')

    sentences = parse_conllu(open('data/UD_Finnish-TDT/fi_tdt-ud-test.conllu'))

    print(f'Evaluating on {len(sentences)} sentences')
    
    num_tokens = 0
    num_correct_lemmas = 0
    num_correct_pos = 0
    for sent in sentences:
        text = ' '.join(sent['tokens'])
        doc = nlp(text)

        sentence_len = len(sent['tokens'])
        num_tokens += sentence_len

        observed_pos = [t.pos_ for t in doc]
        expected_pos = sent['pos']
        n = compute_matches(observed_pos, expected_pos)
        num_correct_pos += min(n, sentence_len)
        
        observed_lemmas = [t.lemma_ for t in doc]
        expected_lemmas = sent['lemmas']
        n = compute_matches(observed_lemmas, expected_lemmas)
        num_correct_lemmas += min(n, sentence_len)

    print(f'POS accuracy: {num_correct_pos/num_tokens}')
    print(f'Lemma accuracy: {num_correct_lemmas/num_tokens}')


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


if __name__ == '__main__':
    main()
