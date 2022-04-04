"""Cleanup input conllu files.

This script normalizes source POS tags to UPOS, removes multi-words tokens,
imputes missing dependencies, handles lines with too few or too many columns
and fixes various other issues (the FTB datasets are quite broken).
"""


import logging
import os
import os.path


def main():
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

    inputdir = 'data/test'
    destdir = 'data/preprocessed'

    logging.info('Preprocessing FTB1')
    preprocess_ftb1u('ftb1/ftb1u_sample.tsv', inputdir, destdir)

    logging.info('Preprocessing FTB2 Wikipedia')
    preprocess_ftb2('ftb2/FinnTreeBank_2/wikipedia-samples_tab.txt', inputdir, destdir)

    logging.info('Preprocessing FTB2 news')
    preprocess_ftb2('ftb2/FinnTreeBank_2/news-samples_tab.txt', inputdir, destdir)

    logging.info('Preprocessing FTB2 Sofie')
    preprocess_ftb2('ftb2/FinnTreeBank_2/sofie12_tab.txt', inputdir, destdir)

    logging.info('Preprocessing UD-TDT')
    preprocess_ud_tdt('UD_Finnish-TDT/fi_tdt-ud-test.conllu', inputdir, destdir)


def preprocess_ftb1u(filename, inputdir, destdir):
    tag_map = {'CONJ': 'CCONJ'}
    preprocess(filename, inputdir, destdir, tag_map, aux_from_deprel=True)


def preprocess_ftb2(filename, inputdir, destdir):
    tag_map = {
        'A': 'ADJ',
        'Abbr': 'X',
        'Adp': 'ADP',
        'Adv': 'ADV',
        'CC': 'CCONJ',
        'CS': 'SCONJ',
        'INTERJ': 'INTJ',
        'N': 'NOUN',
        'Num': 'NUM',
        'POST': 'X',
        'Pron': 'PRON',
        'Pun': 'PUNCT',
        'V': 'VERB'
    }
    preprocess(filename, inputdir, destdir, tag_map, aux_from_deprel=True)


def preprocess_ud_tdt(filename, inputdir, destdir):
    preprocess(filename, inputdir, destdir, {}, aux_from_deprel=False)


def preprocess(filename, inputdir, destdir, tag_map, aux_from_deprel):
    in_path = os.path.join(inputdir, filename)
    out_path = os.path.join(destdir, filename)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    num_sentences = 0
    num_tokens = 0
    num_too_few_columns = 0
    num_too_many_columns = 0
    num_multiword_tokens = 0
    num_non_verb_aux = 0
    num_missing_head = 0
    num_non_continuous_index = 0

    with open(in_path, 'r', encoding='utf-8') as inf, \
         open(out_path, 'w', encoding='utf-8') as outf:

        for sentence in split_into_sentences(inf):
            num_sentences += 1

            prev_index = 0
            for token_line in sentence:
                num_tokens += 1
                if token_line == '' or token_line.startswith('#'):
                    outf.write(token_line)
                    outf.write('\n')
                    continue

                cols = token_line.strip('\n').split('\t')
                if len(cols) < 10:
                    num_too_few_columns += 1
                    # This happens a few times in FTB1/2. Most of these should
                    # be commas. We'll set all of them to commas with heads
                    # pointing to previous tokens.
                    #
                    # In few cases, the token should be something else, but
                    # we set them to commas anyway.
                    cols = [
                        cols[0], ',', ',', 'PUNCT', 'punct', '_',
                        str(int(cols[0]) - 1), 'punct', '_', '_'
                    ]
                if len(cols) > 10:
                    num_too_many_columns += 1
                    cols = cols[:10]

                # Skip multiword tokens and empty nodes
                if '-' in cols[0] or '.' in cols[0]:
                    num_multiword_tokens += 1
                    continue

                if int(cols[0]) != prev_index + 1:
                    num_non_continuous_index += 1
                    cols[0] = str(prev_index + 1)

                cols[3] = normalize_pos(cols[3], tag_map)

                if aux_from_deprel and cols[7] in ['aux', 'aux:pass', 'cop']:
                    if cols[3] in ['VERB', 'AUX']:
                        cols[3] = 'AUX'
                    else:
                        num_non_verb_aux += 1

                if cols[6] == '_':
                    # FTB2 doesn't set heads for punctuations. The evaluation
                    # script needs these, so we set head to the previous token.
                    # This is probably wrong, but since we're not using them,
                    # that's OK.
                    assert cols[3] == 'PUNCT'
                    cols[6] = '2' if cols[0] == '1' else str(int(cols[0]) - 1)
                    cols[7] = 'punct'
                    num_missing_head += 1

                prev_index = int(cols[0])

                outf.write('\t'.join(cols))
                outf.write('\n')
            outf.write('\n')

    logging.info(f'Processed {num_sentences} sentences with {num_tokens} tokens')
    if num_too_few_columns > 0:
        logging.warning(f'Replaced {num_too_few_columns} tokens with too few columns with commas.')
    if num_too_many_columns > 0:
        logging.warning(f'Encountered {num_too_many_columns} tokens with too many columns. Extra columns were ignored')
    if num_non_verb_aux > 0:
        logging.warning(f'Leaving {num_non_verb_aux} non-verb AUX tags unchanged')
    if num_multiword_tokens > 0:
        logging.info(f'Skipped {num_multiword_tokens} multiword tokens')
    if num_missing_head > 0:
        logging.warning(f'Fixed {num_missing_head} missing heads')
    if num_non_continuous_index > 0:
        logging.warning(f'Detected {num_non_continuous_index} tokens with non-continuous indices')


def split_into_sentences(lines):
    index = None
    sentence = []
    for line in lines:
        line = line.strip('\n')
        if line and not line.startswith('#'):
            index, _ = line.split('\t', 1)

        if index == '1':
            if sentence:
                yield sentence
            sentence = []

        if line and not line.startswith('#'):
            sentence.append(line)

    if sentence:
        yield sentence


def normalize_pos(pos, tag_map):
    if '|' in pos:
        # FTB2 has some erroneous tags like "N|Sg|Ine"
        pos = pos.split('|', 1)[0]

    return tag_map.get(pos, pos)


if __name__ == '__main__':
    main()
