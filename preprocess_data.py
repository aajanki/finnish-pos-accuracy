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
    preprocess(filename, inputdir, destdir, tag_map)


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
    preprocess(filename, inputdir, destdir, tag_map)


def preprocess_ud_tdt(filename, inputdir, destdir):
    preprocess(filename, inputdir, destdir, {})


def preprocess(filename, inputdir, destdir, tag_map):
    in_path = os.path.join(inputdir, filename)
    out_path = os.path.join(destdir, filename)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    num_sentences = 0
    num_tokens = 0
    num_bad_sentences = 0
    num_bad_tokens = 0
    num_multiword_tokens = 0
    num_empty_tokens = 0

    with open(in_path, 'r', encoding='utf-8') as inf, \
         open(out_path, 'w', encoding='utf-8') as outf:

        for sentence in split_into_sentences(inf):
            num_sentences += 1
            if sentence_has_invalid_tags(sentence):
                num_bad_sentences += 1
                continue

            for token_line in sentence:
                num_tokens += 1
                if token_line == '' or token_line.startswith('#'):
                    outf.write(token_line)
                    outf.write('\n')
                    continue

                cols = token_line.strip('\n').split('\t')
                if len(cols) < 10:
                    num_bad_tokens += 1
                    continue

                # Skip multiword tokens and empty nodes
                if '-' in cols[0] or '.' in cols[0]:
                    num_multiword_tokens += 1
                    continue
                if '.' in cols[0]:
                    num_empty_tokens += 1
                    continue

                cols[3] = tag_map.get(cols[3], cols[3])

                outf.write('\t'.join(cols))
                outf.write('\n')
            outf.write('\n')

    logging.info(f'Processed {num_sentences} sentences with {num_tokens} tokens')
    if num_bad_sentences > 0:
        logging.warning(f'Skipped {num_bad_sentences} sentences with invalid POS tags')
    if num_bad_tokens > 0:
        logging.warning(f'Skipped {num_bad_tokens} tokens that had too few columns')
    if num_multiword_tokens > 0:
        logging.info(f'Skipped {num_multiword_tokens} multiword tokens')
    if num_empty_tokens > 0:
        logging.info(f'Skipped {num_empty_tokens} empty tokens')


def split_into_sentences(lines):
    sentence = []
    for line in lines:
        line = line.strip('\n')
        if line == '':
            if sentence:
                yield sentence

            sentence = []
        else:
            sentence.append(line)

    if sentence:
        yield sentence


def sentence_has_invalid_tags(sentence):
    token_columns = (line.split('\t') for line in sentence)
    pos_tags = (line[3] if len(line) > 3 else '' for line in token_columns)
    return any('|' in tag for tag in pos_tags)


if __name__ == '__main__':
    main()
