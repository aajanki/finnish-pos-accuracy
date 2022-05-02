from pathlib import Path
from preprocess_data import split_into_sentences


class Dataset:
    def __init__(self, name, datapath):
        self.name = name
        self.datapath = datapath
        self.sentences = self._parse_conllu(open(datapath))

    def _parse_conllu(self, f):
        sentences = []
        for sentence in split_into_sentences(f.readlines()):
            tokens = []
            for line in sentence:
                if line and not line.startswith('#'):
                    fields = line.split('\t')
                    assert len(fields) == 10

                    text = fields[1].replace(' ', '')
                    lemma = fields[2].replace(' ', '')
                    space_after = fields[9] != 'SpaceAfter=No'
                    tokens.append(Token(text, lemma, fields[3], space_after))
            sentences.append(TestSentence(tokens))
        return sentences

    def count_tokens(self):
        return sum(s.count_tokens() for s in self.sentences)


class Token:
    def __init__(self, text, lemma, pos, space_after):
        self.text = text
        self.lemma = lemma
        self.pos = pos
        self.space_after = space_after


class TestSentence:
    def __init__(self, tokens):
        self.tokens = tokens

    def text(self):
        s = []
        for t in self.tokens:
            s.append(t.text)
            if t.space_after:
                s.append(' ')
        return ''.join(s).rstrip(' ')

    def lemmas(self):
        return [t.lemma for t in self.tokens]

    def pos(self):
        return [t.pos for t in self.tokens]

    def count_tokens(self):
        return len(self.tokens)


def gold_path(testset_name):
    return {
        'UD_Finnish_TDT': Path('data/preprocessed/UD_Finnish-TDT/fi_tdt-ud-test.conllu'),
        'ftb1u': Path('data/preprocessed/ftb1/ftb1u_sample.tsv'),
        'ftb2-news': Path('data/preprocessed/ftb2/FinnTreeBank_2/news-samples_tab.txt'),
        'ftb2-sofie': Path('data/preprocessed/ftb2/FinnTreeBank_2/sofie12_tab.txt'),
        'ftb2-wikipedia': Path('data/preprocessed/ftb2/FinnTreeBank_2/wikipedia-samples_tab.txt'),
    }.get(testset_name)


all_testsets = [
    Dataset('UD_Finnish_TDT', gold_path('UD_Finnish_TDT')),
    Dataset('ftb1u', gold_path('ftb1u')),
    Dataset('ftb2-news', gold_path('ftb2-news')),
    Dataset('ftb2-sofie', gold_path('ftb2-sofie')),
    Dataset('ftb2-wikipedia', gold_path('ftb2-wikipedia'))
]
