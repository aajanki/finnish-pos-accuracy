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
