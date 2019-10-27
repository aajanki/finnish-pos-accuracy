import re
import spacy_udpipe
from voikko import libvoikko

inflection_postfix_re = re.compile(r'(.{2,}):\w{1,4}$')


class UDPipe():
    def __init__(self, language='fi-tdt'):
        self.name = f'UDPipe-{language}'
        self.nlp = spacy_udpipe.load(language)

    def parse(self, tokens):
        text = ' '.join(tokens)
        doc = self.nlp(text)

        pos = [t.pos_ for t in doc]
        lemmas = [t.lemma_ for t in doc]

        return (lemmas, pos)


class Voikko():
    def __init__(self):
        self.name = 'voikko'
        self.voikko = libvoikko.Voikko('fi')
        self.tag_map = {
            'nimisana': 'NOUN',
            'laatusana': 'ADJ',
            'nimisana_laatusana': 'ADJ',
            'teonsana': 'VERB',
            'seikkasana': 'ADV',
            'asemosana': 'PRON',
            'suhdesana': 'ADP',
            'huudahdussana': 'INTJ',
            'sidesana': 'CCONJ',
            'etunimi': 'PROPN',
            'sukunimi': 'PROPN',
            'paikannimi': 'PROPN',
            'nimi': 'PROPN',
            'kieltosana': 'AUX',
            'lyhenne': 'ADV',
            'lukusana': 'NUM',
            'etuliite': 'X'
        }

    def parse(self, tokens):
        lemmas = []
        pos = []
        for t in tokens:
            analyzed = self.voikko.analyze(t)
            if analyzed:
                try:
                    i = [x.get('BASEFORM', '_').lower() for x in analyzed].index(t.lower())
                except ValueError:
                    i = 0

                baseform = analyzed[i].get('BASEFORM', t)
                lemmas.append(baseform)

                if baseform in ['olla', 'voida']:
                    tag = 'AUX'
                else:
                    word_class = analyzed[i].get('CLASS', 'X')
                    tag = self.tag_map[word_class]
                pos.append(tag)
            else:
                t_without_inflection = inflection_postfix_re.sub(r'\1', t)
                lemmas.append(t_without_inflection)

                if all(x in '.,?!:;()[]{}"”\'-+…' for x in t):
                    tag = 'PUNCT'
                elif t.istitle() or t.isupper(): # Name or acronym
                    tag = 'PROPN'
                elif all(x.isdigit() or x.isspace() for x in t): # 50 000
                    tag = 'NUM'
                else:
                    tag = 'NOUN' # guess
                pos.append(tag)

        return lemmas, pos
