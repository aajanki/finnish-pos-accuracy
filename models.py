import spacy_udpipe
from voikko import libvoikko


class UDPipe():
    def __init__(self):
        self.name = 'UDPipe'
        self.nlp = spacy_udpipe.load('fi-tdt')

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
            'nimisana_laatusana': 'NOUN',
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
                lemmas.append(analyzed[0].get('BASEFORM', t))
                tag = analyzed[0].get('CLASS', 'X')
                pos.append(self.tag_map[tag])
            else:
                lemmas.append(t)

                if len(t) == 1 and t in '.,?!:;()[]{}"\'-+':
                    pos.append('PUNCT')
                else:
                    pos.append('X')

        return lemmas, pos
