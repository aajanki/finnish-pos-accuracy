import re
import spacy_udpipe
import stanfordnlp
from spacy_stanfordnlp import StanfordNLPLanguage
from voikko import libvoikko

inflection_postfix_re = re.compile(r'(.{2,}):\w{1,4}$')


class UDPipe():
    def __init__(self, language='fi-tdt'):
        self.name = f'UDPipe-{language}'
        self.nlp = spacy_udpipe.load(language)

    def parse(self, tokens):
        text = ' '.join(tokens)
        return process_spacy(self.nlp, text)


class StanfordNLP():
    def __init__(self):
        self.name = 'stanfordnlp'
        self.snlp = stanfordnlp.Pipeline(lang='fi', models_dir='model_resources')
        self.nlp = StanfordNLPLanguage(self.snlp)

    def parse(self, tokens):
        text = ' '.join(tokens)
        return process_spacy(self.nlp, text)


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
            lemmas.append(self.analyzed_to_lemma(analyzed, t))
            pos.append(self.analyzed_to_pos_tag(analyzed, t))

        return lemmas, pos

    def analyzed_to_lemma(self, analyzed, orig):
        if analyzed:
            lemma, _ = self._choose_baseform(analyzed, orig)
        else:
            lemma = inflection_postfix_re.sub(r'\1', orig)

        return lemma

    def analyzed_to_pos_tag(self, analyzed, orig):
        if analyzed:
            baseform, word_class = self._choose_baseform(analyzed, orig)
            if baseform in ['olla', 'voida']:
                tag = 'AUX'
            else:
                tag = self.tag_map[word_class]
        else:
            if all(x in '.,?!:;()[]{}"”\'-+…' for x in orig):
                tag = 'PUNCT'
            elif orig.istitle() or orig.isupper(): # Name or acronym
                tag = 'PROPN'
            elif all(x.isdigit() or x.isspace() for x in orig): # 50 000
                tag = 'NUM'
            else:
                tag = 'NOUN' # guess

        return tag

    def _choose_baseform(self, analyzed, orig):
        if not analyzed:
            return (None, None)

        try:
            bases = [x.get('BASEFORM', '_').lower() for x in analyzed]
            i = bases.index(orig.lower())
        except ValueError:
            i = 0

        return (analyzed[i].get('BASEFORM', orig), analyzed[i].get('CLASS', 'X'))


def process_spacy(nlp, text):
    doc = nlp(text)
    pos = [t.pos_ for t in doc]
    lemmas = [t.lemma_ for t in doc]
    return (lemmas, pos)
