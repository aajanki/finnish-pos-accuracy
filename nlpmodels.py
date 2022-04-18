import re
import subprocess
import sys
import time
import requests
import spacy
import spacy_udpipe
import stanza
import trankit
import simplemma
from itertools import zip_longest
from voikko import libvoikko
from uralicNLP.cg3 import Cg3

inflection_postfix_re = re.compile(r'(.{2,}):\w{1,4}$')


class PosLemmaToken:
    def __init__(self, pos, lemma):
        self.pos_ = pos
        self.lemma_ = lemma


class UDPipe:
    def __init__(self, language='fi-tdt'):
        self.name = f'UDPipe-{language}'
        self.language = language
        self.nlp = None

    def initialize(self):
        self.nlp = spacy_udpipe.load(self.language)

    def parse(self, texts):
        return process_spacy(self.nlp, texts)


class Voikko:
    def __init__(self):
        self.name = 'Voikko'
        self.voikko = None
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

    def initialize(self):
        self.voikko = libvoikko.Voikko('fi')

    def parse(self, texts):
        res = []
        for text in texts:
            words = []
            lemmas = []
            pos = []
            for t in self.tokenize(text):
                words.append(t)
                analyzed = self.voikko.analyze(t)
                lemmas.append(self.analyzed_to_lemma(analyzed, t))
                pos.append(self.analyzed_to_pos_tag(analyzed, t))
            res.append({'texts': words, 'lemmas': lemmas, 'pos': pos})
        return res

    def tokenize(self, text):
        return [
            t.tokenText for t in self.voikko.tokens(text)
            if t.tokenTypeName != 'WHITESPACE'
        ]

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


class TurkuNeuralParser:
    def __init__(self):
        self.name = 'Turku-neural-parser'
        self.server_process = None

    def __del__(self):
        self.stop()

    def initialize(self):
        if self.server_process is None:
            print('Starting Turku-neural-parser server...')
            env = {
                'TNPP_MODEL': 'models/models_fi_tdt_dia/pipelines.yaml',
                'TNPP_PIPELINE': 'parse_plaintext',
                'TNPP_PORT': '7689',
                'TNPP_MAX_CHARS': '15000',
                'FLASK_APP': 'models/Turku-neural-parser-pipeline/tnpp_serve.py',
            }
            args = [
                'venv/bin/flask', 'run', '--host', '0.0.0.0', '--port', '7689'
            ]
            self.server_process = subprocess.Popen(
                args, stdout=sys.stdout, stderr=sys.stderr, env=env
            )

            server_ready = False
            attempt = 0
            max_retries = 10
            while not server_ready:
                time.sleep(5)

                try:
                    self._process_sentences(['ABC, kissa kävelee'])
                    server_ready = True
                except requests.exceptions.ConnectionError:
                    attempt += 1
                    if attempt >= max_retries:
                        self.stop()
                        raise RuntimeError('Turku-neural-parser server failed to start')

    def stop(self):
        if self.server_process is not None:
            print('Shutting down Turku-neural-parser server...')
            self.server_process.terminate()
            self.server_process.wait()
            self.server_process = None

    def parse(self, texts):
        res = []
        for batch in chunks(texts, 50):
            batch = [x for x in batch if x is not None]
            sentences = self._process_sentences(batch)
            for sentence in sentences:
                words = []
                lemmas = []
                pos = []
                for line in sentence.split('\n'):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue

                    fields = line.split('\t')
                    assert len(fields) == 10

                    words.append(fields[1])
                    lemmas.append(fields[2])
                    pos.append(fields[3])

                res.append({'texts': words, 'lemmas': lemmas, 'pos': pos})

        return res

    def split_sentences(self, response):
        sentences = []
        for block in response.split('\n\n'):
            if block.startswith('# newpar') or block.startswith('# newdoc') or not sentences:
                # sentence boundary
                sentences.append(block)
            else:
                # sentence continues
                sentences[-1] = sentences[-1] + block

        return sentences

    def _process_sentences(self, texts):
        r = requests.post('http://localhost:7689',
                          data='\n\n'.join(texts).encode('utf-8'),
                          headers={'Content-Type': 'text/plain; charset=utf-8'})
        r.raise_for_status()

        return self.split_sentences(r.text)


class FinnPos:
    def __init__(self):
        self.name = 'FinnPos'
        self.voikko = None

    def initialize(self):
        self.voikko = libvoikko.Voikko('fi')

    def parse(self, texts):
        token_lines = '\n\n'.join('\n'.join(self.tokenize(text)) for text in texts)
        p = subprocess.run(['models/FinnPos/bin/ftb-label'], input=token_lines,
                           stdout=subprocess.PIPE, encoding='utf-8', check=True)
        sentences = p.stdout.split('\n\n')
        if len(sentences) == len(texts) + 1:
            assert sentences[-1] == ''
            sentences.pop(-1)

        res = []
        for sentence in sentences:
            words = []
            lemmas = []
            pos = []
            for token in sentence.split('\n'):
                token = token.strip()
                if not token:
                    continue

                columns = token.split('\t')
                assert len(columns) == 5, 'Unexpected number of columns ' \
                    f'in the ftb-label output. Expected 5, got {len(columns)}'

                words.append(columns[0])
                lemmas.append(columns[2])
                pos.append(self._finnpos_to_upos(columns[3]))

            res.append({'texts': words, 'lemmas': lemmas, 'pos': pos})

        return res

    def tokenize(self, text):
        return [
            t.tokenText for t in self.voikko.tokens(text)
            if t.tokenTypeName != 'WHITESPACE'
        ]

    def _finnpos_to_upos(self, tag):
        if tag.startswith('[POS=NOUN]') and '[PROPER=PROPER]' in tag:
            return 'PROPN'
        elif tag.startswith('[POS=NOUN]'):
            return 'NOUN'
        elif tag.startswith('[POS=VERB]'):
            return 'VERB'
        elif tag.startswith('[POS=ADJECTIVE]'):
            return 'ADJ'
        elif tag.startswith('[POS=PRONOUN]'):
            return 'PRON'
        elif tag.startswith('[POS=ADVERB]'):
            return 'ADV'
        elif tag.startswith('[POS=ADPOSITION]'):
            return 'ADP'
        elif tag.startswith('[POS=NUMERAL]'):
            return 'NUM'
        elif tag.startswith('[POS=PUNCTUATION]'):
            return 'PUNCT'
        elif tag == '[POS=PARTICLE]|[SUBCAT=INTERJECTION]':
            return 'INTJ'
        elif tag == '[POS=PARTICLE]|[SUBCAT=CONJUNCTION]|[CONJ=COORD]':
            return 'CCONJ'
        elif tag.startswith('[POS=PARTICLE]'):
            return 'PART'
        elif tag.startswith('[POS=UNKNOWN]') or tag.startswith('[POS=TRUNCATED]'):
            return 'X'
        else:
            assert False, f'Unknown tag: {tag}'


class Stanza:
    def __init__(self):
        self.name = 'stanza'
        self.nlp = None

    def initialize(self):
        self.nlp = stanza.Pipeline(lang='fi',
                                   dir='models/stanza_resources',
                                   processors='tokenize,mwt,pos,lemma')

    def parse(self, texts):
        in_docs = [stanza.Document([], text=t) for t in texts]
        res = []
        for doc in self.nlp(in_docs):
            words = []
            lemmas = []
            pos = []
            for sent in doc.sentences:
                for w in sent.words:
                    words.append(w.text)
                    lemmas.append(w.lemma)
                    pos.append(w.pos)
            res.append({'texts': words, 'lemmas': lemmas, 'pos': pos})
        return res


class SpacyFiExperimental:
    def __init__(self):
        self.name = 'spacy-fi'
        self.nlp = None

    def initialize(self):
        self.nlp = spacy.load('spacy_fi_experimental_web_md')

    def parse(self, texts):
        return process_spacy(self.nlp, texts)


class Trankit:
    def __init__(self, embedding='base'):
        if embedding not in ['base', 'large']:
            raise ValueError(f'Unknown embedding: {embedding}')

        self.name = f'trankit-{embedding}'
        self.embedding = f'xlm-roberta-{embedding}'
        self.nlp = None

    def initialize(self):
        self.nlp = trankit.Pipeline('finnish',
                                    embedding=self.embedding,
                                    cache_dir='models/trankit_resources')

    def parse(self, texts):
        res = []
        for text in texts:
            doc = self.nlp(text, is_sent=True)
            words = []
            lemmas = []
            pos = []
            for token in doc['tokens']:
                if 'expanded' in token:
                    for exp_token in token['expanded']:
                        words.append(exp_token['text'])
                        lemmas.append(exp_token['lemma'])
                        pos.append(exp_token['upos'])
                else:
                    words.append(token['text'])
                    lemmas.append(token['lemma'])
                    pos.append(token['upos'])

            res.append({'texts': words, 'lemmas': lemmas, 'pos': pos})

        return res


class Simplemma:
    def __init__(self):
        self.name = 'simplemma'
        self.langdata = None

    def initialize(self):
        self.langdata = simplemma.load_data('fi')

    def parse(self, texts):
        res = []
        for text in texts:
            words = []
            lemmas = []
            pos = []
            for t in simplemma.simple_tokenizer(text):
                words.append(t)
                lemmas.append(simplemma.lemmatize(t, self.langdata))
            res.append({'texts': words, 'lemmas': lemmas, 'pos': pos})
        return res


class UralicNLP:
    def __init__(self):
        self.name = 'uralicnlp'
        self.cg = None
        self.voikko = None

    def initialize(self):
        self.cg = Cg3('fin')
        self.voikko = libvoikko.Voikko('fi')

    def parse(self, texts):
        res = []
        for text in texts:
            words = []
            lemmas = []
            pos = []
            tokens = self.tokenize(text)
            for word, analyses in self.cg.disambiguate(tokens):
                words.append(word)
                lemmas.append(analyses[0].lemma)
                pos.append(self._uralic_pos_to_upos(analyses[0].morphology))
            res.append({'texts': words, 'lemmas': lemmas, 'pos': pos})
        return res

    def tokenize(self, text):
        return [
            t.tokenText for t in self.voikko.tokens(text)
            if t.tokenTypeName != 'WHITESPACE'
        ]

    def _uralic_pos_to_upos(self, morphology):
        if 'A' in morphology:
            return 'ADJ'
        elif 'ABBR' in morphology or 'ACR' in morphology:
            if 'Prop' in morphology:
                return 'PROPN'
            else:
                return 'ADV'
        elif 'Adv' in morphology:
            return 'ADV'
        elif 'CC' in morphology:
            return 'CCONJ'
        elif 'CS' in morphology:
            return 'SCONJ'
        elif 'Interj' in morphology:
            return 'INTJ'
        elif 'N' in morphology:
            if 'Prop' in morphology:
                return 'PROPN'
            else:
                return 'NOUN'
        elif 'Num' in morphology:
            return 'NUM'
        elif 'Pcle' in morphology:
            return 'ADV'
        elif 'Po' in morphology or 'Pr' in morphology:
            return 'ADP'
        elif 'Pron' in morphology:
            return 'PRON'
        elif 'Punct' in morphology:
            return 'PUNCT'
        elif 'V' in morphology:
            if '@+FAUXV' in morphology:
                return 'AUX'
            else:
                return 'VERB'
        else:
            return 'X'


def process_spacy(nlp, texts):
    docs = list(nlp.pipe(texts, disable=['ner', 'parser', 'morphologizer']))
    return [{
        'texts': [t.text for t in doc],
        'lemmas': [t.lemma_ for t in doc],
        'pos': [t.pos_ for t in doc]
    } for doc in docs]


def chunks(iterable, n):
    """Collect data into non-overlapping fixed-length chunks"""
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=None)
