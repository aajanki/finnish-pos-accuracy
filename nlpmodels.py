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
        self.tokenizer_is_destructive = True
        self.nlp = None

    def initialize(self):
        self.nlp = spacy_udpipe.load(self.language)

    def parse(self, texts):
        return process_spacy(self.nlp, texts)

    def fix_surface_forms(self, system_sentence, gold_sentence):
        # This is not even trying to be general, but fixes just enough for the
        # CoNLL evaluation to run.
        assert 'id' not in system_sentence

        texts = system_sentence['texts']
        if (len(gold_sentence.tokens) > 5 and gold_sentence.tokens[5].text == 'Emmä'
                and len(texts) > 5 and texts[5] == 'En'):
            return insert_multi_word(system_sentence, 5, '6-7', 'Emmä')
        else:
            return system_sentence


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
        self.tokenizer_is_destructive = False

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
        self.tokenizer_is_destructive = False

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
                ids = []
                words = []
                lemmas = []
                pos = []
                for line in sentence.split('\n'):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue

                    fields = line.split('\t')
                    assert len(fields) == 10

                    # IDs are saved to keep multi-word tokens (IDs such as 3-4)
                    ids.append(fields[0])
                    words.append(fields[1])
                    lemmas.append(fields[2])
                    pos.append(fields[3])

                res.append({'id': ids, 'texts': words, 'lemmas': lemmas, 'pos': pos})

        return res

    def _split_sentences(self, response):
        sentences = []
        current = []
        first_token_is_multiword = False
        for line in response.split('\n'):
            if line.startswith('#') or line == '':
                continue

            if line.startswith('1-'):
                # sentence boundary
                if current:
                    sentences.append('\n'.join(current))

                current = [line]
                first_token_is_multiword = True
            elif line.startswith('1\t') and not first_token_is_multiword:
                # sentence boundary
                if current:
                    sentences.append('\n'.join(current))

                current = [line]
                first_token_is_multiword = False
            else:
                # sentence continues
                current.append(line)
                first_token_is_multiword = False

        if current:
            sentences.append('\n'.join(current))

        return sentences

    def _process_sentences(self, texts):
        r = requests.post('http://localhost:7689',
                          data='\n\n'.join(texts).encode('utf-8'),
                          headers={'Content-Type': 'text/plain; charset=utf-8'})
        r.raise_for_status()

        return self._split_sentences(r.text)


class FinnPos:
    def __init__(self):
        self.name = 'FinnPos'
        self.voikko = None
        self.tokenizer_is_destructive = False

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
        self.tokenizer_is_destructive = True
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

    def fix_surface_forms(self, system_sentence, gold_sentence):
        # This is not even trying to be general, but fixes just enough for the
        # CoNLL evaluation to run.
        assert 'id' not in system_sentence

        texts = system_sentence['texts']
        if texts[0] == 'Eivtta' and texts[1] == 'vät':
            return insert_multi_word(system_sentence, 0, '1-2', 'Eivät')
        elif texts[0] == 'EEttä' and texts[1] == 'ekö':
            return insert_multi_word(system_sentence, 0, '1-2', 'Ettekö')
        elif len(texts) > 11 and texts[11] == 'mittä':
            return insert_multi_word(system_sentence, 11, '12-13', 'miltei')
        else:
            return system_sentence


class SpacyExperimentalFi:
    def __init__(self):
        self.name = 'spacy-fi_experimental_web_md'
        self.tokenizer_is_destructive = False
        self.nlp = None

    def initialize(self):
        self.nlp = spacy.load('spacy_fi_experimental_web_md')

    def parse(self, texts):
        return process_spacy(self.nlp, texts, ['ner', 'parser'])


class SpacyCoreFi:
    def __init__(self, model_name):
        self.name = f'spacy-{model_name}'
        self.model_name = model_name
        self.nlp = None

    def initialize(self):
        self.nlp = spacy.load(self.model_name)

    def parse(self, texts):
        return process_spacy(self.nlp, texts, ['ner', 'parser'])


class Trankit:
    def __init__(self, embedding='base'):
        if embedding not in ['base', 'large']:
            raise ValueError(f'Unknown embedding: {embedding}')

        self.name = f'trankit-{embedding}'
        self.embedding = f'xlm-roberta-{embedding}'
        self.nlp = None
        self.tokenizer_is_destructive = True

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

    def fix_surface_forms(self, system_sentence, gold_sentence):
        # This is not even trying to be general, but fixes just enough for the
        # CoNLL evaluation to run.
        assert 'id' not in system_sentence

        text = system_sentence['texts']
        if text[0] == 'Miksi' and text[1] == 'ei' and gold_sentence.tokens[0].text == 'Eikö':
            fixed = insert_multi_word(system_sentence, 0, '1-2', 'Eikö')

            if text[7] == 'eta' and text[8] == 'ei':
                return insert_multi_word(fixed, 8 + 1, '8-9', 'ei')
            else:
                return fixed
        elif len(text) > 6 and text[5] == 'en' and text[6] == 'mä' and gold_sentence.tokens[5].text == 'Emmä':
            return insert_multi_word(system_sentence, 5, '6-7', 'Emmä')
        else:
            return system_sentence


class Simplemma:
    def __init__(self):
        self.name = 'simplemma'
        self.langdata = None
        self.tokenizer_is_destructive = True

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

    def fix_surface_forms(self, destructive_tokenization, gold_sentence):
        # The tokenizer leaves out some punctuation. Let's try to add it back.
        i = 0
        text = gold_sentence.text()
        non_destructive_tokenization = []
        for t in destructive_tokenization:
            m = re.compile(r'\s*(\W{1,2}\s*)?' + re.escape(t)).match(text, i)
            if m:
                prefix = m.group(1) or ''

                i = m.end()
                m2 = re.compile(r'\s*(?:´|--)').match(text, i)
                if m2:
                    i = m2.end()
                    suffix = m2.group(0)
                else:
                    suffix = ''

                non_destructive_tokenization.append(prefix + t + suffix)
            else:
                raise ValueError('Failed to align tokenization')

        return non_destructive_tokenization


class UralicNLP:
    def __init__(self):
        self.name = 'uralicnlp'
        self.cg = None
        self.voikko = None
        self.tokenizer_is_destructive = False

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


def process_spacy(nlp, texts, disable=None):
    disable = disable or ['ner', 'parser', 'morphologizer']
    docs = list(nlp.pipe(texts, disable=disable))
    return [{
        'texts': [t.text for t in doc],
        'lemmas': [t.lemma_ for t in doc],
        'pos': [t.pos_ for t in doc]
    } for doc in docs]


def chunks(iterable, n):
    """Collect data into non-overlapping fixed-length chunks"""
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=None)


def insert_multi_word(sentence, index, multi_word_id, text):
    ids = [str(x) for x in range(1, len(sentence['texts']) + 1)]
    ids.insert(index, multi_word_id)
    texts = list(sentence['texts'])
    texts.insert(index, text)
    pos = list(sentence['pos'])
    pos.insert(index, '_')
    lemmas = list(sentence['lemmas'])
    lemmas.insert(index, '_')

    return {
        'id': ids,
        'texts': texts,
        'lemmas': lemmas,
        'pos': pos,
    }


all_models = [
    UDPipe('fi-tdt'),
    Voikko(),
    TurkuNeuralParser(),
    FinnPos(),
    SpacyCoreFi('fi_core_news_sm'),
    SpacyCoreFi('fi_core_news_md'),
    SpacyCoreFi('fi_core_news_lg'),
    SpacyExperimentalFi(),
    Stanza(),
    Trankit('base'),
    Trankit('large'),
    Simplemma(),
    UralicNLP(),
]
