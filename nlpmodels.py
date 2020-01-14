import io
import logging
import os
import re
import subprocess
import time
import requests
import spacy
import spacy_udpipe
import stanfordnlp
from spacy_stanfordnlp import StanfordNLPLanguage
from voikko import libvoikko

inflection_postfix_re = re.compile(r'(.{2,}):\w{1,4}$')


class PosLemmaToken():
    def __init__(self, pos, lemma):
        self.pos_ = pos
        self.lemma_ = lemma


class UDPipe():
    def __init__(self, language='fi-tdt'):
        self.name = f'UDPipe-{language}'
        self.nlp = spacy_udpipe.load(language)

    def parse(self, tokens):
        text = ' '.join(tokens)
        return process_spacy(self.nlp, text)

    def terminate(self):
        pass


class StanfordNLP():
    def __init__(self):
        self.name = 'StanfordNLP'
        self.snlp = stanfordnlp.Pipeline(lang='fi', models_dir='data/stanfordnlp_resources')
        self.nlp = StanfordNLPLanguage(self.snlp)

    def parse(self, tokens):
        text = ' '.join(tokens)
        return process_spacy(self.nlp, text)

    def terminate(self):
        pass


class Voikko():
    def __init__(self):
        self.name = 'Voikko'
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

    def terminate(self):
        pass

    def _choose_baseform(self, analyzed, orig):
        if not analyzed:
            return (None, None)

        try:
            bases = [x.get('BASEFORM', '_').lower() for x in analyzed]
            i = bases.index(orig.lower())
        except ValueError:
            i = 0

        return (analyzed[i].get('BASEFORM', orig), analyzed[i].get('CLASS', 'X'))


class TurkuNeuralParser():
    def __init__(self):
        self.name = 'Turku-neural-parser'
        self.docker_tag = '1.0.2-fi-en-sv-cpu'
        self.port = 15000
        self.container_name = None
        self.start_server()

    def parse(self, words):
        tokens = self.nlp(' '.join(words))
        pos = [t.pos_ for t in tokens]
        lemmas = [t.lemma_ for t in tokens]
        return (lemmas, pos)

    def nlp(self, text):
        tokens = []
        response = self._send_request(text)
        for line in io.StringIO(response).readlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            fields = line.split('\t')
            assert len(fields) == 10
            tokens.append(PosLemmaToken(fields[3], fields[2]))

        return tokens

    def start_server(self):
        if self.container_name is not None:
            return

        container_name = 'turku_parser'
        docker_image = f'turkunlp/turku-neural-parser:{self.docker_tag}'
        command = ['docker', 'run', '--name', container_name, '-d',
                   '-p', str(self.port) + ':7689', docker_image,
                   'server', 'fi_tdt', 'parse_plaintext']

        logging.info('starting the server')

        if self.sudo_needed():
            command = ['sudo'] + command
            logging.warning('This might ask for the sudo password for launching '
                            'the Turku-neural-parser docker container')

        subprocess.run(command, check=True)
        self.container_name = container_name

        time.sleep(30)

    def stop_server(self):
        if self.container_name is None:
            return

        command_stop = ['docker', 'stop', self.container_name]
        command_rm = ['docker', 'rm', self.container_name]

        logging.info('stopping the server')
        if self.sudo_needed():
            command_stop = ['sudo'] + command_stop
            command_rm = ['sudo'] + command_rm
            logging.warning('This might ask for the sudo password for '
                            'stopping the docker container')

        subprocess.run(command_stop)
        subprocess.run(command_rm)
        self.container_name = None

    def terminate(self):
        self.stop_server()

    def sudo_needed(self):
        return os.environ.get('DOCKER_NEEDS_SUDO') is not None

    def _send_request(self, text):
        logging.debug(f'Sending request: {text}')

        server_url = f'http://localhost:{str(self.port)}'
        r = requests.post(server_url,
                          data=text.encode('utf-8'),
                          headers={'Content-Type': 'text/plain; charset=utf-8'})
        r.raise_for_status()
        return r.text


class FinnPos():
    def __init__(self):
        self.name = 'FinnPos'

    def parse(self, words):
        tokens = self.nlp(words)
        pos = [t.pos_ for t in tokens]
        lemmas = [t.lemma_ for t in tokens]
        return (lemmas, pos)

    def nlp(self, tokens):
        text='\n'.join(tokens)
        p = subprocess.run(['data/FinnPos/bin/ftb-label'], input=text,
                           stdout=subprocess.PIPE, encoding='utf-8', check=True)

        tokens = []
        for line in p.stdout.split('\n'):
            line = line.strip()
            if not line:
                continue

            columns = line.split('\t')
            assert len(columns) == 5, 'Unexpected number of columns ' \
                f'in the ftb-label output. Expected 5, got {len(columns)}'

            lemma = columns[2]
            pos = self._finnpos_to_upos(columns[3])
            tokens.append(PosLemmaToken(pos, lemma))

        return tokens

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

    def terminate(self):
        pass


class SpacyFiExperimental():
    def __init__(self):
        self.name = 'spacy-fi'
        self.nlp = spacy.load('fi_experimental_web_md')

    def parse(self, tokens):
        text = ' '.join(tokens)
        return process_spacy(self.nlp, text)

    def terminate(self):
        pass


def process_spacy(nlp, text):
    doc = list(nlp.pipe([text], disable=['ner', 'parser']))[0]
    pos = [t.pos_ for t in doc]
    lemmas = [t.lemma_ for t in doc]
    return (lemmas, pos)
