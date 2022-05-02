import os
import subprocess
import sys
import spacy_udpipe
import stanza
import trankit
from uralicNLP import uralicApi

print('Downloading spaCy models...')
subprocess.run(['python', '-m', 'spacy', 'download', 'fi_core_news_sm'], stdout=sys.stdout, check=True)
subprocess.run(['python', '-m', 'spacy', 'download', 'fi_core_news_md'], stdout=sys.stdout, check=True)
subprocess.run(['python', '-m', 'spacy', 'download', 'fi_core_news_lg'], stdout=sys.stdout, check=True)

print('Downloading UDPipe model...')
spacy_udpipe.download('fi-tdt')
spacy_udpipe.download('fi')

print('Downloading Stanza model...')
os.makedirs('models/stanza_resources', exist_ok=True)
stanza.download('fi', model_dir='models/stanza_resources')

print('Downloading Trankit model...')
# Initializing the trankit Pipeline will download the models if they are not
# already downloaded
trankit.Pipeline('finnish', cache_dir='models/trankit_resources')
trankit.Pipeline('finnish', embedding='xml-roberta-large',
                 cache_dir='models/trankit_resources')

print('Downloading UralicNLP model...')
uralicApi.download("fin")
