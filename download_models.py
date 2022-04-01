import os
import spacy_udpipe
import stanza

spacy_udpipe.download('fi-tdt')
spacy_udpipe.download('fi')

os.makedirs('data/stanza_resources', exist_ok=True)
stanza.download('fi', model_dir='data/stanza_resources')
