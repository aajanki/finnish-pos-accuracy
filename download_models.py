import os
import spacy_udpipe
import stanfordnlp

spacy_udpipe.download('fi-tdt')
spacy_udpipe.download('fi')

os.makedirs('data/stanfordnlp_resources', exist_ok=True)
stanfordnlp.download('fi_tdt', resource_dir='data/stanfordnlp_resources', force=True)
