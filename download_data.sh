#!/bin/sh

set -eu

mkdir -p data/test

git clone --branch r2.9 --single-branch --depth 1 https://github.com/UniversalDependencies/UD_Finnish-TDT data/test/UD_Finnish-TDT

wget --directory-prefix data/test/ftb1 http://www.ling.helsinki.fi/kieliteknologia/tutkimus/treebank/sources/ftb1u-v1.zip
unzip -p data/test/ftb1/ftb1u-v1.zip ftb1u-v1/ftb1u.tsv > data/test/ftb1/ftb1u.tsv
python sample_ftb1.py

wget --directory-prefix data/test/ftb2 http://www.ling.helsinki.fi/kieliteknologia/tutkimus/treebank/sources/ftb2.tar.gz
tar --one-top-level=data/test/ftb2 -xzf data/test/ftb2/ftb2.tar.gz \
    FinnTreeBank_2/wikipedia-samples_tab.txt \
    FinnTreeBank_2/news-samples_tab.txt \
    FinnTreeBank_2/sofie12_tab.txt
