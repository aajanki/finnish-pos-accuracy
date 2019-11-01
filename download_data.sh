#!/bin/sh

mkdir -p data/test

git clone --branch r2.4 --single-branch --depth 1 https://github.com/UniversalDependencies/UD_Finnish-TDT data/test/UD_Finnish-TDT

wget --directory-prefix data/test/ftb1 http://www.ling.helsinki.fi/kieliteknologia/tutkimus/treebank/sources/ftb1u-v1.zip
unzip -p data/test/ftb1/ftb1u-v1.zip ftb1u-v1/ftb1u.tsv > data/test/ftb1/ftb1u.tsv
