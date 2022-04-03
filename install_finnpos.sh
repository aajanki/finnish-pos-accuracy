#!/bin/sh

set -eu

echo "Compiling FinnPos"
cd models/FinnPos

wget --directory-prefix share/finnpos/omorfi https://github.com/mpsilfve/FinnPos/releases/download/v0.1-alpha/morphology.omor.hfst.gz
gunzip share/finnpos/omorfi/morphology.omor.hfst.gz

make && make ftb-omorfi-tagger
