#!/bin/sh

set -eu

echo "Downloading and compiling FinnPos"

OLD_WD="$PWD"

git clone https://github.com/mpsilfve/FinnPos data/FinnPos
cd data/FinnPos
git checkout 81c1f735

wget --directory-prefix share/finnpos/omorfi https://github.com/mpsilfve/FinnPos/releases/download/v0.1-alpha/morphology.omor.hfst.gz
gunzip share/finnpos/omorfi/morphology.omor.hfst.gz

make && make ftb-omorfi-tagger

cd "$OLD_WD"
