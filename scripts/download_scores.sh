#!/bin/bash
# Downloads pre-computed datamodel and TRAK scores for all learning algorithms / case studies in the paper
wget -O datamodels.zip https://www.dropbox.com/s/9ohxrrba8wb2piv/datamodels.zip?dl=0
unzip datamodels.zip
rm -rf __MACOSX
rm -rf datamodels.zip
