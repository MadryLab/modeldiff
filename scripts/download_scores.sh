#!/bin/bash
# Downloads pre-computed datamodel and TRAK scores for all learning algorithms / case studies in the paper
script_dir=$(cd -P -- "$(dirname "$0")" && pwd -P)
ckpt_dir=$script_dir/../checkpoints
mkdir $ckpt_dir
wget -O $ckpt_dir/datamodels.zip https://www.dropbox.com/s/9ohxrrba8wb2piv/datamodels.zip?dl=0
unzip $ckpt_dir/datamodels.zip
rm -rf $ckpt_dir/__MACOSX
rm -rf $ckpt_dir/datamodels.zip
