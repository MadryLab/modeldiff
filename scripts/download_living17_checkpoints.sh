#!/bin/bash
# Downloads resnet18 checkpoints trained on the Living17 dataset with and without standard data augmentation

script_dir=$(cd -P -- "$(dirname "$0")" && pwd -P)
ckpt_dir=$script_dir/../checkpoints
mkdir $ckpt_dir
wget -O $ckpt_dir/living17.pt https://www.dropbox.com/scl/fi/2ezqphmgwetw2j477i6r6/living17_ckpts.pt?rlkey=5ep3p1hhw8qfaxdlqjzd9m2q3&dl=0 
