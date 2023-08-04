### Note 

Before running the notebooks:

1. **Download data attribution scores.**  Just run `download_scores.sh` or (a) download pre-computed datamodel (and TRAK) scores corresponding to each case study in the paperfrom [here](https://www.dropbox.com/s/9ohxrrba8wb2piv/datamodels.zip?dl=0) and (b) unzip them into `datamodels/`.

2. **Setup datasets.**  We use CIFAR-10 ([torchvision](https://pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html)), Waterbirds ([WILDS](https://github.com/p-lambda/wilds)), and Living17 ([BREEDS](https://github.com/MadryLab/BREEDS-Benchmarks)). Also, change the `DATA_DIR` path in `src/data/datasets.py` to the parent directory of ImageNet data
