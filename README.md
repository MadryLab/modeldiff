<h1>ModelDiff: A Framework for Comparing Learning Algorithms</h1>

This repository contains the code for *ModelDiff*, a framework for feature-based comparisons of ML models trained with two different learning algorithms:

**ModelDiff: A Framework for Comparing Learning Algorithms** <br>
*Harshay Shah\*, Sung Min Park\*, Andrew Ilyas\*, Aleksander Madry* <br>
**Paper**: https://arxiv.org/abs/2211.12491 <br>
**Blog post**: http://gradientscience.org/modeldiff/

```bibtex
@inproceedings{shah2022modeldiff,
  title={ModelDiff: A Framework for Comparing Learning Algorithms},
  author = {Harshay Shah and Sung Min Park and Andrew Ilyas and Aleksander Madry},
  booktitle = {ArXiv preprint arXiv:2211.12491},
  year = {2022}
}
```

## Overview
<p align='center'><img src="static/visual_summary.png"/></p>

The figure above summarizes our algorithm comparisos framework, *ModelDiff*.
- First, our method computes example-level data attributions (e.g., using [datamodels](https://gradientscience.org/datamodels-1/) or [TRAK](http://gradientscience.org/trak/)) for both learning algorithms (part A). In part B, we identify directions (in training set space) that are specific to each algorithm using *residual datamodels*.
- Then, we run PCA on the residual datamodels (part C) to find a set of *distinguishing training directions*---weighted combinations of training examples that disparately impact predictions of models trained with different algorithms. Each distinguishing direction surfaces a distinguishing subpopulation, from which we infer a testable *distinguishing transformation* (part D) that significantly impacts predictions of models trained with one algorithm but not the other.

In our [paper](https://arxiv.org/abs/2211.12491), we apply *ModelDiff* to three case studies that compare models trained with/without standard data augmentation, with/without ImageNet pre-training, and with different SGD hyperparameters. As shown below, in all three cases, our framework allows us to pinpoint concrete ways in which the two algorithms being compared differ:

<p align='center'>
        <img src="static/case_studies.jpg"/>
</p>



## Getting started

1. Clone the repo: `git clone git@github.com:MadryLab/modeldiff.git`

2. Our code relies on the FFCV Library. To install this library along with other dependencies including PyTorch, follow the instructions below:
    ```
        conda create -n ffcv python=3.9 cupy pkg-config compilers libjpeg-turbo opencv pytorch torchvision cudatoolkit=11.3 numba -c pytorch -c conda-forge
        conda activate ffcv

        cd <REPO-DIR>
        pip install -r requirements.txt
    ```

3. Setup datasets. We use CIFAR-10 ([torchvision](https://pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html)), Waterbirds ([WILDS](https://github.com/p-lambda/wilds)), and Living17 ([BREEDS](https://github.com/MadryLab/BREEDS-Benchmarks)). Also, change the `DATA_DIR` path in `src/data/datasets.py` to the parent directory of ImageNet data.

4. Our framework essentially relies on data attributions (e.g., datamodel or TRAK scores) to identify distinguishing features. **Download pre-computed datamodel (and TRAK) scores for these case studies from [here](https://www.dropbox.com/s/9ohxrrba8wb2piv/datamodels.zip?dl=0) and unzip them into  `datamodels/`**

That's it! Now you can run notebooks (one corresponding to each case study in `analysis/`), or take a look at our scripts (in `counterfactuals/`) that evaluate the average treatment effect of distinguishing feature transformations identified via ModelDiff.

## Maintainers

* [Harshay Shah](https://twitter.com/harshays_)
* [Sung Min Park](https://twitter.com/smsampark)
* [Andrew Ilyas](https://twitter.com/andrew_ilyas)
