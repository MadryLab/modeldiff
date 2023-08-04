
<h1>ModelDiff: A Framework for Comparing Learning Algorithms</h1>

This repository contains the code for *ModelDiff*, a framework for feature-based comparisons of ML models trained with two different learning algorithms:

**ModelDiff: A Framework for Comparing Learning Algorithms** <br>
*Harshay Shah\*, Sung Min Park\*, Andrew Ilyas\*, Aleksander Madry* <br>
**Paper**: https://arxiv.org/abs/2211.12491 <br>
**Blog post**: http://gradientscience.org/modeldiff/

```bibtex
@inproceedings{shah2023modeldiff,
  title={Modeldiff: A framework for comparing learning algorithms},
  author={Shah, Harshay and Park, Sung Min and Ilyas, Andrew and Madry, Aleksander},
  booktitle={International Conference on Machine Learning},
  pages={30646--30688},
  year={2023},
  organization={PMLR}
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


## Basic usage (TODO: use new api here)

```python

# setup datasets
dataset_map = {'train': train_dataset, 'test': test_dataset}

# setup paths to data attribution scores (for both learning algorithms)
scores_1 = ...
scores_2 = ...  # path to score matrix of size [num_train x num_test] 

# run ModelDiff  
K = ... # number of distinguishing directions 
pca = dm_utils.ModelDiff(dm_1, dm_2, K, dataset_map)
# pca.pca_components is a [K x N] matrix where N is train set size
```

Check out [our notebooks](https://github.com/MadryLab/modeldiff/tree/master/analysis) for end-to-end examples of using ModelDiff to analyze the effect of standard data augmentation, ImageNet pre-training, and SGD hyperparameters! 

## Getting started

1. Clone the repo: `git clone git@github.com:MadryLab/modeldiff.git`

2. Our code relies on the FFCV Library. To install this library along with other dependencies including PyTorch, follow the instructions below [TODO: update this]
    ```
        conda create -n ffcv python=3.9 cupy pkg-config compilers libjpeg-turbo opencv pytorch torchvision cudatoolkit=11.3 numba -c pytorch -c conda-forge
        conda activate ffcv

        cd <REPO-DIR>
        pip install -r requirements.txt
    ```
3. Check out [our notebooks](https://github.com/MadryLab/modeldiff/tree/master/notebooks) for end-to-end ModelDiff examples; each notebook corresponds to a case study in our [paper](https://arxiv.org/abs/2211.12491). For each case study, we provide scripts in `counterfactuals/` to test the effect of the distinguishing transformationss (inferred via ModelDiff) on the predictions of  trained using different learning algorithms. 

4. Compute data attribution scores from scratch using datamodels (https://github.com/MadryLab/datamodels) or TRAK (https://github.com/MadryLab/trak) to run ModelDiff for any two learning algorithms. See `notebooks/api_example.ipynb` for an example on how to compute TRAK scores (from scratch) for two learning algorithms and (b) then run ModelDiff to compare these algorithms. [TODO: add this notebook]

## Maintainers

* [Harshay Shah](https://twitter.com/harshays_)
* [Sung Min Park](https://twitter.com/smsampark)
* [Andrew Ilyas](https://twitter.com/andrew_ilyas)
