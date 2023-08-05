
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


## Usage (TODO: use new api here)

```python
from modeldiff import ModelDiff

# setup models, checkpoints, and loaders
modelA = ...
modelB = ...
ckptsA = [...] 
ckptsB = [...] 
train_loader = ...

# init ModelDiff  
md = ModelDiff(modelA, modelB, ckptsA, ckptsB, train_loader)

# compare models 
val_loader = ...
diff = md.get_A_minus_B(val_loader, num_pca_components=2)

another_val_loader = ...
diff2 = md.get_B_minus_A(another_val_loader, num_pca_components=2)
```

Check out [our notebooks](https://github.com/MadryLab/modeldiff/tree/master/analysis) for end-to-end examples of using ModelDiff to analyze the effect of standard data augmentation, ImageNet pre-training, and SGD hyperparameters! 

## Getting started

1. To use the API, simply
   ```
   pip install modeldiff
   ```
2. For an example usage, check out `notebooks/api_example.ipynb`. In there, we (a) compute TRAK scores (from scratch) for two learning algorithms and (b) then run ModelDiff to compare these algorithms (this is all achieved with one line of code using `modeldiff`!).

3. Check out [our notebooks](https://github.com/MadryLab/modeldiff/tree/master/notebooks) for end-to-end ModelDiff examples; each notebook corresponds to a case study in our [paper](https://arxiv.org/abs/2211.12491). For each case study, we provide scripts in `counterfactuals/` to test the effect of the distinguishing transformationss (inferred via ModelDiff) on the predictions of  trained using different learning algorithms. 

If you want to compute data attribution scores from scratch with a method different from TRAK (e.g. [datamodels](https://github.com/MadryLab/datamodels)), you can pre-compute those yourself and use the `.get_A_minus_B_from_scores()` and `.get_B_minus_A_from_scores` methods.

## Maintainers

* [Harshay Shah](https://twitter.com/harshays_)
* [Sung Min Park](https://twitter.com/smsampark)
* [Andrew Ilyas](https://twitter.com/andrew_ilyas)
