from torch.utils.data import DataLoader
from torch import Tensor
import torch
import numpy as np
from sklearn.decomposition import PCA

def get_variance_along_directions(X: Tensor, P: Tensor,
                                  device: str = 'cuda',
                                  batch_size: int = 50):
    """
    Compute variance of X along directions P

    Args:
        X: data attribution matrix [num_test x num_train]
        P: directions in train set space [num_comp x num_train]
        device: device to use for computation {'cuda', 'cpu'}
        batch_size: batch size over P (pca directions)

    Returns:
        variances: variance of X along directions P [num_comp]

    Note: for row-normalized X, total variance is 1 so variance equals explained variance
    """
    # normalize and center X
    X = X.float().to(device)
    X = X/X.norm(dim=1).view(-1,1) # normalizing data attribution embeddings
    X = X - X.mean(axis=0) # centering beforehand to compute variance in each direction
    n = float(len(X))

    # batch pca directions
    sampler = torch.utils.data.SequentialSampler(P)
    sampler = torch.utils.data.BatchSampler(sampler, batch_size, drop_last=False)

    # compute variance along directions
    variances = []

    for indices in sampler:
        P_ = P[indices].to(device)
        xw = torch.mm(X, P_.T) # project centered X onto P_
        var = torch.multiply(xw, xw).sum(axis=0)/n # compute variance along P_
        variances.append(var)
        P_ = P_.cpu()

    X = X.cpu()
    torch.cuda.empty_cache()

    variances = torch.cat(variances).cpu().numpy()
    return variances

def get_residual_attributions(scores_A: Tensor, scores_B: Tensor):
    """
    Compute residual data attributions

    Args:
        scores_A: data attributions for model A [num_test x num_train]
        scores_B: data attributions for model B [num_test x num_train]

    Returns:
        residual_scores: residual data attributions [num_test x num_train]
    """
    scores_A = scores_A.cpu().numpy()
    scores_B = scores_B.cpu().numpy()
    assert scores_A.shape == scores_B.shape

    num = np.multiply(scores_A, scores_B).sum(axis=1)
    denom = np.multiply(scores_B, scores_B).sum(axis=1)
    proj = np.multiply((num/denom).reshape(-1,1), scores_B)
    residual_scores = scores_A - proj

    return residual_scores

def residual_pca(scores_A: Tensor, scores_B: Tensor, num_pca_comps: int):
    """
    Computes residual PCA directions

    Args:
        scores_A: data attributions for model A [num_test x num_train]
        scores_B: data attributions for model B [num_test x num_train]
        num_pca_comps: number of top-k pca components to compute

    Returns: a dictionary with the following keys:
        directions: residual pca directions
        projections: residual pca projections
        variances: variance of A and B along residual pca directions
    """
    # get residual scores
    residual_scores = get_residual_attributions(scores_A, scores_B)

    # run pca
    pca = PCA(n_components=num_pca_comps)
    residual_score_projections = pca.fit_transform(residual_scores)

    # pca metadata
    directions = pca.components_

    variances = {
        'A': get_variance_along_directions(scores_A, torch.Tensor(directions)),
        'B': get_variance_along_directions(scores_B, torch.Tensor(directions))
    }

    return {
        'directions': directions,
        'projections': residual_score_projections,
        'variances': variances,
        'scores': {
            'A': scores_A,
            'B': scores_B
        }
    }
