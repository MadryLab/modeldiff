import numpy as np
from pathlib import Path
import typing
from matplotlib import pyplot as plt

import torch
import pandas as pd
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from . import common_utils
from . import data_utils
from . import plot_utils


def load_datamodel(path, row_indices=None, col_indices=None,
                    normalize_embeddings=True, center_data=True):
    # returns [num_test x num_train] weight matrix
    X = torch.load(path, map_location='cpu')['weight'].numpy().T
    if row_indices is not None: X = X[row_indices]
    if col_indices is not None: X = X[:, col_indices]
    if normalize_embeddings: X /= np.linalg.norm(X, axis=1).reshape(-1,1)
    if center_data: X = X - X.mean(axis=0)
    return X

def get_skipped_datamodel_jobs(dir_path, min_worker_id=None, max_worker_id=None):
    dir_path = Path(dir_path)
    incomplete = ~np.load(dir_path / '_completed.npy')
    incomplete = np.array(incomplete.nonzero()[0])
    if min_worker_id is not None:
        incomplete = incomplete[incomplete >= min_worker_id]
    if max_worker_id is not None:
        incomplete = incomplete[incomplete < max_worker_id]
    return list(incomplete)

def get_cosine_similarity(dm1, dm2):
    n1 = np.linalg.norm(dm1, axis=1)
    n2 = np.linalg.norm(dm2, axis=1)
    return (dm1*dm2).sum(axis=1)/(n1*n2)

def get_subset_total_influence(X, indices, is_train=False, use_abs=True):
    X = X[:,indices] if is_train else X[indices]
    if use_abs: X = np.abs(X)
    return X.sum(axis=1) if is_train else X.sum(axis=0)

def get_explained_variance(X, P, device, batch_size=100, normalize=True, center=True):
    """
    X: datamodel matrix [num_test x num_train]
    P: pca directions [num_comp x num_train]
    """
    torch.cuda.empty_cache()
    to_torch = lambda A: torch.FloatTensor(A)
    X, P = map(to_torch, [X, P])
    X = X.to(device)
    if normalize: X = X/X.norm(dim=1).view(-1,1)
    if center: X = X - X.mean(axis=0)
    n = float(len(X))

    sampler = torch.utils.data.SequentialSampler(P)
    sampler = torch.utils.data.BatchSampler(sampler, batch_size, drop_last=False)
    exp_vars = []

    for indices in sampler:
        P_ = P[indices].to(device)
        xw = torch.mm(X, P_.T)
        var = torch.multiply(xw, xw).sum(axis=0)/n
        exp_vars.append(var)
        P_ = P_.cpu()

    torch.cuda.empty_cache()
    return torch.cat(exp_vars).cpu().numpy()

def get_residual_datamodel(base_dm, other_dm):
    def _project(X, P):
        num = np.multiply(X, P).sum(axis=1)
        denom = np.multiply(P, P).sum(axis=1)
        proj = np.multiply((num/denom).reshape(-1,1), P)
        orth = X - proj
        return orth, proj

    orth, proj = _project(base_dm, other_dm)

    return {
        'orth': orth,
        'proj': proj
    }

class DatamodelPCABase(object):

    def get_explained_variance(self, datamodels_map):
        ev_map = {}
        for name, X in datamodels_map.items():
            X_norm = X / np.linalg.norm(X, axis=1).reshape(-1,1)
            ev_map[name] = get_explained_variance(X_norm, self.pca_components, 'cpu')
        return ev_map

    def plot_explained_variance(self, datamodels_map, num_components=None):
        exp_vars = self.get_explained_variance(datamodels_map)
        df = pd.DataFrame(exp_vars).reset_index().melt(id_vars='index')
        df.columns = ['index', 'datamodel', 'explained_variance']
        num_components = num_components if num_components else max(df['index'])
        fg = sns.catplot(x='index', y='explained_variance', hue='datamodel', kind='bar', data=df[df['index']<num_components])
        return fg

    def _load_datamodel(self, path):
        # returns [num_test x num_train] weight matrix
        X = torch.load(path, map_location='cpu')['weight'].numpy().T
        if self.row_indices is not None:
            X = X[self.row_indices]
        if self.col_indices is not None:
            X = X[:, self.col_indices]
        if self.normalize_embeddings:
            X /= np.linalg.norm(X, axis=1).reshape(-1,1)
        return X

    def get_explained_variance_ratio_cdf(self, cumulative=True):
        ratio = self.pca_stats['explained_variance_ratio_']
        if cumulative: return np.cumsum(ratio)
        return ratio

    def plot_explained_variance_ratio(self, num_components, label=None, ax=None, **kw):
        if not ax: _, ax = plt.subplots(1,1,figsize=(10,4))
        cdf = self.get_explained_variance_ratio_cdf(cumulative=True)[:num_components]
        plot_kw = dict(lw=2, marker='.', label=label)
        plot_kw.update(kw)
        ax.plot(cdf, **plot_kw)
        common_utils.update_ax(ax, '', '$k^{th}$ principal component',
                               'Explained variance ratio',
                               ticks_fs=13, label_fs=15, title_fs=17)
        return ax

    def plot_principal_component_images(self, component_indices, num_imgs,
                                        ax_grid=None, class_names=None, **row_kw):
        """
        - component_indices: list of principal component indices
        - num_imgs: num_imgs with top-k and bottom-k scores
        - class_names: [index] -> name map
        - kw: img_height, img_width, title_fs, label_fs
        """
        ax_shape = (len(component_indices), num_imgs*2)
        kw = dict(img_height=3, img_width=3, title_fs=16, label_fs=16)
        kw.update(row_kw)

        if ax_grid is not None:
            assert ax_grid.shape == ax_shape, "ax_grid shape wrong"
        else:
            figsize = (kw['img_width']*ax_shape[1], kw['img_height']*ax_shape[0])
            fig, ax_grid = plt.subplots(nrows=ax_shape[0], ncols=ax_shape[1], figsize=figsize)
            if ax_shape[0]==1: ax_grid = [ax_grid]


        for pc_index, ax_row in zip(component_indices, ax_grid):
            self._plot_principal_component_images(pc_index, num_imgs, ax_row=ax_row, class_names=class_names, **kw)

            # add green/red borders
            for idx in range(num_imgs):
                plot_utils.add_axis_border(ax_row[idx], 'green', lw=3)
                plot_utils.add_axis_border(ax_row[num_imgs+idx], 'red', lw=3)

        return fig, ax_grid

    def _get_dataset_index(self, pca_index, split):
        return pca_index

    def _plot_principal_component_images(self, component_index, num_imgs,
                                        ax_row=None, add_indices=False,
                                        class_names=None, **row_kw):
        if ax_row is not None:
            assert len(ax_row) == num_imgs*2, "ax_row shape wrong"

        class_names = class_names if class_names is not None else {}
        get_class_name = lambda idx: class_names.get(idx, f'Class {idx}')
        projections = self.X_proj[:, component_index]

        sorted_indices = np.argsort(projections)[::-1]
        indices = np.concatenate((sorted_indices[:num_imgs], sorted_indices[-num_imgs:]))
        scores = np.round(projections[indices], 2)
        is_single_label = len(self.label_indices)==1

        dset_split = 'train' if self.pca_wrt_train else 'test'
        images, labels, titles = [], [], []
        for idx in indices:
            dset_tuple = self.datasets_map[dset_split][self._get_dataset_index(idx, dset_split)]
            image = dset_tuple[0]
            if is_single_label: idx_labels = dset_tuple[self.label_indices[0]]
            else: idx_labels = tuple([int(dset_tuple[x]) for x in self.label_indices])
            images.append(image)
            labels.append(idx_labels)
            titles.append('#{}'.format(idx))

        labels = [f'{cname} ({sc:.2f})' for cname, sc in zip(list(map(get_class_name, labels)), scores)]
        images = torch.stack(images)

        kw = {}
        kw.update(row_kw)

        titles = titles if add_indices else None
        fig, ax_row = data_utils.plot_image_row(images, titles=titles, labels=labels, axs=ax_row, **kw)
        ratio = self.pca_stats['explained_variance_ratio_'][component_index]*100
        ylabel = f'PC #{component_index} ({ratio:.2f}%)'
        ax_row[0].set_ylabel(ylabel, fontweight='bold', fontsize=kw['label_fs'])

        return fig, ax_row

class DatamodelResidualPCA(DatamodelPCABase):

    def __init__(self, datamodel1_path, datamodel2_path,
                 num_pca_components, datasets_map,
                 label_indices=[1], project_orthogonal=True,
                 pca_kw=None, row_indices=None, col_indices=None,
                 normalize_embeddings=True):
        super().__init__()
        # setup
        self.datamodel1_path = datamodel1_path
        self.datamodel2_path = datamodel2_path
        self.project_orthogonal = project_orthogonal
        self.datasets_map = datasets_map
        self.label_indices = label_indices if label_indices else [1]
        self.row_indices = self.datamodel_indices = row_indices
        self.col_indices = self.datamodel_column_indices = col_indices
        self.normalize_embeddings = normalize_embeddings
        self.num_pca_components = num_pca_components
        self.pca_kw = {} if pca_kw is None else pca_kw
        self.pca_wrt_train = False

        # load datamodels (see clustering base)
        self.X1 = self._load_datamodel(self.datamodel1_path)
        self.X2 = self._load_datamodel(self.datamodel2_path)
        self.X = self._get_projected_datamodel()

        # run pca
        self.pca = PCA(n_components=self.num_pca_components, **self.pca_kw)
        self.X_proj = self.pca.fit_transform(self.X)
        self.pca_stats = vars(self.pca)
        self.pca_components = self.pca.components_

    def _get_projected_datamodel(self):
        def _project(X, P):
            num = np.multiply(X, P).sum(axis=1)
            denom = np.multiply(P, P).sum(axis=1)
            proj = np.multiply((num/denom).reshape(-1,1), P)
            orth = X - proj
            return orth, proj

        orth, proj = _project(self.X1, self.X2)
        return orth if self.project_orthogonal else proj

    def _load_datamodel(self, path):
        # returns [num_test x num_train] weight matrix
        X = torch.load(path, map_location='cpu')['weight'].numpy().T
        if self.row_indices is not None:
            X = X[self.row_indices]
        if self.col_indices is not None:
            X = X[:, self.col_indices]
        if self.normalize_embeddings:
            X /= np.linalg.norm(X, axis=1).reshape(-1,1)
        return X
    
ModelDiff = DatamodelResidualPCA

class InfluenceVisualizer(object):
    """
    Helper class to visualize top/bottom datamodel influencers
    """

    def __init__(self, datamodel, datamodel_split, datasets_map, label_indices=None, label_tuples=None):
        """
        Arguments
            - datamodel: torch.load(datamodel)['weight']
            - datamodel_split: {train, test, val}
            - datasets_map: [split] -> dataset
            - label_tuples: [split] -> [index] -> [label tuple]
        """
        assert datamodel_split in datasets_map
        self.dm = datamodel
        self.dm = self.dm.transpose(0,1)
        self.dm_split = datamodel_split
        self.dsets = datasets_map
        self.label_indices = label_indices or [1]

        if not label_tuples:
            self.label_tuples = {s: self._get_labels(s) for s in self.dsets}
        else:
            self.label_tuples = label_tuples

    def _get_labels(self, split):
        tups = []
        dset = self.dsets[split]
        for tup in dset:
            tups.append([tup[idx] for idx in self.label_indices])
        return tups

    def get_indices(self, indices, num_infl, mode, use_abs=False):
        # note: mode = {top, bottom, both}
        assert mode in {'top', 'bottom', 'both'}
        if mode == 'both':
            assert not use_abs, "use {top, bottom} if use_abs=True"
            num_top, num_bot = int(np.ceil(num_infl/2)), int(np.floor(num_infl/2))
            top_ind, top_vals = self.get_indices(indices, num_top, 'top', use_abs=False)
            bot_ind, bot_vals = self.get_indices(indices, num_bot, 'bottom', use_abs=False)
            indices = torch.cat((top_ind, bot_ind), axis=1)
            vals = torch.cat((top_vals, bot_vals), axis=1)
            return indices, vals

        largest = mode=='top'

        if not use_abs:
            infl = self.dm[indices].topk(num_infl, largest=largest)
            return infl.indices, infl.values

        # get abs-indices and signed-values
        infl = np.abs(self.dm[indices]).topk(num_infl, largest=largest)
        signed_vals = []

        for index, inf_indices in zip(indices, infl.indices):
            vals = self.dm[index][inf_indices].numpy()
            signed_vals.append(vals)

        signed_vals = torch.Tensor(np.array(signed_vals))
        return infl.indices, signed_vals

    def get_labels(self, infl_indices, split):
        """
        input: infl_indices: list of dataset[split] indices, dataset split
        output: infl_indices x num_infl x label_indices shaped array of labels
        """
        assert split in self.label_tuples

        try: infl_indices = infl_indices.numpy()
        except: pass

        if not isinstance(infl_indices[0], typing.Iterable):
            infl_indices = [infl_indices]

        label_tups = self.label_tuples[split]
        labels = np.zeros((len(infl_indices), len(infl_indices[0]), len(self.label_indices)))

        for idx1, indices in enumerate(infl_indices):
            for idx2, index in enumerate(indices):
                tup = label_tups[index]
                for idx3, label_index in enumerate(self.label_indices):
                    labels[idx1, idx2, idx3] = int(tup[idx3])

        return labels.astype(int)

    def _get_images(self, ind, split):
        imgs = []
        try: ind = ind.numpy()
        except: pass
        for i in ind:
            try: i = i.item()
            except: pass
            img = self.dsets[split][i][0]
            imgs.append(img)
        return torch.stack(imgs)

    def get_images(self, infl_indices, split):
        assert split in self.dsets

        if not isinstance(infl_indices[0], typing.Iterable):
            infl_indices = [infl_indices]

        get_imgs = lambda ind: torch.stack([self.dsets[split][i][0] for i in ind])
        return torch.stack([self._get_images(ind, split) for ind in infl_indices])

    def plot_image_row(self, image_tensor, titles, labels, axs=None,
                       img_height=3, img_width=3, title_fs=16, label_fs=16):
        return data_utils.plot_image_row(image_tensor, titles=titles, labels=labels, axs=axs, img_height=img_height, img_width=img_width, title_fs=title_fs, label_fs=label_fs)

    def _map_labels(self, label_tuple, class_names):
        if not class_names:
            return 'cls {}'.format('-'.join(map(str, label_tuple)))

        size = len(self.label_indices)
        key = label_tuple[0] if size==1 else label_tuple
        try:
            return class_names[key]
        except:
            if size > 1: key = tuple([x.item() for x in key])
            else: key = key.item()
            return class_names[key]


    def plot_influencers(self, indices, num_infl, mode,
                         use_abs=False, img_height=2.5, img_width=2.5,
                         title_fs=16, label_fs=16, axs=None, class_names=None):
        # load images+labels
        infl_indices, infl_values = self.get_indices(indices, num_infl, mode=mode, use_abs=use_abs)
        infl_images = self.get_images(infl_indices, 'train')
        infl_labels = self.get_labels(infl_indices, 'train')

        query_images = self.get_images(indices, self.dm_split)
        query_labels = self.get_labels(indices, self.dm_split)[0]

        # combine images
        query_images = query_images.permute(1,0,2,3,4)
        images = torch.cat((query_images, infl_images), dim=1)

        # combine labels
        labels = [[self._map_labels(l, class_names) for l in [q]+list(i)] for q, i in zip(query_labels, infl_labels)]

        # plot
        nrows, ncols = len(indices), 1+num_infl
        if axs is None:
            figsize = (img_width*ncols, img_height*nrows)
            fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
            if nrows==1: axs=[axs]
        else:
            if len(axs.shape) == 1: axs = axs[None, :]
            assert axs.shape[0] == nrows
            assert axs.shape[1] == ncols
            fig = None

        _thresh = int(np.ceil(num_infl/2))
        _infl_sign = lambda idx: '+' if mode=='top' or (mode=='both' and idx-1 < num_infl/2) else '-'
        _infl_index = lambda idx: (idx if idx <= _thresh else idx-_thresh) if mode=='both' else idx

        for it_id, (ax_row, img_row, label_row, infl_row, infl_ind_row) in enumerate(zip(axs, images, labels, infl_values, infl_indices)):
            titles = ['#{} | {}'.format(indices[it_id], label_row[0])] + [f'#{ind} | {label}' for ind, label in zip(infl_ind_row, label_row[1:])]
            labels = ['Query'] + [f'({_infl_sign(idx)}{_infl_index(idx)}): {score:.3f}' for idx, score in enumerate(infl_row,1)]
            self.plot_image_row(img_row, titles, labels, axs=ax_row, title_fs=title_fs, label_fs=label_fs)

            plot_utils.add_axis_border(ax_row[0], 'black', 3)
            diff_class = (np.array(label_row)!=label_row[0]).nonzero()[0]
            for class_idx in diff_class:
                plot_utils.add_axis_border(ax_row[class_idx], 'red', 3)

        if fig: fig.tight_layout()
        return fig, axs

# direction utils
class DirectionBase(object):

    def __init__(self):
        pass

    @staticmethod
    def load_datamodel(path, row_indices=None, col_indices=None,
                        normalize_embeddings=True, center_data=True):
        # returns [num_test x num_train] weight matrix
        X = torch.load(path, map_location='cpu')['weight'].numpy().T
        if row_indices is not None: X = X[row_indices]
        if col_indices is not None: X = X[:, col_indices]
        if normalize_embeddings: X /= np.linalg.norm(X, axis=1).reshape(-1,1)
        if center_data: X = X - X.mean(axis=0)
        return X

    def get_representative_indices(self, num_images, num_skip=1):
        grouped_ind = [self.sorted_indices[:(num_images*num_skip):num_skip], self.sorted_indices[::-1][:num_images*num_skip:num_skip]]
        titles = ['Top-k', 'Bottom-k']
        return {k: v for k, v in zip(titles, grouped_ind)}

    def visualize_grid(self, num_images, images_per_row=10, width_multiplier=1, axs=None,
                       height_multiplier=1, num_skip=1, pad_value=1., padding=0, titles=['Top-k', 'Bottom-k']):

        if axs is None:
            fig, axs = plt.subplots(1, 2, figsize=(width_multiplier*10, height_multiplier*(num_images//images_per_row)))
        else:
            fig = None

        grouped_ind = [self.sorted_indices[:(num_images*num_skip):num_skip], self.sorted_indices[::-1][:num_images*num_skip:num_skip]]

        for ax, indices, title in zip(axs, grouped_ind, titles):
            images = torch.stack([self.dataset[idx][0] for idx in indices])
            _, ax = plot_utils.plot_image_grid(images, len(images), images_per_row, ax=ax, pad_value=pad_value, padding=padding)
            ax.set_title(title, fontweight='bold', fontsize=13)

        if fig:
            fig.tight_layout()
        return fig, axs

class Direction(DirectionBase):

    def __init__(self, direction, datamodel, dataset, use_abs=False):
        """
        - direction: num_train dimensional array
        - datamodel: [num_test x num_train] matrix
        - dataset: dataset[x] -> (tensor, label, ...) corresp. to datamodel
        - use_abs: use absolute cosine similarity for sorting
        """
        super().__init__()

        assert len(direction)==datamodel.shape[1]

        self.direction = np.array(direction)
        self.direction /= np.linalg.norm(self.direction)

        self.datamodel = datamodel
        self.dataset = dataset

        self.use_abs = use_abs
        self.proj_scores = _proj = self._get_projection_score()
        self.sorted_indices = np.argsort(_proj)[::-1]

    def _get_projection_score(self):
        # if datamodel normalized: cosine similarity (default)
        # otherwise: x cos theta
        proj_scores = self.datamodel.dot(self.direction.T)
        if self.use_abs: proj_scores = np.abs(proj_scores)
        return proj_scores