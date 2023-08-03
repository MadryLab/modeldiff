from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from collections.abc import OrderedDict, Iterable
from .pca import residual_pca
from trak import TRAKer


class ModelDiff():
    def __init__(self,
                 train_loader: DataLoader,
                 model_arch: Module,
                 modelA_ckpts: Iterable[OrderedDict],
                 modelB_ckpts: Iterable[OrderedDict],
                 scores_save_dir: str='./modeldiff_scores') -> None:
        """
        A list of model parameters (model.state_dict())
        """
        self.train_loader = train_loader
        self.scores_A = None
        self.scores_B = None
        self.model_arch = model_arch
        self.modelA_ckpts = modelA_ckpts
        self.modelB_ckpts = modelB_ckpts
        self.scores_save_dir = scores_save_dir
        self.traker = TRAKer(model=self.model_arch,
                             task='image_classification',
                             train_set_size=self.train_loader.dataset.__len__()
                             )
        self.is_featurized = False

    def _load_scores(self, is_model_A: bool, scores: Tensor):
        if is_model_A:
            self.scores_A = scores
        else:
            self.scores_B = scores

    def _featurize(self, is_model_A: bool):
        """ Computes TRAK features for models A and B.

        is_model_A (bool): if set to True, calculates attributions for model A,
                           otherwise for model B.
        """
        if self.is_featurized:
            return 0

        ckpts = self.modelA_ckpts if is_model_A else self.modelB_ckpts
        for model_id, checkpoint in enumerate(ckpts):
            self.traker.load_checkpoint(checkpoint, model_id=model_id)
            for batch in self.train_loader:
                # batch should be a tuple of inputs and labels
                self.traker.featurize(batch=batch, num_samples=batch[0].shape[0])
        self.traker.finalize_features()
        self.is_featurized = True

    def _attribute_model(self, val_loader: DataLoader, is_model_A: bool, exp_name: str='exp'):
        """ Computes TRAK scores for models A and B for a given val loader.

        is_model_A (bool): if set to True, calculates attributions for model A,
                           otherwise for model B.
        """
        if not self.is_featurized:
            self._featurize(True)
            self._featurize(False)

        ckpts = self.modelA_ckpts if is_model_A else self.modelB_ckpts
        for model_id, checkpoint in enumerate(ckpts):
            self.traker.start_scoring_checkpoint(checkpoint,
                                                 model_id=model_id,
                                                 exp_name='test',
                                                 num_targets=val_loader.dataset.__len__())
            for batch in val_loader:
                self.traker.score(batch=batch, num_targets=batch[0].shape[0])

        if is_model_A:
            self.scores_A = self.traker.finalize_scores(exp_name=exp_name)
        else:
            self.scores_B = self.traker.finalize_scores(exp_name=exp_name)


    def _compare(self, val_loader, num_pca_comps: int, flip: bool):
        """
        flip (bool): if True, flips the scores of models A and B (to compute
                      B-A), otherwise computes A-B
        """
        self.scores_A = self._attribute_model(self.val_loader, True)
        self.scores_B = self._attribute_model(self.val_loader, False)

        self.val_loader = val_loader # @Kristina: Attribute 'val_loader' defined outside __init__ :p

        if flip: # @Kris: don't need val_loader for residual pca stuff
            diff = residual_pca(self.scores_B, self.scores_A, num_pca_comps)
        else:
            diff = residual_pca(self.scores_A, self.scores_B, num_pca_comps)

        return diff

    def get_A_minus_B(self, val_loader, num_pca_comps: int):
        """
        Returns residual PCA directions that have high explained variance
        for model A but low explained variance for model B.
        """
        return self._compare(val_loader, num_pca_comps, flip=False)

    def get_B_minus_A(self, val_loader, num_pca_comps: int):
        """
        Returns residual PCA directions that have high explained variance
        for model B but low explained variance for model A.
        """
        return self._compare(val_loader, num_pca_comps, flip=True)
