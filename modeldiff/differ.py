from torch import Tensor
from torch.utils.data import DataLoader
from collections.abc import OrderedDict, Iterable
from .pca import pca_diff


class ModelDiff():
    def __init__(self,
                 train_loader: DataLoader,
                 modelA_ckpts: Iterable[OrderedDict],
                 modelB_ckpts: Iterable[OrderedDict]):
        """
        A list of model parameters (model.state_dict())
        """
        self.train_loader = train_loader
        self.scores_A = None
        self.scores_B = None

    def _load_scores(self, is_model_A: bool, scores: Tensor):
        if is_model_A:
            self.scores_A = scores
        else:
            self.scores_B = scores

    def _attribute_model(self, is_model_A: bool):
        """
        is_model_A (bool): if set to True, calculates attributions for model A,
                           otherwise for model B.
        """
        if is_model_A:
            scores = self.scores_A
        else:
            scores = self.scores_B

        if scores is not None:
            # already computed / loaded
            return scores

        # TODO: call TRAK
        pass

    def compare(self, val_loader):
        self.scores_A = self._attribute_model(True)
        self.scores_B = self._attribute_model(False)

        self.val_loader = val_loader
        diff = pca_diff(self.scores_A, self.scores_B, self.val_loader)
        return diff
        