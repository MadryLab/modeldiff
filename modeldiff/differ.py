from torch.nn import Module
from torch.utils.data import DataLoader
from collections.abc import Iterable
from typing import Optional
from pathlib import Path
from tqdm import tqdm
from .pca import residual_pca
from trak import TRAKer
import torch


class ModelDiff():
    def __init__(self,
                 modelA: Optional[Module] = None,
                 modelB: Optional[Module] = None,
                 modelA_ckpts: Iterable = [],
                 modelB_ckpts: Iterable = [],
                 train_loader: Optional[DataLoader] = None,
                 scores_save_dir: str='./modeldiff_scores') -> None:
        """
        A list of model parameters (model.state_dict())
        """
        self.train_loader = train_loader
        self.modelA = modelA
        self.modelB = modelB
        self.modelA_ckpts = modelA_ckpts
        self.modelB_ckpts = modelB_ckpts
        self.scores_save_dir = Path(scores_save_dir)

        if modelA is not None and modelB is not None:
            self.trakerA = TRAKer(model=self.modelA,
                                task='image_classification',
                                train_set_size=self.train_loader.dataset.__len__(),
                                save_dir=self.scores_save_dir.joinpath('modelA')
                                )
            self.trakerB = TRAKer(model=self.modelB,
                                task='image_classification',
                                train_set_size=self.train_loader.dataset.__len__(),
                                save_dir=self.scores_save_dir.joinpath('modelB')
                                )

        self.is_featurizedA = False
        self.is_featurizedB = False

    def _featurize(self, is_model_A: bool):
        """ Computes TRAK features for models A and B.

        is_model_A (bool): if set to True, calculates attributions for model A,
                           otherwise for model B.
        """
        if (is_model_A and self.is_featurizedA) or (not is_model_A and self.is_featurizedB):
            return 0

        ckpts = self.modelA_ckpts if is_model_A else self.modelB_ckpts
        traker = self.trakerA if is_model_A else self.trakerB
        for model_id, checkpoint in enumerate(ckpts):
            traker.load_checkpoint(checkpoint, model_id=model_id)
            for batch in self.train_loader:
                # batch should be a tuple of inputs and labels
                batch = [x.cuda() for x in batch]
                traker.featurize(batch=batch, num_samples=batch[0].shape[0])
        traker.finalize_features()

        if is_model_A:
            self.is_featurizedA = True
        else:
            self.is_featurizedB = True

    def _attribute_model(self, val_loader: DataLoader, is_model_A: bool, exp_name: str='exp'):
        """ Computes TRAK scores for models A and B for a given val loader.

        is_model_A (bool): if set to True, calculates attributions for model A,
                           otherwise for model B.
        """
        if (not self.is_featurizedA) or (not self.is_featurizedB):
            print('Featurizing model A')
            self._featurize(True)
            print('Featurizing model B')
            self._featurize(False)

        ckpts = self.modelA_ckpts if is_model_A else self.modelB_ckpts
        traker = self.trakerA if is_model_A else self.trakerB
        for model_id, checkpoint in enumerate(ckpts):
            traker.start_scoring_checkpoint(exp_name=exp_name,
                                            checkpoint=checkpoint,
                                            model_id=model_id,
                                            num_targets=val_loader.dataset.__len__())
            for batch in val_loader:
                batch = [x.cuda() for x in batch]
                traker.score(batch=batch, num_samples=batch[0].shape[0])

        scores = traker.finalize_scores(exp_name=exp_name)
        return torch.tensor(scores).to(torch.float32)

    def _compare(self, val_loader, num_pca_comps: int, flip: bool):
        """
        flip (bool): if True, flips the scores of models A and B (to compute
                      B-A), otherwise computes A-B
        """
        scoresA = self._attribute_model(val_loader, True).T # transpose to get [num_test x num_train]
        scoresB = self._attribute_model(val_loader, False).T

        if flip:
            diff = residual_pca(scoresB, scoresA, num_pca_comps)
        else:
            diff = residual_pca(scoresA, scoresB, num_pca_comps)

        return diff

    def _compare_from_scores(self, scoresA, scoresB, num_pca_comps: int, flip: bool):
        """
        scoresA: [num_test x num_train] matrix of attributions scores corresp. to algorithm A
        scoresB: [num_test x num_train] matrix of attributions scores corresp. to algorithm B
        num_pca_comps: number of distinguishing directions / residual pca directions 
        flip (bool): if True, flips the scores of models A and B (to compute B-A), otherwise computes A-B
        """
        if flip:
            diff = residual_pca(scoresB, scoresA, num_pca_comps)
        else:
            diff = residual_pca(scoresA, scoresB, num_pca_comps)

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

    def get_A_minus_B_from_scores(self, scoresA, scoresB, num_pca_comps: int):
        """
        Returns residual PCA directions that have high explained variance
        for model A but low explained variance for model B.
        """
        return self._compare_from_scores(scoresA, scoresB, num_pca_comps, flip=False)

    def get_B_minus_A_from_scores(self, scoresA, scoresB, num_pca_comps: int):
        """
        Returns residual PCA directions that have high explained variance
        for model B but low explained variance for model A.
        """
        return self._compare_from_scores(scoresA, scoresB, num_pca_comps, flip=True)

