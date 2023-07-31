from modeldiff import ModelDiff
from .utils import get_dataloader, from_npy_to_state_dict
from pathlib import Path
from torchvision import models
import torch


def test_diff(tmp_path):
    model = models.resnet18()
    model.fc = torch.nn.Linear(512, 17)
    model.eval()
    model.cuda()

    tr_loader = get_dataloader(split='train')
    val_loader = get_dataloader(split='val')

    # run scripts/download_living17_checkpoints.sh first
    all_ckpts = torch.load(Path('./checkpoints/living17.pt'))
    ckpts_a = from_npy_to_state_dict(all_ckpts['with data aug'], model)
    ckpts_b = from_npy_to_state_dict(all_ckpts['without data aug'], model)

    print(ckpts_a[0]['conv1.weight'][0, 0, 0])
    print(ckpts_a[1]['conv1.weight'][0, 0, 0])

    md = ModelDiff(model, model, ckpts_a, ckpts_b, train_loader=tr_loader, scores_save_dir=tmp_path)
    md.get_A_minus_B(val_loader=val_loader, num_pca_comps=2)


def test_diff_from_scores(tmp_path):

    # run scripts/download_scores.sh first
    scores_dir = Path('./datamodels/')
    scoresA = torch.load(scores_dir.joinpath('living17_data-aug.pt'))['weight']
    scoresB = torch.load(scores_dir.joinpath('living17_without-data-aug.pt'))['weight']

    md = ModelDiff()
    diff = md.get_A_minus_B_from_scores(scoresA, scoresB, num_pca_comps=2)

    print(diff.keys())
    # @Harshay do we want to make some assertions about diff above?
