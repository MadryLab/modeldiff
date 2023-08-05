from modeldiff import ModelDiff
from .utils import get_dataloader, construct_model
import torch


def test_integration(tmp_path):
    model = construct_model()
    tr_loader = get_dataloader(split='train')
    val_loader = get_dataloader(split='val')

    ckpts_dir = '/mnt/xfs/home/krisgrg/projects/trak/examples/checkpoints/'
    ckpts_a = [torch.load(ckpts_dir + 'sd_10_epoch_23.pt')]
    ckpts_b = [torch.load(ckpts_dir + 'sd_20_epoch_23.pt')]

    md = ModelDiff(model, model, ckpts_a, ckpts_b, train_loader=tr_loader, scores_save_dir=tmp_path)
    md.get_A_minus_B(val_loader=val_loader, num_pca_comps=2)

