import datetime as dt
from pathlib import Path
import random
import torch
from torch import optim

class BCOLORS(object):

    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @classmethod
    def print(cls, s, style='bold'):
        style = style.upper()
        assert style in vars(cls), 'invalid style'
        print (f'{getattr(cls, style)}{s}{cls.ENDC}')

class AverageMeter(object):
    def __init__(self):
        self.num = 0
        self.tot = 0

    def update(self, val, sz):
        self.num += val*sz
        self.tot += sz

    def mean(self):
        if self.tot==0: return None
        return self.num/self.tot

def get_timestamp(fmt='%y-%b%d-%H%M-%S', add_randint=True):
    r = random.randint(0, 100_000)
    ts = dt.datetime.now().strftime(fmt)
    if add_randint: ts = f'{ts}-r{r}'
    return ts

def load_run(save_dir, load_checkpoints=True):
    save_dir = Path(save_dir)
    assert save_dir.is_dir()
    ckpts, data = {}, {}

    def load_if_exists(pth):
        pth = Path(pth)
        if pth.exists():
            return torch.load(pth, map_location='cpu')

    for key in ['stats', 'metadata', 'eval']:
        out = load_if_exists(save_dir / f'{key}.pkl')
        if out: data[key] = out

    if load_checkpoints:
        for key in ['best', 'final']:
            ckpts[key] = load_if_exists(save_dir / f'checkpoint_{key}.pt')

        data['checkpoints'] = ckpts

    return data