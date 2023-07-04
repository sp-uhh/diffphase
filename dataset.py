from os.path import join
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset
from glob import glob
from torchaudio import load
import numpy as np
import torch.nn.functional as F

from sgmse.data_module import SpecsDataModule


class PRSpecs(Dataset):
    def __init__(self, data_dir, subset, dummy, shuffle_spec, num_frames,
            format='default', spec_transform=None,
            stft_kwargs=None, phase_init="zero", **ignored_kwargs):

        # Read file paths according to file naming format.
        if format == "default":
            self.clean_files = sorted(glob(join(data_dir, subset) + '/clean/*.wav'))
        elif format == "no_noisy":
            self.clean_files = sorted(glob(join(data_dir, subset, '**','*.wav')))
        else:
            # Feel free to add your own directory format
            raise NotImplementedError(f"Directory format {format} unknown!")

        self.dummy = dummy
        self.num_frames = num_frames
        self.shuffle_spec = shuffle_spec
        self.spec_transform = spec_transform
        self.phase_init = phase_init

        assert all(k in stft_kwargs.keys() for k in ["n_fft", "hop_length", "center", "window"]), "misconfigured STFT kwargs"
        self.stft_kwargs = stft_kwargs
        self.hop_length = self.stft_kwargs["hop_length"]
        assert self.stft_kwargs.get("center", None) == True, "'center' must be True for current implementation"

    def __getitem__(self, i):
        x, _ = load(self.clean_files[i])


        # formula applies for center=True
        target_len = (self.num_frames - 1) * self.hop_length
        current_len = x.size(-1)
        pad = max(target_len - current_len, 0)
        if pad == 0:
            # extract random part of the audio file
            if self.shuffle_spec:
                start = int(np.random.uniform(0, current_len-target_len))
            else:
                start = int((current_len-target_len)/2)
            x = x[..., start:start+target_len]
        else:
            # pad audio if the length T is smaller than num_frames
            x = F.pad(x, (pad//2, pad//2+(pad%2)), mode='constant')

        normfac = x.abs().max()
        x = x / normfac

        # X is the Clean complex spectrogram
        X = torch.stft(x, **self.stft_kwargs)

        # Y is the phaseless (or random phase) spectrogram
        if self.phase_init == 'random':
            Y = X.abs() * torch.exp(1j * 2*np.pi * torch.rand_like(X.abs()))
        elif self.phase_init == 'zero':
            Y = X.abs() + 0j
        else:
            raise NotImplementedError(f"Phase initialization mode {self.phase_init} not implemented!")

        X, Y = self.spec_transform(X), self.spec_transform(Y)
        return X, Y

    def __len__(self):
        if self.dummy:
            # for debugging shrink the data set size
            return int(len(self.clean_files)/200)
        else:
            return len(self.clean_files)

class PRDataModule(SpecsDataModule):
    @staticmethod 
    def add_argparse_args(parser):
        parser = super(PRDataModule, PRDataModule).add_argparse_args(parser)
        parser.add_argument("--phase_init", type=str, default="zero", choices=["zero", "random"], help="Type of phase initalization")
        return parser
    
    def __init__(self, phase_init="zero", **kwargs):
        super().__init__(**kwargs)
        self.phase_init = phase_init
        
    def setup(self, stage=None):
            specs_kwargs = dict(
                stft_kwargs=self.stft_kwargs, num_frames=self.num_frames,
                spec_transform=self.spec_fwd, phase_init=self.phase_init, **self.kwargs
            )
            if stage == 'fit' or stage is None:
                self.train_set = PRSpecs(data_dir=self.base_dir, subset='train',
                    dummy=self.dummy, shuffle_spec=True, format=self.format, **specs_kwargs)
                self.valid_set = PRSpecs(data_dir=self.base_dir, subset='valid',
                    dummy=self.dummy, shuffle_spec=False, format=self.format, **specs_kwargs)
            if stage == 'test' or stage is None:
                self.test_set = PRSpecs(data_dir=self.base_dir, subset='test',
                    dummy=self.dummy, shuffle_spec=False, format=self.format, **specs_kwargs)
