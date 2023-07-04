import torch
from torchaudio import load

from pesq import pesq
from pystoi import stoi

from sgmse.util.other import si_sdr, pad_spec

def evaluate_model_pr(model, num_eval_files):
    # Settings
    sr = 16000
    snr = 0.5
    N = 30
    corrector_steps = 0

    valid_set = model.data_module.valid_set

    # Select test files uniformly accros validation files
    total_num_files = len(valid_set)
    indices = torch.linspace(0, total_num_files-1, num_eval_files, dtype=torch.int)

    _pesq = 0
    _si_sdr = 0
    _estoi = 0
    
    # iterate over files
    for i in indices:
        # Load wavs
        X, Y = valid_set[i]
        x, y = model.to_audio(X), model.to_audio(Y)

        norm_factor = y.abs().max().item()
        y = y / norm_factor
        x = x / norm_factor

        # Reverse sampling
        sampler = model.get_pc_sampler(
            'reverse_diffusion', 'none', Y.unsqueeze(0).cuda(), N=N,
            corrector_steps=corrector_steps, snr=snr)
        sample, _ = sampler()

        x_hat = model.to_audio(sample.squeeze(), y.shape[-1])      
        x_hat = x_hat.squeeze().cpu().numpy()
        
        x = x.squeeze().cpu().numpy()
        y = y.squeeze().cpu().numpy()

        _si_sdr += si_sdr(x, x_hat)
        _pesq += pesq(sr, x, x_hat, 'wb')
        _estoi += stoi(x, x_hat, sr, extended=True)

    return _pesq/num_eval_files, _si_sdr/num_eval_files, _estoi/num_eval_files

