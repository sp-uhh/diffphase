import torch
import soundfile as sf
from torchaudio import load
import sys
from argparse import ArgumentParser


sys.path.append("sgmse")
from sgmse.util.other import  pad_spec
from model import PRScoreModel

def main():
    parser = ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to input WAV file")
    parser.add_argument("--output", type=str, required=True, help="Output filename for reconstructed audio")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--N", type=int, default=30, help="The number of steps for the reverse SDE solver")
    
    
    args = parser.parse_args()
    
    # Load score model
    model = PRScoreModel.load_from_checkpoint(args.ckpt, base_dir='', batch_size=1, num_workers=0, kwargs=dict(gpu=False))
    model.eval(no_ema=False)
    model.freeze()
    model.cuda()
    
    reconstruct(in_file=args.input, out_file=args.output, model=model, N=args.N)
    
    
def reconstruct(in_file, out_file, model, N):
    model_fs = 16000
    
    # Load wav
    y, fs = load(in_file)
    assert fs == model_fs
    T_orig = y.size(1)

    # Normalize
    norm_factor = y.abs().max()
    y = y / norm_factor

    # Prepare DNN input
    Y = torch.unsqueeze(model._forward_transform(model._stft(y.cuda())), 0)
    Y = pad_spec(Y)

    # Discard phase
    Y = Y.abs() + 0j

    # Reverse sampling
    sampler = model.get_pc_sampler('reverse_diffusion', "none", Y.cuda(), N=N,corrector_steps=0, snr=0)

    sample, _ = sampler()

    # Apply final magnitude projection (enforce known magnitudes on output)
    sample = model._pA(sample, Y)

    # Backward transform in time domain
    x_hat = model.to_audio(sample.squeeze(), T_orig)

    # Renormalize
    x_hat = x_hat * norm_factor

    # Write enhanced wav file
    sf.write(out_file,x_hat.cpu().numpy(), model_fs)


if __name__ == "__main__":
    main()


