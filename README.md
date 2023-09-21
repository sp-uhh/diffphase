# DiffPhase: Generative Diffusion-based STFT Phase Retrieval

This repository contains the official PyTorch implementation for the paper [1]:

- Tal Peer, Simon Welker, Timo Gerkmann. [*"DiffPhase: Generative Diffusion-based STFT Phase Retrieval"*](https://ieeexplore.ieee.org/abstract/document/10095396), 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Rhodes Island, Greece, Jun. 2023. [[arxiv]](https://arxiv.org/abs/2211.04332) [[bibtex]](#citations--references)

Audio examples are available [on our project page](https://www.inf.uni-hamburg.de/en/inst/ab/sp/publications/icassp2023-diffphase).

DiffPhase is an adaptation of the SGMSE+ diffusion-based speech enhancement method to phase retrieval. SGMSE+ is described in [2] and [3] and has its own [repository](https://github.com/sp-uhh/sgmse).

## Installation
- Clone this repository along with the [sgmse](https://github.com/sp-uhh/sgmse) repository which is included as a submodule:
```bash
git clone --recurse-submodules https://github.com/sp-uhh/diffphase.git
```
- Create a new virtual environment with Python 3.8 (we have not tested other Python versions, but they may work).
- Install the package dependencies via `pip install -r requirements.txt`.
- If using W&B logging (default):
    - Set up a [wandb.ai](https://wandb.ai/) account
    - Log in via `wandb login` before running our code.
- If not using W&B logging:
    - Pass the option `--no_wandb` to `train.py`.
    - Your logs will be stored as local TensorBoard logs. Run `tensorboard --logdir logs/` to see them.


## Pretrained checkpoints

We provide two pretrained checkpoints:
- [DiffPhase](https://drive.google.com/file/d/19sQLF20kmkdvCxVhiP2e8y_BrrFqwTyB/view?usp=sharing) using the default SGMSE configuration. This model has ~65M parameters
- [DiffPhase-small](https://drive.google.com/file/d/1zsp-bqhB9G_KWHeK8HaFAgNSzZ5epVbW/view?usp=sharing) with ~22M parameters

Usage:
- For resuming training, you can use the `--resume_from_checkpoint` option of `train.py`.
- For performing phase reconstructions with these checkpoints, use the `--ckpt` option of `reconstruct.py` (see section **Evaluation** below).


## Training

Training is done by executing `train.py`. A minimal running example with default settings (as in our paper [1]) can be run with

```bash
python train.py --base_dir <your_base_dir>
```

where `your_base_dir` should be a path to a folder containing subdirectories `train/` and `valid/`. Each subdirectory must itself have a directory `clean/`. We currently only support training with `.wav` files sampled at 16 kHz.

For the DiffPhase-small variant, use the following options:

```bash
python train.py --num_res_blocks 1 --attn_resolutions 0 --ch_mult 1 1 2 2 1 --base_dir <your_base_dir>
```

To see all available training options, run `python train.py --help`. Also see the [sgmse](https://github.com/sp-uhh/sgmse) repository for more information.


## Evaluation

We provide an example script that takes a `.wav` file as an input, removes the phase and writes a reconstructed signal to another `.wav` file. Reconstruction is performed using the same procedure described in our paper. To use it, run

```bash
python reconstruct.py --input <input_wav> --output <reconstructed_wav> --ckpt <path_to_model_checkpoint> --N <number_of_reverse_steps>
```


## Citations / References

We kindly ask you to cite our paper in your publication when using any of our research or code:
```bib
@inproceedings{peerDiffPhase2023,
  title = {{DiffPhase: Generative Diffusion-based STFT Phase Retrieval}},
  booktitle = {{2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}},
  author = {Peer, Tal and Welker, Simon and Gerkmann, Timo},
  date = {2023-06},
  doi = {10.1109/ICASSP49357.2023.10095396}
}
```

>[1] Tal Peer, Simon Welker, Timo Gerkmann. "DiffPhase: Generative Diffusion-based STFT Phase Retrieval", 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Rhodes Island, Greece, Jun. 2023.
>
>[2] Simon Welker, Julius Richter, Timo Gerkmann. "Speech Enhancement with Score-Based Generative Models in the Complex STFT Domain", ISCA Interspeech, Incheon, Korea, Sep. 2022.
>
>[3] Julius Richter, Simon Welker, Jean-Marie Lemercier, Bunlong Lay, Timo Gerkmann. "Speech Enhancement and Dereverberation with Diffusion-Based Generative Models", IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 31, pp. 2351-2364, 2023.
