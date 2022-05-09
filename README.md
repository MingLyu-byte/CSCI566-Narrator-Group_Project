# CSCI566-Narrator-Group-Project

## Prepare Dataset and Models
1. Download and unzip [LJSpeech dataset](https://keithito.com/LJ-Speech-Dataset/).
2. Put LJSpeech dataset under `data` folder.
3. Unzip `alignments.zip`.
4. Put [Nvidia pretrained waveglow model](https://drive.google.com/file/d/1WN1IIpEIW_fW4CMT8K8Gb9K4JCz0ks-S/view?usp=sharing) under `waveglow/pretrained_model` folder.
5. Download the trained [Alpha GAN model](https://drive.google.com/file/d/1e4fFZqXBUiArT-A2knKTJVbTHAgwISvs/view?usp=sharing) and put it under `model_alpha_gan` folder.
6. Download the trained [FastSpeech model](https://drive.google.com/file/d/1jNTwF1achfLva-d-8rXs8uGQ1-q9Yp9b/view?usp=sharing) and put it under `model_fastspeech` folder.
7. Run `python3 preprocess.py`.

## Training
Run `python3 train.py`.

## Evaluate Newly Trained Model
1. Run `python3 eval.py`. 

## Evaluate Trained Models in Report
1. Change the `checkpoint_path` in `hparams.py` to either `model_alpha_gan` or `model_fastspeech`.
2. Run `python3 eval.py`. 

## Plot Training Curves
Run `python3 plot.py`

### References
- [FastSpeech Paper](https://arxiv.org/abs/1905.09263)
- [Code Forked From](https://github.com/xcmyz/FastSpeech)