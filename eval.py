import torch
import torch.nn as nn
import argparse
import numpy as np
import random
import time
import shutil
import os

import hparams as hp
import audio
import utils
import dataset
import text
import model as M
import waveglow

from transformer.Models import Discriminator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_DNN(num):
    checkpoint_path = "checkpoint_" + str(num) + ".pth.tar"
    model = nn.DataParallel(M.FastSpeech()).to(device)
    model.load_state_dict(torch.load(os.path.join(hp.checkpoint_path,
                                                  checkpoint_path))['model'])
    model.eval()
    return model


def synthesis(model, phn, alpha=1.0):
    text = np.array(phn)
    text = np.stack([text])
    src_pos = np.array([i + 1 for i in range(text.shape[1])])
    src_pos = np.stack([src_pos])
    sequence = torch.from_numpy(text).cuda().long()
    src_pos = torch.from_numpy(src_pos).cuda().long()

    with torch.no_grad():
        _, mel = model.module.forward(sequence, src_pos, alpha=alpha)
    return mel[0].cpu().transpose(0, 1), mel.contiguous().transpose(1, 2)


# def get_data():
#     test1 = "I am very happy to see you again!"
#     test2 = "Durian model is a very good speech synthesis!"
#     test3 = "When I was twenty, I fell in love with a girl."
#     test4 = "I remove attention module in decoder and use average pooling to implement predicting r frames at once"
#     test5 = "You can not improve your past, but you can improve your future. Once time is wasted, life is wasted."
#     test6 = "Death comes to all, but great achievements raise a monument which shall endure until the sun grows old."
#     data_list = list()
#     data_list.append(text.text_to_sequence(test1, hp.text_cleaners))
#     data_list.append(text.text_to_sequence(test2, hp.text_cleaners))
#     data_list.append(text.text_to_sequence(test3, hp.text_cleaners))
#     data_list.append(text.text_to_sequence(test4, hp.text_cleaners))
#     data_list.append(text.text_to_sequence(test5, hp.text_cleaners))
#     data_list.append(text.text_to_sequence(test6, hp.text_cleaners))
#     return data_list


def get_data():
    with open('data/LJSpeech-1.1/metadata.csv') as f:
        # data_raw = f.read().splitlines()[:50]
        data_raw = f.read().splitlines()
        data_raw = np.array(data_raw)
    data_list = list()
    text_list = list()
    sample_idx = random.sample(range(len(data_raw)), k=50)
    sample_idx.sort()
    sample_idx = np.array(sample_idx)
    data_raw = data_raw[sample_idx]
    for i, data in enumerate(data_raw):
        meta = data.split('|')
        norm_text = meta[-1]
        text_list.append(str(i) + "\t" + norm_text + "\n")
        data_list.append(text.text_to_sequence(norm_text, hp.text_cleaners))
    with open('text_eval.txt', "w") as f:
        f.writelines(text_list)
    return data_list, sample_idx


if __name__ == "__main__":
    # Test
    WaveGlow = utils.get_WaveGlow()
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=int, default=318000)
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()

    discriminator = Discriminator().cuda()

    print("use griffin-lim and waveglow")

    model = get_DNN(args.step)

    data_list, sample_idx = get_data()

    for i, phn in enumerate(data_list):
        idx = sample_idx[i]
        mel, mel_cuda = synthesis(model, phn, args.alpha)
        if not os.path.exists("results"):
            os.mkdir("results")

        # audio.tools.inv_mel_spec(
        #     mel, "results/"+str(args.step)+"_"+str(idx)+".wav")
        waveglow.inference.inference(
            mel_cuda, WaveGlow,
            "results/" + str(args.step) + "_" + str(idx) + "_waveglow.wav")
        print("Done", i + 1)

    # s_t = time.perf_counter()
    # for i in range(100):
    #     for _, phn in enumerate(data_list):
    #         _, _, = synthesis(model, phn, args.alpha)
    #     print(i)
    # e_t = time.perf_counter()
    # print((e_t - s_t) / 100.)
