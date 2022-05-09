import torch
import torch.nn as nn
import torch.nn.functional as F

from multiprocessing import cpu_count
import numpy as np
import argparse
import os
import time
import math

from model import FastSpeech
from transformer.Models import Discriminator
from loss import DNNLoss
from dataset import BufferDataset, DataLoader
from dataset import get_data_to_buffer, collate_fn_tensor
from optimizer import ScheduledOptim
import hparams as hp
import utils


def main(args):
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define model
    print("Use FastSpeech")
    tts_model = nn.DataParallel(FastSpeech()).to(device)
    discriminator = nn.DataParallel(Discriminator()).to(device)
    print("Model Has Been Defined")
    num_param = utils.get_param_num(tts_model)
    num_disc_param = utils.get_param_num(discriminator)
    print('Number of TTS Parameters:', num_param)
    print('Number of Discriminator Parameters:', num_disc_param)
    # Get buffer
    print("Load data to buffer")
    buffer = get_data_to_buffer()

    # Optimizer and loss
    optimizer_tts = torch.optim.Adam(tts_model.parameters(),
                                     betas=(0.9, 0.98),
                                     eps=1e-9)
    optimizer_disc = torch.optim.Adam(discriminator.parameters(),
                                      betas=(0.9, 0.98),
                                      eps=1e-9)
    tts_scheduled_optim = ScheduledOptim(optimizer_tts,
                                         hp.decoder_dim,
                                         hp.n_warm_up_step,
                                         args.restore_step)
    disc_scheduled_optim = ScheduledOptim(optimizer_disc,
                                          hp.num_mels,
                                          hp.n_warm_up_step,
                                          args.restore_step)
    fastspeech_loss = DNNLoss().to(device)
    discriminator_loss = nn.BCEWithLogitsLoss().to(device)
    print("Defined Optimizer and Loss Function.")

    real_label = 1
    fake_label = 0

    # Load checkpoint if exists
    try:
        checkpoint = torch.load(os.path.join(
            hp.checkpoint_path, 'checkpoint_%d.pth.tar' % args.restore_step))
        tts_model.load_state_dict(checkpoint['model'])
        optimizer_tts.load_state_dict(checkpoint['optimizer'])
        print("\n---Model Restored at Step %d---\n" % args.restore_step)
    except:
        print("\n---Start New Training---\n")
        if not os.path.exists(hp.checkpoint_path):
            os.mkdir(hp.checkpoint_path)

    # Init logger
    if not os.path.exists(hp.logger_path):
        os.mkdir(hp.logger_path)

    # Get dataset
    dataset = BufferDataset(buffer)

    # Get Training Loader
    training_loader = DataLoader(dataset,
                                 batch_size=hp.batch_expand_size * hp.batch_size,
                                 shuffle=True,
                                 collate_fn=collate_fn_tensor,
                                 drop_last=True,
                                 num_workers=0)
    total_step = hp.epochs * len(training_loader) * hp.batch_expand_size

    # Define Some Information
    Time = np.array([])
    Start = time.perf_counter()

    # Training
    # tts_model = tts_model.train()
    tts_model.train()
    discriminator.train()

    for epoch in range(hp.epochs):
        for i, batchs in enumerate(training_loader):
            # real batch start here
            for j, db in enumerate(batchs):
                start_time = time.perf_counter()

                current_step = i * hp.batch_expand_size + j + args.restore_step + \
                               epoch * len(training_loader) * hp.batch_expand_size + 1

                # Get Data
                character = db["text"].long().to(device)
                mel_target = db["mel_target"].float().to(device)
                duration = db["duration"].int().to(device)
                mel_pos = db["mel_pos"].long().to(device)
                src_pos = db["src_pos"].long().to(device)
                max_mel_len = db["mel_max_len"]

                batch_size = mel_target.size(0)

                # Train the Discriminator with real data
                disc_scheduled_optim.zero_grad()
                d_real_output = discriminator(mel_target, mel_pos).view(-1)
                label = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
                d_real_loss = discriminator_loss(d_real_output, label)
                d_real_loss.backward()
                D_x = d_real_output.mean().item()

                # # Train the Discriminator with generated data
                mel_output, mel_postnet_output, duration_predictor_output = tts_model(character,
                                                                                      src_pos,
                                                                                      mel_pos=mel_pos,
                                                                                      mel_max_length=max_mel_len,
                                                                                      length_target=duration)
                d_fake_output = discriminator(mel_output.detach(), mel_pos).view(-1)
                label.fill_(fake_label)
                d_fake_loss = discriminator_loss(d_fake_output, label)
                d_fake_loss.backward()
                D_G_z1 = d_fake_output.mean().item()

                dis_loss = d_real_loss + d_fake_loss
                # Update Discriminator
                # Clipping gradients to avoid gradient explosion
                nn.utils.clip_grad_norm_(
                    discriminator.parameters(), hp.grad_clip_thresh)
                if args.frozen_learning_rate:
                    disc_scheduled_optim.step_and_update_lr_frozen(
                        args.learning_rate_frozen)
                else:
                    disc_scheduled_optim.step_and_update_lr()

                # Update Generator
                tts_scheduled_optim.zero_grad()
                # Cal Loss
                mel_loss, mel_postnet_loss, duration_loss = fastspeech_loss(mel_output,
                                                                            mel_postnet_output,
                                                                            duration_predictor_output,
                                                                            mel_target,
                                                                            duration)
                rec_loss = mel_loss + mel_postnet_loss + duration_loss

                label.fill_(real_label)  # fake labels are real for generator cost
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = discriminator(mel_output, mel_pos).view(-1)
                # Calculate G's loss based on this output
                d_gen_loss = discriminator_loss(output, label)
                # Calculate gradients for G
                gen_loss = rec_loss + d_gen_loss
                gen_loss.backward()

                D_G_z2 = output.mean().item()
                # Update G

                # Clipping gradients to avoid gradient explosion
                nn.utils.clip_grad_norm_(
                    tts_model.parameters(), hp.grad_clip_thresh)

                if args.frozen_learning_rate:
                    tts_scheduled_optim.step_and_update_lr_frozen(
                        args.learning_rate_frozen)
                else:
                    tts_scheduled_optim.step_and_update_lr()

                # Logger
                t_l = rec_loss.item()
                m_l = mel_loss.item()
                m_p_l = mel_postnet_loss.item()
                d_l = duration_loss.item()
                dis_l = dis_loss.item()
                gen_l = d_gen_loss.item()

                with open(os.path.join("logger", "total_loss.txt"), "a") as f_total_loss:
                    f_total_loss.write(str(t_l) + "\n")

                with open(os.path.join("logger", "mel_loss.txt"), "a") as f_mel_loss:
                    f_mel_loss.write(str(m_l) + "\n")

                with open(os.path.join("logger", "mel_postnet_loss.txt"), "a") as f_mel_postnet_loss:
                    f_mel_postnet_loss.write(str(m_p_l) + "\n")

                with open(os.path.join("logger", "duration_loss.txt"), "a") as f_d_loss:
                    f_d_loss.write(str(d_l) + "\n")
                with open(os.path.join("logger", "discriminator_loss.txt"), "a") as f_dis_loss:
                    f_dis_loss.write(str(dis_l) + "\n")
                with open(os.path.join("logger", "generator_loss.txt"), "a") as f_gen_loss:
                    f_gen_loss.write(str(gen_l) + "\n")

                # Print
                if current_step % hp.log_step == 0:
                    Now = time.perf_counter()

                    str1 = "Epoch [{}/{}], Step [{}/{}]:".format(
                        epoch + 1, hp.epochs, current_step, total_step)
                    str2 = "Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Duration Loss: {:.4f};".format(
                        m_l, m_p_l, d_l)
                    str3 = "Loss_D Loss: {:.4f}, Loss_G: {:.4f};".format(
                        dis_l, gen_l)
                    str4 = "Current Learning Rate is {:.6f}.".format(
                        tts_scheduled_optim.get_learning_rate())
                    str5 = "Time Used: {:.3f}s, Estimated Time Remaining: {:.3f}s.".format(
                        (Now - Start), (total_step - current_step) * np.mean(Time))

                    print("\n" + str1)
                    print(str2)
                    print(str3)
                    print(str4)
                    print(str5)

                    with open(os.path.join("logger", "logger.txt"), "a") as f_logger:
                        f_logger.write(str1 + "\n")
                        f_logger.write(str2 + "\n")
                        f_logger.write(str3 + "\n")
                        f_logger.write(str4 + "\n")
                        f_logger.write(str5 + "\n")
                        f_logger.write("\n")

                if current_step % hp.save_step == 0:
                    torch.save({'model': tts_model.state_dict(), 'optimizer': optimizer_tts.state_dict(
                    )}, os.path.join(hp.checkpoint_path, 'checkpoint_%d.pth.tar' % current_step))
                    print("save model at step %d ..." % current_step)

                end_time = time.perf_counter()
                Time = np.append(Time, end_time - start_time)
                if len(Time) == hp.clear_Time:
                    temp_value = np.mean(Time)
                    Time = np.delete(
                        Time, [i for i in range(len(Time))], axis=None)
                    Time = np.append(Time, temp_value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_step', type=int, default=0)
    parser.add_argument('--frozen_learning_rate', type=bool, default=False)
    parser.add_argument("--learning_rate_frozen", type=float, default=1e-3)
    args = parser.parse_args()
    main(args)
