import matplotlib.pyplot as plt

STEP = 800
STEP_GAN = 500

with open("logger_raw/mel_loss.txt") as f:
    raw_mel_loss = f.read().splitlines()
    raw_mel_loss = list(map(float, raw_mel_loss))
    raw_mel_loss = raw_mel_loss[::STEP]
    raw_mel_x = [i * STEP + 1 for i in range(len(raw_mel_loss))]

with open("logger_raw/total_loss.txt") as f:
    raw_total_loss = f.read().splitlines()
    raw_total_loss = list(map(float, raw_total_loss))
    raw_total_loss = raw_total_loss[::STEP]
    raw_total_x = [i * STEP + 1 for i in range(len(raw_total_loss))]

with open("logger_alpha_gan/mel_loss.txt") as f:
    gan_mel_loss = f.read().splitlines()
    gan_mel_loss = list(map(float, gan_mel_loss))
    gan_mel_loss = gan_mel_loss[::STEP]
    gan_mel_x = [i * STEP + 1 for i in range(len(gan_mel_loss))]

with open("logger_alpha_gan/total_loss.txt") as f:
    gan_total_loss = f.read().splitlines()
    gan_total_loss = list(map(float, gan_total_loss))
    gan_total_loss = gan_total_loss[::STEP]
    gan_total_x = [i * STEP + 1 for i in range(len(gan_total_loss))]

plt.plot(raw_mel_x, raw_mel_loss, label="FastSpeech MSE for Mel Spectrogram")
plt.plot(gan_mel_x, gan_mel_loss, label="Alpha GAN MSE for Mel Spectrogram")
plt.plot(raw_total_x, raw_total_loss, label="FastSpeech total reconstruction loss")
plt.plot(gan_total_x, gan_total_loss, label="Alpha GAN total reconstruction loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.tight_layout()
plt.legend()
plt.ylim([0, 10])
plt.show()

with open("logger_alpha_gan/discriminator_loss.txt") as f:
    disc_loss = f.read().splitlines()
    disc_loss = list(map(float, disc_loss))
    disc_loss = disc_loss[::STEP_GAN]
    disc_x = [i * STEP_GAN + 1 for i in range(len(disc_loss))]

with open("logger_alpha_gan/generator_loss.txt") as f:
    gen_loss = f.read().splitlines()
    gen_loss = list(map(float, gen_loss))
    gen_loss = gen_loss[::STEP_GAN]
    gen_x = [i * STEP_GAN + 1 for i in range(len(gen_loss))]

plt.plot(disc_x, disc_loss, label="Discriminator Loss")
plt.plot(gen_x, gen_loss, label="Generator Loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.tight_layout()
plt.legend()
plt.ylim([0, 1.5])
plt.show()
