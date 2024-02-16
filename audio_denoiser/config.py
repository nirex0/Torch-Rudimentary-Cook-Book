#CONFIG

weights_path = 'weights.pth'
device = 'mps' # use this device for Mac's Metal GPU
device = 'cuda:1'
channels = 1




epochs = 50
audio_dim = 200
batch_size = 3200
latent_size = 64
learning_rate = 0.03


num_noises = 4
noise_mean = 0
noise_variance = 0.15
