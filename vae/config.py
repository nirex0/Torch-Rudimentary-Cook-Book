#CONFIG

weights_path = 'weights.pth'
device = 'mps' # use this device for Mac's Metal GPU
device = 'cuda:1'
image_dims = (32, 32)
image_channels = 3


flattened_dim = image_dims[0] * image_dims[1]


epochs = 10
batch_size = 64
latent_size = 48
learning_rate = 0.005


num_noises = 4
noise_mean = 0
noise_variance = 0.15
