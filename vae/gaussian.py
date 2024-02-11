#GAUSSIAN

from config import *

from autoencoder import Encoder, Decoder
import torch
import torch.nn as nn 
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch import optim


transform = transforms.Compose([transforms.ToTensor()])
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)



encoder_model = Encoder(latent_size=latent_size).to(device=device)
decoder_model = Decoder(latent_size=latent_size, output_channels=1).to(device=device)



model = nn.Sequential(
    encoder_model,
    decoder_model,
).to(device=device)

model.load_state_dict(torch.load(weights_path))

noises = []
for i in range(num_noises):
    noise = torch.tensor(np.random.normal(noise_mean,noise_variance,latent_size), dtype = torch.float32).to(device)
    noises.append(noise)


# Testing the network
images, labels = next(iter(trainloader))
img = images[0]# .view(1, 784)

x = np.expand_dims(img, 0)
x = torch.tensor(x).to(device)
# print(img.shape)
# print(x.shape)
encoded = encoder_model(x)

print("OG")
plt.imshow(img.view(28, 28), cmap='gray')
plt.show()

with torch.no_grad():
    for noise in noises:
        latent_vector = encoded + noise
        print(latent_vector)
        decoded = decoder_model(latent_vector.to(device)) 

        plt.imshow(decoded.cpu().view(28, 28), cmap='gray')
        plt.show()
