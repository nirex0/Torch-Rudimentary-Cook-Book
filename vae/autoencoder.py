from config import *

import torch
import torch.nn as nn 
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch import optim
from tqdm import tqdm 
import os 


class Encoder(nn.Module):
    def __init__(self, latent_size):        
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=image_channels, out_channels=256, kernel_size=(3,3), stride=1, padding=1).to(device=device)
        self.bn1 = nn.BatchNorm2d(256) 
        self.conv2 = nn.Conv2d(256, 128, kernel_size=(3,3), stride=1, padding=1).to(device=device)
        self.bn2 = nn.BatchNorm2d(128) 
        self.conv3 = nn.Conv2d(128, latent_size, kernel_size=(3,3), stride=1, padding=1).to(device=device)
        self.bn3 = nn.BatchNorm2d(latent_size) 
        self.flat = nn.Flatten().to(device=device)
        self.linear_1 = nn.Linear(latent_size * flattened_dim, 784).to(device=device)
        self.linear_2 = nn.Linear(784, 392).to(device=device)
        self.linear_3 = nn.Linear(392, latent_size).to(device=device)

    def forward(self, x):
        x = torch.tanh(self.bn1(self.conv1(x))).to(device=device)
        x = torch.tanh(self.bn2(self.conv2(x))).to(device=device)
        x = torch.tanh(self.bn3(self.conv3(x))).to(device=device)
        x = self.flat(x).to(device=device)
        x = torch.tanh(self.linear_1(x).to(device=device))
        x = torch.tanh(self.linear_2(x).to(device=device))
        x = torch.tanh(self.linear_3(x).to(device=device))
        return x
    
class Decoder(nn.Module):
    def __init__(self, latent_size, output_channels):
        super(Decoder, self).__init__()
        self.linear_1 = nn.Linear(latent_size, 392).to(device=device)
        self.linear_2 = nn.Linear(392, 784).to(device=device)
        self.linear_3 = nn.Linear(784, latent_size * flattened_dim).to(device=device)
        self.unflatten = nn.Unflatten(1, (latent_size, image_dims[0], image_dims[1])).to(device=device)
        self.deconv1 = nn.ConvTranspose2d(latent_size, 128, kernel_size=(3,3), stride=1, padding=1).to(device=device)
        self.bn1 = nn.BatchNorm2d(128) 
        self.deconv2 = nn.ConvTranspose2d(128, 256, kernel_size=(3,3), stride=1, padding=1).to(device=device)
        self.bn2 = nn.BatchNorm2d(256) 
        self.deconv3 = nn.ConvTranspose2d(256, output_channels, kernel_size=(3,3), stride=1, padding=1).to(device=device)

    def forward(self, x):
        x = torch.relu(self.linear_1(x))
        x = torch.relu(self.linear_2(x))
        x = torch.relu(self.linear_3(x))
        x = self.unflatten(x)
        x = torch.relu(self.bn1(self.deconv1(x))).to(device=device)
        x = torch.relu(self.bn2(self.deconv2(x))).to(device=device)
        x = torch.sigmoid(self.deconv3(x)).to(device=device)
        # x = torch.relu(self.deconv3(x)).to(device=device)
        # x = torch.relu(self.deconv4(x)).to(device=device)
        # reconstructed_image = torch.sigmoid(self.deconv5(x)).to(device=device)  # Output in [0, 1] range
        return x


if __name__ == '__main__':
    
    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])
    # trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
    trainset = datasets.CIFAR10('~/.pytorch/CIFAR10_data/', download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

  

    encoder_model = Encoder(latent_size=latent_size).to(device=device)
    decoder_model = Decoder(latent_size=latent_size, output_channels=image_channels).to(device=device)

    # Initialize the network and the optimizer
    model = nn.Sequential(
        encoder_model,
        decoder_model,
    ).to(device=device)

    is_loaded = False
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path))
        is_loaded = True


    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss().to(device)
    # loss_fn = nn.L1Loss().to(device)

    if not is_loaded:
        # Training loop
        for e in tqdm(range(epochs)):
            print('Start Training')
            running_loss = 0
            for images, labels in tqdm(trainloader):
                
                # images = images.view(images.shape[0], -1)
                # print(images.shape)
                # exit()
                
                # Training pass
                optimizer.zero_grad()
                
                output = model(images.to(device))
                loss = loss_fn(output.to(device), images.to(device)) # Mean Squared Error Loss
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                # print(loss.item())
            else:
                print(f"Training loss: {running_loss/len(trainloader)}")

        torch.save(model.state_dict(), 'weights.pth')

    # Testing the network
    images, labels = next(iter(trainloader))
    img = images[0]# .view(1, 784)

    x = np.expand_dims(img, 0)
    x = torch.tensor(x).to(device)
    # print(img.shape)
    # print(x.shape)
    with torch.no_grad():
        encoded = encoder_model(x)
        print(encoded)
        decoded = decoder_model(encoded.to(device)) 

    # Display the original image and the reconstruction
    decoded_reshaped = np.reshape(decoded.cpu().T,(32,32,3))
    
    plt.imshow(img.T)
    plt.show()
    plt.imshow(decoded_reshaped)
    plt.show()
