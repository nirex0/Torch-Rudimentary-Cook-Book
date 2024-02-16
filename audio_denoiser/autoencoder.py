from config import *
import matplotlib.pyplot as plt
import torch
import torch.nn as nn 
import tensorflow as tf
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torch import optim
from tqdm import tqdm 
import os 

class RNGDataset(Dataset):

    def __init__(self,num_samples,size = 100,rng_seed = 42):
        self.num_samples = num_samples
        self.rng_seed = rng_seed
        self.size = size

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self,idx):
        x = self._rnd_audio()
        return x

    def _rnd_audio(self):
        x = np.random.random(self.size)
        #adjust interval
        y = 2*x - 1
        y = torch.tensor(y, dtype=torch.float32)
        return y


class Encoder(nn.Module):
    def __init__(self, latent_size):        
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv1d(channels, 256, kernel_size=3, stride=1, padding=1).to(device=device)
        self.bn1 = nn.BatchNorm1d(256) 
        self.conv2 = nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1).to(device=device)
        self.bn2 = nn.BatchNorm1d(128) 
        self.conv3 = nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1).to(device=device)
        self.bn3 = nn.BatchNorm1d(64) 
        self.flat = nn.Flatten().to(device=device)
        self.linear_1 = nn.Linear(64 * audio_dim, 784).to(device=device)
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
    def __init__(self, latent_size):
        super(Decoder, self).__init__()
        self.linear_1 = nn.Linear(latent_size, 392).to(device=device)
        self.linear_2 = nn.Linear(392, 784).to(device=device)
        self.linear_3 = nn.Linear(784, 64).to(device=device)
        self.unflatten = nn.Unflatten(1, (64, 1)).to(device=device)
        self.deconv1 = nn.ConvTranspose1d(64, 128, kernel_size=3, stride=1, padding=1).to(device=device)
        self.bn1 = nn.BatchNorm1d(128) 
        self.deconv2 = nn.ConvTranspose1d(128, 256, kernel_size=3, stride=1, padding=1).to(device=device)
        self.bn2 = nn.BatchNorm1d(256) 
        self.deconv3 = nn.ConvTranspose1d(256, channels, kernel_size=3, stride=1, padding=1).to(device=device)

    def forward(self, x):
        x = torch.relu(self.linear_1(x))
        x = torch.relu(self.linear_2(x))
        x = torch.relu(self.linear_3(x))
        x = self.unflatten(x)
        x = torch.relu(self.bn1(self.deconv1(x))).to(device=device)
        x = torch.relu(self.bn2(self.deconv2(x))).to(device=device)
        x = torch.tanh(self.deconv3(x)).to(device=device)
        # x = torch.relu(self.deconv3(x)).to(device=device)
        # x = torch.relu(self.deconv4(x)).to(device=device)
        # reconstructed_image = torch.sigmoid(self.deconv5(x)).to(device=device)  # Output in [0, 1] range
        return x


if __name__ == '__main__':
    
    #Load the mp3 with the right sample rate.
    np.random.seed(42)

    dataset = RNGDataset(10000, audio_dim, 42)

    train_loader = DataLoader(dataset, batch_size = batch_size, shuffle = False)

    encoder_model = Encoder(latent_size=latent_size).to(device=device)
    decoder_model = Decoder(latent_size=latent_size).to(device=device)

    # Initialize the network and the optimizer
    model = nn.Sequential(
        encoder_model,
        decoder_model,
    ).to(device=device)
    
    # model = nn.Sequential(
    #     nn.Conv1d(channels, 256, kernel_size=3, stride=1, padding=1),
    #     nn.Linear(200,64),
    #     nn.ReLU(),
    #     nn.Linear(64, audio_dim),
    #     nn.ConvTranspose1d(256, 1, kernel_size=3, stride=1, padding=1),
    # ).to(device)

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
            for audios in tqdm(train_loader):
                  
                # Training pass
                # print(audios.shape)
                audios = audios.unsqueeze(1)
                # print(audios.shape)
                # exit()
                # conv = nn.Conv1d(1,32,3)
                # encoded = encoder_model(audios.to(device))
                # output = decoder_model(encoded.to(device))
                # output = model(audios.to(device))
                # print(output)

                # exit()
                optimizer.zero_grad()
                
                output = model(audios.to(device))
                loss = loss_fn(output.to(device), audios.to(device)) # Mean Squared Error Loss
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                # print(loss.item())
            else:
                print(f"Training loss: {running_loss/len(train_loader)}")

        #torch.save(model.state_dict(), 'weights.pth')


    for audios in tqdm(train_loader):
        audios = audios.unsqueeze(1)
        with torch.no_grad():
            #encoded = encoder_model(audios.to(device))
            #decoded = decoder_model(encoded.to(device))
            decoded = model(audios.to(device))
            
            plt.plot(audios[1].cpu().T)
            plt.show()
            plt.plot(decoded[1].cpu().T)
            plt.show()
            #print(audios.to(device) - decoded.to(device))
        exit()


    # Testing the network
        

    audios = next(iter(train_loader))
    audios = audios.unsqueeze(1)
    aud = audios[0] # .view(1, 784)

    # x = np.expand_dims(aud, 0)
    # x = torch.tensor(x).to(device)
    # print(img.shape)
    # print(x.shape)
    print(aud.shape)
    exit()
    with torch.no_grad():
        encoded = encoder_model(aud.to(device))
        print(encoded)
        decoded = decoder_model(encoded.to(device)) 

    # Display the original image and the reconstruction
    # decoded_reshaped = np.reshape(decoded.cpu().T,(32,32,3))
    
    plt.imshow(aud.cpu().T)
    plt.show()
    # plt.imshow(decoded_reshaped)
    # plt.show()
