import torch
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader

data_channels = 1
seq_len = 100
latent_size = 25
conv_channels = 32
conv_kernel = 3 

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv1d(data_channels, conv_channels, conv_kernel, padding=1)
        self.bn1 = nn.BatchNorm1d(conv_channels)
        self.linear = nn.Linear(seq_len, latent_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.linear(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.linear = nn.Linear(latent_size, seq_len)
        self.bn1 = nn.BatchNorm1d(conv_channels)
        self.convT1 = nn.ConvTranspose1d(conv_channels, data_channels, conv_kernel, padding=1)

    def forward(self, x):
        x = self.linear(x)
        x = self.bn1(x)
        x = self.convT1(x)
        return x

# Define your custom dataset
class MyDataset(Dataset):
    def __init__(self, n):
        self.n = n
        self.data = torch.randn(n, data_channels, seq_len)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.data[idx]


# Function to train the model
def train_model(dataloader, encoder, decoder, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        for batch in dataloader:
            # Forward pass
            encoded = encoder(batch)
            decoded = decoder(encoded)

            # Compute loss
            loss = criterion(decoded, batch)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


if __name__ == '__main__':

    # Generate n training samples
    n = 1000
    dataset = MyDataset(n)

    # Create auto-encoder models
    encoder = Encoder()
    decoder = Decoder()

    # Create a DataLoader
    batch_size = 32
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Define the loss function
    criterion = nn.MSELoss()

    # Define the optimizer
    learning_rate = 0.001
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)

    # Train the model
    num_epochs = 100
    train_model(dataloader, encoder, decoder, criterion, optimizer, num_epochs)
