import torch
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt 

data_channels = 1
seq_len = 100

latent_size = 25

conv_channels = 32
conv_kernel = 3 

num_samples_in_dataset = 1000
batch_size = 256
learning_rate = 0.005
num_epochs = 100
    
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv1d(data_channels, conv_channels, conv_kernel, padding=1)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(conv_channels)
        self.conv2 = nn.Conv1d(conv_channels, conv_channels, conv_kernel, padding=1)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(conv_channels)
        self.linear = nn.Linear(seq_len, latent_size)


    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.bn2(x)
        x = self.linear(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_size, seq_len)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(conv_channels)
        self.linear2 = nn.Linear(seq_len, seq_len)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(conv_channels)
        self.convT1 = nn.ConvTranspose1d(conv_channels, conv_channels, conv_kernel, padding=1)
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm1d(conv_channels)
        self.convT2 = nn.ConvTranspose1d(conv_channels, data_channels, conv_kernel, padding=1)
        # self.tanh1 = nn.Tanh()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.bn2(x)
        x = self.convT1(x)
        x = self.relu3(x)
        x = self.bn3(x)
        x = self.convT2(x)
        # x = self.tanh1(x)
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
    dataset = MyDataset(num_samples_in_dataset)

    # Create auto-encoder models
    encoder = Encoder()
    decoder = Decoder()

    # Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Define the loss function
    criterion = nn.MSELoss()

    # Define the optimizer
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)

    # Train the model
    train_model(dataloader, encoder, decoder, criterion, optimizer, num_epochs)

    # After training, let's input 1 data point and get the output
    single_data_point = torch.randn(1, data_channels, seq_len)
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        encoded = encoder(single_data_point)
        decoded = decoder(encoded)

    plt.figure(figsize=(6, 6))

    # Now let's plot the input and output side by side
    plt.title('Input and Output')
    plt.plot(single_data_point.view(-1).numpy(), label='Input', color='blue')
    plt.plot(decoded.view(-1).numpy(), label='Output', color='red')

    plt.legend(loc='upper right')
    plt.show()

    exit()
    # Now let's plot the input and output side by side
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title('Input')
    plt.plot(single_data_point.view(-1).numpy())
    
    plt.subplot(1, 2, 2)
    plt.title('Output')
    plt.plot(decoded.view(-1).numpy())

    plt.show()