import numpy as np

import os
os.environ["CUDA_VISIBLE_DEVICES"] = f"{2}"

import torch
import torch.nn as nn

import numpy as np
from torch.utils.data import Dataset, DataLoader

import torch.optim as optim

from torchvision import transforms

Latent_data_Velocity = np.load('/home/dg321/gitTest/PRI/irp/Flow_Data/Latent_data_Velocity_256_256_326464.npy')
Latent_data_Velocity = Latent_data_Velocity.reshape(501, 32, 64, 64)
print(Latent_data_Velocity.shape)

# training set
Latent_data_Velocity_training = Latent_data_Velocity[:450,:,:,:]

# test set
Latent_data_Velocity_test = Latent_data_Velocity[450:,:,:,:]

Latent_data_Building = np.load("/home/dg321/gitTest/PRI/irp/Flow_Data/Latent_data_Building_256_256_166464.npy")
Latent_data_Building = Latent_data_Building.reshape(1, 16, 64, 64)
print(Latent_data_Building.shape)


n_sampels = 80
t_gaps_sampels = 5
dt = 5
ntimes = 3

num_epochs = 1000
batch_size = 1  # Choose your desired batch size
lr=0.005


samples_training = []
samples_training_X = []
samples_training_Y = []

for i in range(1, n_sampels+1):
    ii = 1 + i*t_gaps_sampels
    # s = np.concatenate([Latent_data_VelocityXs_training[ss] for ss in range(ii,ii + dt*ntimes, dt )], axis = 0)
    s = np.stack([Latent_data_Velocity_training[ss] for ss in range(ii,ii + dt*ntimes, dt )])
    s_building = Latent_data_Building
    # ss_0 = np.concatenate((s[0].reshape(1, 32, 64, 64), s_building), axis=1)   # X
    ss_0 = np.concatenate((s[0].reshape(1, 32, 64, 64), s[1].reshape(1, 32, 64, 64), s_building), axis=1)   # X

    ss_1 = s[2].reshape(1, 32, 64, 64)   # Y
    # ss_1 = ss_0


    ss = (ss_0, ss_1)
    print(ss[0].shape)
    print(ss[1].shape)
    samples_training.append(ss)
    samples_training_X.append(ss_0)
    samples_training_Y.append(ss_1)


samples_training_X_stacked = (np.stack(samples_training_X)).reshape(n_sampels, ss_0.shape[1], 64, 64)
print(samples_training_X_stacked.shape)
samples_training_Y_stacked = (np.stack(samples_training_Y)).reshape(n_sampels, ss_1.shape[1], 64, 64)
print(samples_training_Y_stacked.shape)

data = samples_training
print(len(data))


import torch
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, data_x, data_y):
        self.data_x = data_x
        self.data_y = data_y

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        sample = {'input': self.data_x[idx], 'target': self.data_y[idx]}
        return sample

# Create an instance of your custom dataset
my_dataset = MyDataset(samples_training_X_stacked, samples_training_Y_stacked)

# Create a DataLoader for batching and shuffling the data
shuffle = False  # Set to True if you want to shuffle the data during training

data_loader = DataLoader(my_dataset, batch_size=batch_size, shuffle=shuffle)


# Encoder and Decoder (same as before)
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(ss_0.shape[1], 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            # Add more convolutional layers as needed
        )

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, ss_1.shape[1], kernel_size=5, stride=1, padding=2),
            nn.Identity(),
            # Add more convolutional layers as needed
        )

    def forward(self, x):
        return self.decoder(x)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 1, kernel_size=5, stride=1, padding=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.discriminator(x)

# Adversarial Autoencoder (AAE)
class AdversarialAutoencoder(nn.Module):
    def __init__(self, encoder, decoder, discriminator):
        super(AdversarialAutoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        discriminated = self.discriminator(encoded)
        return decoded, discriminated

# Instantiate components
encoder = Encoder()
decoder = Decoder()
discriminator = Discriminator()
aae_model = AdversarialAutoencoder(encoder, decoder, discriminator)

# Loss functions
criterion_autoencoder = nn.MSELoss()
criterion_adversarial = nn.BCELoss()

# Optimizers
optimizer_aae = optim.Adam(aae_model.parameters(), lr=lr, betas=(0.5, 0.999))

# Initialize lists to store losses and epochs for plotting
all_losses = []
epochs = []

# Training loop
for epoch in range(num_epochs):
    for batch in data_loader:
        inputs, targets = batch['input'], batch['target']

        # Train autoencoder
        optimizer_aae.zero_grad()
        decoded, discriminated = aae_model(inputs)
        loss_autoencoder = criterion_autoencoder(decoded, targets)
        loss_adversarial = criterion_adversarial(discriminated, torch.ones_like(discriminated))
        total_loss = loss_autoencoder + loss_adversarial
        total_loss.backward()
        optimizer_aae.step()

    # Append the total loss to the list for plotting
    all_losses.append(total_loss.item())
    epochs.append(epoch + 1)

    # Print the loss at the end of each epoch
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss.item():.4f}')

print("Training finished.")


model_saved_path = '/home/dg321/gitTest/PRI/irp/Flow_Data/PredictionModel/autoencoder_Flow_PredictionLatent_{}epochs_AAE80Samples.pth'.format(num_epochs)
torch.save(aae_model.state_dict(), model_saved_path)  ### PredictionMulti
print("Finished model: ", model_saved_path)