import numpy as np

import os
os.environ["CUDA_VISIBLE_DEVICES"] = f"{4}"

import torch
import torch.nn as nn

import numpy as np
from torch.utils.data import Dataset, DataLoader

# Set the device for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set the number of threads for CPU
torch.set_num_threads(2)  # Adjust the number of threads as needed

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

num_epochs = 2000
batch_size = 1  # Choose your desired batch size
lr=0.005

samples_training = []
samples_training_X = []
samples_training_Y = []

## combine Velocity and Building to form training samples
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


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(ss_0.shape[1], 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            # Add more convolutional layers as needed
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, ss_1.shape[1], kernel_size=5, stride=1, padding=2),
            nn.Identity(),
            # Add more convolutional layers as needed
        )

    def forward(self, x):
        x_encoder = self.encoder(x)
        x_decoder = self.decoder(x_encoder)
        return x_decoder

# Instantiate the improved model
model = MyModel().to(device)
model = model.float()


# Assuming you have a suitable loss function, e.g., Mean Squared Error
criterion = nn.MSELoss()

# Assuming you have an optimizer, e.g., Adam
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training loop
for epoch in range(num_epochs):
    for batch in data_loader:
        inputs, targets = batch['input'].to(device), batch['target'].to(device)
        print(inputs.shape)
        print(targets.shape)
        optimizer.zero_grad()
        outputs = model(inputs)  # Adjust input channels based on your architecture
        loss = criterion(outputs, targets)  # Adjust target channels based on your architecture
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


model_saved_path = '/home/dg321/gitTest/PRI/irp/Flow_Data/PredictionModel/autoencoder_Flow_PredictionLatent_{}epochs_AE80Samples.pth'.format(num_epochs)
torch.save(model.state_dict(), model_saved_path)  ### PredictionMulti
print("Finished model: ", model_saved_path)