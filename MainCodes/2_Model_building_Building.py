import numpy as np
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = f"{1}"

import torch
import torch.nn as nn

import numpy as np
from torch.utils.data import Dataset, DataLoader

import torch.optim as optim

from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms


num_epochs = 2000  # Adjust the number of training epochs
lr=0.001
batch_size=32

t0 = time.time()

print(t0)

concatenated_data = np.load("/home/dg321/gitTest/PRI/irp/Flow_Data/InterpolatedResult256/concatenated_data-1_1.npy")
# concatenated_data = np.load("/home/dg321/gitTest/PRI/irp/Flow_Data/InterpolatedResult256/sampelXs_stacked.npy")
# # data = concatenated_data[200:255,:,:,0]
# data = concatenated_data[200:255,:,:]

concatenated_data = concatenated_data.reshape(450, 2, 256, 256)

concatenated_data_list = []

dt = 5

for i in range(50):
    cd = concatenated_data[1 + i*dt,:,:, :]
    print(1 + i*dt)
    concatenated_data_list.append(cd)
data = np.stack(concatenated_data_list)

print(data.shape)

# Assuming you have a PyTorch DataLoader for your dataset (adjust the DataLoader creation according to your dataset)
# DataLoader should provide batches of images with shape (batch_size, 1, 1024, 1024)

dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=2, stride=2, padding=0),  # Change kernel_size and stride to 2
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=2, stride=2, padding=0),  # Change kernel_size and stride to 2
            nn.ReLU()
        )

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 32, kernel_size=2, stride=2, padding=0),  # Change kernel_size and stride to 2
            nn.ReLU(),
            nn.ConvTranspose2d(32, 2, kernel_size=2, stride=2, padding=0),   # Change kernel_size and stride to 2
            nn.Identity()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Instantiate the model, loss function, and optimizer
autoencoder = Autoencoder()
criterion = nn.MSELoss()  # Mean Squared Error loss works well for image reconstruction
optimizer = optim.Adam(autoencoder.parameters(), lr=lr)

# Training loop
for epoch in range(num_epochs):
    for data in dataloader:
        inputs = data  # Assuming DataLoader provides batches of images
        optimizer.zero_grad()
        outputs = autoencoder(inputs)
        print(inputs.shape)
        print(outputs.shape)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Training time:", time.time()-t0)


model_save_path = '/home/dg321/gitTest/PRI/irp/interpolation_code_example_2D/models/Velocity256_Compression_{}epochs_{}batchsize_lr{}_NewWorkingNN16_1-1.pth'.format(num_epochs, batch_size, lr)
torch.save(autoencoder.state_dict(), model_save_path)  ### Prediction
print("Saved model path: " + model_save_path)
