import numpy as np
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = f"{4}"

import torch
import torch.nn as nn

import numpy as np
from torch.utils.data import Dataset, DataLoader

import torch.optim as optim

from torchvision import transforms


n_samples = 32
n_stackedSamples = 16
num_epochs = 50
lr = 0.003
batch_size = 8


samples = []
for i in range(n_samples):
    s = np.load('/home/dg321/gitTest/PRI/irp/Flow_Data/InterpolatedResult/FpB_Interpolated_t{}_Velocity_9600_9600.npy'.format(i))
    samples.append(s)

sampels_stacked = np.stack(samples)
print(sampels_stacked.shape)

samples_2steps = []

for i in range(8):
    samples_2step = sampels_stacked[i:i+2,:,:]
    samples_2steps.append(samples_2step)

print(len(samples_2steps))


data = samples_2steps

size_start_train = 0
size_end_train = data[0].shape[-1]

class ConvAutoencoder5Conv2ds(nn.Module):
    def __init__(self):
        super(ConvAutoencoder5Conv2ds, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=2, stride=2, padding=0),  # Layer 1
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=2, stride=2, padding=0),  # Layer 2
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=2, stride=2, padding=0),  # Layer 3
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=2, stride=2, padding=0),  # Layer 4
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=2, stride=2, padding=0),  # Layer 5
            nn.ReLU(True),
        )

       # Latent Space

        self.latentspace = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=5, stride=1, padding=0),  # Layer 1
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=0),  # Layer 2
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=0),  # Layer 3
            nn.ReLU(True)
        )


        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0),  # Layer 1
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0),  # Layer 2
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=0),  # Layer 3
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2, padding=0),  # Layer 4
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 2, kernel_size=2, stride=2, padding=0),  # Layer 5
            nn.Identity()
        )

    def forward(self, x):
        print(x.shape)
        # x = x.view(16, 2, 9600, -1)
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Instantiate the autoencoder
autoencoder = ConvAutoencoder5Conv2ds()

# You can print the model to see its updated architecture
print(autoencoder)


class CustomImageDataset(Dataset):
    def __init__(self, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample
    

from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor()
])


# Create the dataset and dataloader
dataset = CustomImageDataset(transform=transform)
# dataloader = DataLoader(dataset)  # Adjust batch size as needed
dataloader = DataLoader(dataset, batch_size=batch_size)

criterion = nn.MSELoss()  # Mean Squared Error loss for image reconstruction
optimizer = optim.Adam(autoencoder.parameters(), lr=lr)  # Adjust learning rate as needed

t0 = time.time()

print(t0)

for epoch in range(num_epochs):
    for d in dataloader:
        optimizer.zero_grad()
        outputs = autoencoder(d.view(-1, 2, 9600, 9600))
        print(d.shape)
        print(outputs.shape)
        loss = criterion(outputs.view(-1, 2, 9600, 9600), d.view(-1, 2, 9600, 9600))
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

torch.save(autoencoder.state_dict(), '/home/dg321/gitTest/PRI/irp/interpolation_code_example_2D/models/autoencoderBatches2T_Velocity_{}epochs_{}batchsize_lr{}.pth'.format(num_epochs, batch_size, lr))  ### Prediction
print("Training time:", time.time()-t0)
print("saved model name: /home/dg321/gitTest/PRI/irp/interpolation_code_example_2D/models/autoencoderBatches2T_Velocity_{}epochs_{}batchsize_lr{}.pth".format(num_epochs, batch_size, lr))




