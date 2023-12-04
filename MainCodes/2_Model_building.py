import numpy as np
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = f"{2}"

import torch
import torch.nn as nn

import numpy as np
from torch.utils.data import Dataset, DataLoader

import torch.optim as optim

from torchvision import transforms


num_epochs = 1000  # Adjust the number of training epochs
lr = 0.003

t0 = time.time()

print(t0)

concatenated_data = np.load("/home/dg321/gitTest/PRI/irp/Flow_Data/InterpolatedResult256/concatenated_data-1_1.npy")
data = concatenated_data

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
batch_size = 64
dataloader = DataLoader(dataset, batch_size=batch_size)


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
            nn.Conv2d(64, 32, kernel_size=2, stride=2, padding=0),  # Layer 4
            nn.ReLU(True),
            nn.Conv2d(32, 16, kernel_size=2, stride=2, padding=0),  # Layer 5
            nn.ReLU(True),
        )


        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 32, kernel_size=2, stride=2, padding=0),  # Layer 1
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 64, kernel_size=2, stride=2, padding=0),  # Layer 2
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=0),  # Layer 3
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2, padding=0),  # Layer 4
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 2, kernel_size=2, stride=2, padding=0),  # Layer 5
            nn.Identity()
        )

    def forward(self, x):
        # print(x.shape)
        # x = x.view(16, 2, 9600, -1)
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Instantiate the autoencoder
autoencoder = ConvAutoencoder5Conv2ds()

# You can print the model to see its updated architecture
print(autoencoder)

criterion = nn.MSELoss()  # Mean Squared Error loss for image reconstruction
optimizer = optim.Adam(autoencoder.parameters(), lr=lr)  # Adjust learning rate as needed

for epoch in range(num_epochs):
    for d in dataloader:
        optimizer.zero_grad()
        outputs = autoencoder(d)
        # print(d.shape)
        # print(outputs.shape)
        # loss = criterion(outputs, d)
        # loss.backward()
        loss1 = criterion(outputs[:,0,:,:], d[:,0,:,:])
        loss2 = criterion(outputs[:,1,:,:], d[:,1,:,:])
        loss = loss1 + loss2
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Training time:", time.time()-t0)

torch.save(autoencoder.state_dict(), '/home/dg321/gitTest/PRI/irp/interpolation_code_example_2D/models/Velocity256_Compression_{}epochs_{}batchsize_lr{}_updatedLoss.pth'.format(num_epochs, batch_size, lr))  ### Prediction
print("Saved model path: /home/dg321/gitTest/PRI/irp/interpolation_code_example_2D/models/Velocity256_Compression_{}epochs_{}batchsize_lr{}_updatedLoss.pth".format(num_epochs, batch_size, lr))
