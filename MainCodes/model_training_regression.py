import numpy as np
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = f"{3}"

import torch
import torch.nn as nn

import numpy as np
from torch.utils.data import Dataset, DataLoader

import torch.optim as optim

from torchvision import transforms


data = np.load('/home/dg321/gitTest/PRI/irp/tests_for_donghu1/buildings_960_960_Velocity.npy')

size_start_train = 0
size_end_train = data.shape[-1]

data = np.stack([data[size_start_train:size_end_train,size_start_train:size_end_train]] * 3, axis=0)

print(data.shape)


num_epochs = 1000  # Adjust the number of training epochs

class ConvAutoencoder5Conv2ds(nn.Module):
    def __init__(self):
        super(ConvAutoencoder5Conv2ds, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=2, stride=2, padding=0),  # Layer 1
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
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2, padding=0),  # Layer 5
            nn.Identity()
        )

    def forward(self, x):
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


transform = transforms.Compose([
    transforms.ToTensor()
])


# Create the dataset and dataloader
dataset = CustomImageDataset(transform=transform)
dataloader = DataLoader(dataset)  # Adjust batch size as needed

criterion = nn.MSELoss()  # Mean Squared Error loss for image reconstruction
optimizer = optim.Adam(autoencoder.parameters(), lr=0.002)  # Adjust learning rate as needed

t0 = time.time()

print(t0)


for epoch in range(num_epochs):
    for d in dataloader:
        optimizer.zero_grad()
        outputs = autoencoder(d)
        print(d.shape)
        print(outputs.shape)
        loss = criterion(outputs, d)
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Training time:", time.time()-t0)
torch.save(autoencoder.state_dict(), '/home/dg321/gitTest/PRI/irp/interpolation_code_example_2D/models/autoencoder_model_5convsRegre_0_{}_{}epochs.pth'.format(size_end_train, num_epochs))