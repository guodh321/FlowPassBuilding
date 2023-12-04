import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os

os.environ["CUDA_VISIBLE_DEVICES"] = f"{4}"


ntimesteps = 1
datasetFolder = 'Flow_Data_9_9'
xysize = 384

samples = []
for i in range(ntimesteps):
    s = np.load('/home/dg321/gitTest/PRI/irp/FlowPassBuilding/Flow_Data_9_9/FpB_Interpolated_VelocityAbsorption_384_384.npy')
    s = s[:1]

    ss = s.copy()
    
    ss[s<1500] = 0
    ss[s>=1500] = 1
    
    samples.append(ss)
    print(s.shape)

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=2, stride=2, padding=0),  # Change kernel_size and stride to 2
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=2, stride=2, padding=0),  # Change kernel_size and stride to 2
            nn.ReLU()
        )

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 32, kernel_size=2, stride=2, padding=0),  # Change kernel_size and stride to 2
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2, padding=0),   # Change kernel_size and stride to 2
            nn.Identity()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Instantiate the autoencoder
autoencoder = Autoencoder()

# You can print the model to see its updated architecture
print(autoencoder)


# Load the saved autoencoder
autoencoder = Autoencoder()  # Assuming you have the ConvAutoencoder class defined
autoencoder.load_state_dict(torch.load('/home/dg321/gitTest/PRI/irp/FlowPassBuilding/MainCodes/models/autoencoder_Flow_Building_256_256_2000epochs.pth'))

autoencoder.eval()  # Set the autoencoder to evaluation mode


latent_samples = []
for i in range(ntimesteps):
    # Assuming you have already defined your autoencoder
    # autoencoder = Autoencoder()

    # Load your dataP as you did
    dataP = samples[i]
    size_start = 0
    size_end = xysize

    # Extract a region of interest and prepare it for input
    data_rotated = dataP[:2, size_start:size_end, size_start:size_end].copy()
    print(data_rotated.shape)
    input_data = torch.from_numpy(data_rotated).unsqueeze(0).float()
    print(input_data.shape)

    # Pass the input through the encoder to get the latent variable
    with torch.no_grad():
        latent_space_output = autoencoder.encoder(input_data)

    print('Latent space shape:', latent_space_output.shape)

    latent_space_output = latent_space_output.detach().numpy()
    latent_samples.append(latent_space_output)



latent_samples_stacked = np.stack(latent_samples)


print(latent_samples_stacked.shape)

savepath = '/home/dg321/gitTest/PRI/irp/FlowPassBuilding/' + datasetFolder + '/Latent_data_Building_{}_{}_326464.npy'.format(xysize, xysize)

np.save(savepath, latent_samples_stacked)
print('Finished: ' + savepath)