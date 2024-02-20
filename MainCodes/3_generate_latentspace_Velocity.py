import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os

import wandb

os.environ["CUDA_VISIBLE_DEVICES"] = f"{4}"



### 256x256
# datasetFolder = 'Flow_Data'
# xysize = 256
# ntimesteps = 501
# samples = []
# for i in range(ntimesteps):
#     s = np.load('/home/dg321/gitTest/PRI/irp/FlowPassBuilding/' + datasetFolder + '/InterpolatedResult256/InterpolatedResult256Raw/FpB_Interpolated_t{}_Velocity_{}_{}.npy'.format(
#         i, xysize, xysize))
#     samples.append(s)
#     print(s.shape)

## 384x384
datasetFolder = 'Flow_Data_9_9'
xysize = 384
ntimesteps = 399
samples = []
for i in range(ntimesteps):
    s = np.load('/home/dg321/gitTest/PRI/irp/FlowPassBuilding/' + datasetFolder + '/FpB_Interpolated_Velocity_384_384/FpB_Interpolated_t{}_Velocity_{}_{}.npy'.format(i, xysize, xysize))
    samples.append(s)

fist_hidden_layer = 16
second_hidden_layer = 32
latent_channel_number = 4

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(2, fist_hidden_layer, kernel_size=2, stride=2, padding=0),  
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(fist_hidden_layer, second_hidden_layer, kernel_size=2, stride=2, padding=0),  
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(second_hidden_layer, fist_hidden_layer, kernel_size=2, stride=1, padding=1),  
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(fist_hidden_layer, latent_channel_number, kernel_size=2, stride=1, padding=0),  
            # nn.LeakyReLU(negative_slope=0.2)
            nn.Tanh()  # constrain the latent space from range -1 to 1, Change this to nn.Sigmoid() for range 0 to 1
        )

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channel_number, fist_hidden_layer, kernel_size=2, stride=1, padding=0),  
            nn.LeakyReLU(negative_slope=0.2),
            nn.ConvTranspose2d(fist_hidden_layer, second_hidden_layer, kernel_size=2, stride=1, padding=1),  
            nn.LeakyReLU(negative_slope=0.2),
            nn.ConvTranspose2d(second_hidden_layer, fist_hidden_layer, kernel_size=2, stride=2, padding=0),  
            nn.LeakyReLU(negative_slope=0.2),
            nn.ConvTranspose2d(fist_hidden_layer, 2, kernel_size=2, stride=2, padding=0),   
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
run = wandb.init()
artifact = run.use_artifact('guodh/compression - 2 latent channels/model:v3', type='model')
artifact_dir = artifact.download()

# Load the model
autoencoder = Autoencoder()

# Load the model state dictionary from the downloaded artifact
autoencoder.load_state_dict(torch.load(os.path.join(artifact_dir, "autoencoder.pth")))

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

savepath = '/home/dg321/gitTest/PRI/irp/FlowPassBuilding/' + datasetFolder + '/Latent_data_Velocity_{}_{}_49696.npy'.format(xysize, xysize)

np.save(savepath, latent_samples_stacked)
print('Finished: ' + savepath)