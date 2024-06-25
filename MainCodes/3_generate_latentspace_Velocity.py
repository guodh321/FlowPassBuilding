import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os

import wandb

os.environ["CUDA_VISIBLE_DEVICES"] = f"{5}"



## 256x256
datasetFolder = 'Flow_Data'
xysize = 256
ntimesteps = 501
samples = []
for i in range(ntimesteps):
    s = np.load('/home/dg321/gitTest/PRI/irp/FlowPassBuilding/' + datasetFolder + '/InterpolatedResult256/InterpolatedResult256Raw/FpB_Interpolated_t{}_Velocity_{}_{}.npy'.format(
        i, xysize, xysize))
    samples.append(s)
    print(s.shape)

# ## 384x384
# datasetFolder = 'Flow_Data_9_9'
# xysize = 384
# ntimesteps = 399
# samples = []
# for i in range(ntimesteps):
#     s = np.load('/home/dg321/gitTest/PRI/irp/FlowPassBuilding/' + datasetFolder + '/FpB_Interpolated_Velocity_384_384/FpB_Interpolated_t{}_Velocity_{}_{}.npy'.format(i, xysize, xysize))
#     samples.append(s)

# Define the autoencoder model
hid1 = 20*2
hid2= 40*2
hid3 = 80*2
hid4 = 80*2
hid5 = 40*2
hid6 = 20*2
            
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
 
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(2, hid1, kernel_size=3, stride=1, padding=1),#1 - Smoothing the change in number of channels; no reduction in output size
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(hid1, hid2, kernel_size=2, stride=2),#2 - reduction by 2
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(hid2, hid3, kernel_size=3, stride=1, padding=1),#3 - Smoothing the change in number of channels; no reduction in output size
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(hid3, hid4, kernel_size=2, stride=2),#4 - reduction by 2
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(hid4, hid5, kernel_size=3, stride=1, padding=1),#5 - Smoothing the change in number of channels; no reduction in output size
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(hid5, hid6, kernel_size=2, stride=2),#6 - reduction by 2
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(hid6, 2, kernel_size=3, stride=1, padding=1),#7 - Smoothing the change in number of channels; no reduction in output size
            #We should finish with the same number of channels of the input (4)
            # nn.LeakyReLU(negative_slope=0.2)
            # nn.Tanh()
            nn.Identity()
        )
 
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2, hid6, kernel_size=3, stride=1, padding=1),# Smoothing the change in number of channels; no increase in output size
            nn.LeakyReLU(negative_slope=0.2),
            nn.ConvTranspose2d(hid6, hid5, kernel_size=2, stride=2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.ConvTranspose2d(hid5, hid4, kernel_size=3, stride=1, padding=1),# Smoothing the change in number of channels; no increase in output size
            nn.LeakyReLU(negative_slope=0.2),
            nn.ConvTranspose2d(hid4, hid3, kernel_size=2, stride=2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.ConvTranspose2d(hid3, hid2, kernel_size=3, stride=1, padding=1),# Smoothing the change in number of channels; no increase in output size
            nn.LeakyReLU(negative_slope=0.2),
            nn.ConvTranspose2d(hid2, hid1, kernel_size=2, stride=2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.ConvTranspose2d(hid1, 2, kernel_size=3, stride=1, padding=1),# Smoothing the change in number of channels; no increase in output size
            nn.Identity()
            #nn.Sigmoid()
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
# artifact = run.use_artifact('guodh/compression - Containing kernel size 3/model:v8', type='model')   # Tanh()
artifact = run.use_artifact('guodh/compression - 2 latent channels 1-4 xy size/model:v11', type='model')   # Identity()
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

# savepath = '/home/dg321/gitTest/PRI/irp/FlowPassBuilding/' + datasetFolder + '/Latent_data_Velocity_{}_{}_24848_identitylatent.npy'.format(xysize, xysize)
savepath = '/home/dg321/gitTest/PRI/irp/FlowPassBuilding/' + datasetFolder + '/Latent_data_Velocity_{}_{}_23232_identitylatent.npy'.format(xysize, xysize)

np.save(savepath, latent_samples_stacked)
print('Finished: ' + savepath)