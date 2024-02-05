import numpy as np

import os
os.environ["CUDA_VISIBLE_DEVICES"] = f"{2}"

import torch
import torch.nn as nn

import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch.optim as optim

from torchvision import transforms

root_path = '/home/dg321/gitTest/PRI/irp/FlowPassBuilding/'

Latent_data_Velocity = np.load(root_path + 'Flow_Data/Latent_data_Velocity_256_256_326464.npy')
Latent_data_Velocity = Latent_data_Velocity.reshape(501, 32, 64, 64)
print(Latent_data_Velocity.shape)

# training set
Latent_data_Velocity_training = Latent_data_Velocity[:450,:,:,:]

# test set
Latent_data_Velocity_test = Latent_data_Velocity[450:,:,:,:]

Latent_data_Building = np.load(root_path + 'Flow_Data/Latent_data_Building_256_256_166464.npy')
Latent_data_Building = Latent_data_Building.reshape(1, 16, 64, 64)
print(Latent_data_Building.shape)


n_sampels = 200
t_gaps_sampels = 1
dt = 5
ntimes = 3

num_epochs = 10000
batch_size = 16  # Choose your desired batch size
model_saved_path = root_path + 'Flow_Data/PredictionModel/autoencoder_Flow_PredictionLatent_{}epochs_AAEVen200Samples_3ts.pth'.format(num_epochs)

lr_ae = 0.001
lr_d = 0.001
lr_ed = 0.001

samples_training = []
samples_training_X = []
samples_training_Y = []

for i in range(1, n_sampels+1):
    ii = 1 + i*t_gaps_sampels
    # s = np.concatenate([Latent_data_VelocityXs_training[ss] for ss in range(ii,ii + dt*ntimes, dt )], axis = 0)
    s = np.stack([Latent_data_Velocity_training[ss] for ss in range(ii,ii + dt*ntimes, dt )])
    s_building = Latent_data_Building
    # ss_0 = np.concatenate((s[0].reshape(1, 32, 64, 64), s_building), axis=1)   # X
    ss_0 = np.concatenate((s[0].reshape(1, 32, 64, 64), s[1].reshape(1, 32, 64, 64), s[2].reshape(1, 32, 64, 64), s_building), axis=1)   # X

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

# ## ############################## ## #######################
# ## #######################Old Model
# # Encoder and Decoder (same as before)
# class Encoder(nn.Module):
#     def __init__(self):
#         super(Encoder, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(ss_0.shape[1], 64, kernel_size=5, stride=1, padding=2),
#             nn.ReLU(),
#             nn.BatchNorm2d(64),
#             nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
#             nn.ReLU(),
#             nn.BatchNorm2d(128),
#             # Add more convolutional layers as needed
#         )

#     def forward(self, x):
#         return self.encoder(x)

# class Decoder(nn.Module):
#     def __init__(self):
#         super(Decoder, self).__init__()
#         self.decoder = nn.Sequential(
#             nn.Conv2d(128, 64, kernel_size=5, stride=1, padding=2),
#             nn.ReLU(),
#             nn.BatchNorm2d(64),
#             nn.Conv2d(64, ss_1.shape[1], kernel_size=5, stride=1, padding=2),
#             nn.Identity(),
#             # Add more convolutional layers as needed
#         )

#     def forward(self, x):
#         return self.decoder(x)

# # Discriminator
# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         self.discriminator = nn.Sequential(
#             nn.Conv2d(128, 64, kernel_size=5, stride=1, padding=2),
#             nn.ReLU(),
#             nn.BatchNorm2d(64),
#             nn.Conv2d(64, 1, kernel_size=5, stride=1, padding=2),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         return self.discriminator(x)

# # Adversarial Autoencoder (AAE)
# class AdversarialAutoencoder(nn.Module):
#     def __init__(self, encoder, decoder, discriminator):
#         super(AdversarialAutoencoder, self).__init__()
#         self.encoder = encoder
#         self.decoder = decoder
#         self.discriminator = discriminator

#     def forward(self, x):
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         discriminated = self.discriminator(encoded)
#         return decoded, discriminated


# ## ############################## ## #######################
# ## #######################AAE Model from Ventilation
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(ss_0.shape[1], 256, kernel_size=(3, 3), stride=1, padding=1),
            nn.LeakyReLU(),

            nn.Conv2d(256, 128, kernel_size=(3, 3), stride=1, padding=1),
            nn.LeakyReLU(),

            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=1),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.model(x)

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=1, padding=1, output_padding=0, bias=False),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(64, ss_1.shape[1], kernel_size=(3, 3), stride=1, padding=1, output_padding=0, bias=False),
            nn.Identity()
        )

    def forward(self, x):
        return self.model(x)

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

# Instantiate components and move to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = Encoder().to(device)
decoder = Decoder().to(device)
discriminator = Discriminator().to(device)
aae_model = AdversarialAutoencoder(encoder, decoder, discriminator).to(device)

# Loss functions
reconstruction_loss_fn = nn.MSELoss()
adversarial_loss_fn = nn.BCELoss()

# Optimizers
optimizer_ae = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr_ae)
optimizer_d = optim.Adam(discriminator.parameters(), lr=lr_d)
optimizer_ed = optim.Adam(encoder.parameters(), lr=lr_ed)

# Lists to store losses for plotting
all_losses_ae = []
all_losses_d = []
all_losses_ed = []

# Print the GPU device
print(f"Running on GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
# Training loop
for epoch in range(num_epochs):
    epoch_losses_ae = []
    epoch_losses_d = []
    epoch_losses_ed = []

    for batch in data_loader:
        inputs, targets = batch['input'].to(device), batch['target'].to(device)

        # Train autoencoder
        optimizer_ae.zero_grad()
        decoded, discriminated = aae_model(inputs)
        loss_ae = reconstruction_loss_fn(decoded, targets)
        loss_ae.backward()
        optimizer_ae.step()
        epoch_losses_ae.append(loss_ae.item())

        # Train discriminator
        optimizer_d.zero_grad()
        with torch.no_grad():
            encoded_fake = encoder(inputs).detach()
            encoded_true = torch.randn_like(encoded_fake).to(device)   # the true sample
        fake_output = discriminator(encoded_fake)
        true_output = discriminator(encoded_true)
        loss_d = adversarial_loss_fn(fake_output, torch.zeros_like(fake_output).to(device)) + \
                 adversarial_loss_fn(true_output, torch.ones_like(true_output).to(device))
        loss_d.backward()
        optimizer_d.step()
        epoch_losses_d.append(loss_d.item())

        # Train encoder/generator
        optimizer_ed.zero_grad()
        with torch.no_grad():
            encoded_fool = encoder(inputs).detach()
        fake_output = discriminator(encoded_fool)
        loss_ed = adversarial_loss_fn(fake_output, torch.ones_like(fake_output).to(device))
        loss_ed.backward()
        optimizer_ed.step()
        epoch_losses_ed.append(loss_ed.item())

    # Average losses over the epoch
    avg_loss_ae = sum(epoch_losses_ae) / len(epoch_losses_ae)
    avg_loss_d = sum(epoch_losses_d) / len(epoch_losses_d)
    avg_loss_ed = sum(epoch_losses_ed) / len(epoch_losses_ed)

    # Print the average loss at the end of each epoch
    print(f'Epoch [{epoch+1}/{num_epochs}], AE Loss: {avg_loss_ae:.4f}, D Loss: {avg_loss_d:.4f}, ED Loss: {avg_loss_ed:.4f}')

    # Record the losses for plotting
    all_losses_ae.append(avg_loss_ae)
    all_losses_d.append(avg_loss_d)
    all_losses_ed.append(avg_loss_ed)

# Plot the losses separately
fig = plt.figure(figsize=(10, 5))

# Autoencoder Loss Plot
plt.subplot(1, 3, 1)
plt.plot(all_losses_ae, label='Autoencoder Loss')
plt.title('Autoencoder Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Discriminator Loss Plot
plt.subplot(1, 3, 2)
plt.plot(all_losses_d, label='Discriminator Loss', color='orange')
plt.title('Discriminator Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# ED Loss Plot
plt.subplot(1, 3, 3)
plt.plot(all_losses_ed, label='Generator Loss')
plt.title('Generator Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
fig.savefig(model_saved_path[:-5]+'.png')
plt.close(fig)

print("Training finished.")
aae_model = aae_model.cpu()
print("Moved model back to cpu.")


torch.save(aae_model.state_dict(), model_saved_path)  ### PredictionMulti
print("Finished model: ", model_saved_path)
