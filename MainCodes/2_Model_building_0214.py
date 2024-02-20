import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchsummary import summary
import wandb

# Constants
CUDA_DEVICE = 3
ROOT_PATH = '/home/dg321/gitTest/PRI/irp/FlowPassBuilding/'
DATA_FILE = "Flow_Data/InterpolatedResult256/concatenated_data-1_1.npy"
NUM_EPOCHS = 1500
latent_channel_number = 2
ACTIVATION = nn.LeakyReLU(negative_slope=0.2)

# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = f"{CUDA_DEVICE}"

def load_data():
    datasetFolder = 'Flow_Data'
    xysize = 256

    samples = []
    for i in range(ntimesteps=499):
        s = np.load('/home/dg321/gitTest/PRI/irp/FlowPassBuilding/' + datasetFolder + '/InterpolatedResult256/InterpolatedResult256Raw/FpB_Interpolated_t{}_Velocity_{}_{}.npy'.format(
            i, xysize, xysize))
        samples.append(s)
    data = np.stack(samples, axis=0)
    return data

def create_dataloader(data, batch_size):
    return DataLoader(data, batch_size=batch_size, shuffle=True)

class Autoencoder(nn.Module):
    def __init__(self, first_hidden_layer, second_hidden_layer):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, first_hidden_layer, kernel_size=2, stride=2, padding=0),
            ACTIVATION,
            nn.Conv2d(first_hidden_layer, second_hidden_layer, kernel_size=2, stride=2, padding=0),
            ACTIVATION,
            nn.Conv2d(second_hidden_layer, first_hidden_layer, kernel_size=2, stride=1, padding=1),
            ACTIVATION,
            nn.Conv2d(first_hidden_layer, latent_channel_number, kernel_size=2, stride=1, padding=0),
            ACTIVATION
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channel_number, first_hidden_layer, kernel_size=2, stride=1, padding=0),
            ACTIVATION,
            nn.ConvTranspose2d(first_hidden_layer, second_hidden_layer, kernel_size=2, stride=1, padding=1),
            ACTIVATION,
            nn.ConvTranspose2d(second_hidden_layer, first_hidden_layer, kernel_size=2, stride=2, padding=0),
            ACTIVATION,
            nn.ConvTranspose2d(first_hidden_layer, 2, kernel_size=2, stride=2, padding=0),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def train_model(dataloader, device, config):
    autoencoder = Autoencoder(config.first_hidden_layer, config.second_hidden_layer).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=config.learning_rate)
    wandb.watch(autoencoder)
    for epoch in range(NUM_EPOCHS):
        for data in dataloader:
            inputs = data.to(device)
            optimizer.zero_grad()
            outputs = autoencoder(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item():.4f}')
        wandb.log({"loss": loss.item()})
    return autoencoder

def train():
    # Initialize a new run
    run = wandb.init()

    # Get the hyperparameters
    config = run.config

    # Load the data and create the dataloader
    data = load_data()
    dataloader = create_dataloader(data, config.batch_size)

    # Create the model, loss function, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_model = train_model(dataloader, device, config)

    torch.save(trained_model.state_dict(), "autoencoder.pth")
    wandb.save("autoencoder.pth")
    wandb.config.model_architecture = str(trained_model)
    artifact = wandb.Artifact('model', type='model')
    artifact.add_file('autoencoder.pth')
    wandb.log_artifact(artifact)

# Define the sweep configuration
sweep_config = {
    'method': 'random',  # or 'grid' or 'bayes'
    'metric': {
        'name': 'loss',
        'goal': 'minimize'
    },
    'parameters': {
        'learning_rate': {
            'min': 1e-5,
            'max': 1e-2
        },
        'first_hidden_layer': {
            'values': [16, 32, 64]
        },
        'second_hidden_layer': {
            'values': [32, 64, 128]
        },
        'batch_size': {
            'values': [16, 32, 64]
        }
    }
}

# Initialize the sweep
sweep_id = wandb.sweep(sweep_config, project="compression")

# Run the sweep
wandb.agent(sweep_id, train)


