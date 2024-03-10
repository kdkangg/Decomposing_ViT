
import os
import torch.nn as nn
import torch
import numpy as np
from learned_dict import TiedSAE
import h5py
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
import argparse
from transformers import ViTForImageClassification
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--ratio', type=int, help='Dim ratio to original', default =10)
parser.add_argument('--epochs', type=int, help='epochs', default = 100)
parser.add_argument('--alpha', type=int, help='L1 alpha', default = 1e-3)
parser.add_argument('--lr', type=float, help='Learning rate', default = 1e-4)
parser.add_argument('--vit', type=str, help='ViT model name')
parser.add_argument('--modelpath', type=str, help='Path to save model')
parser.add_argument('--actpath', type=str, help='Path to saved activations')

args = parser.parse_args()

# Create the directory if it doesn't exist
os.makedirs(f'SAE_models/{args.modelpath}/', exist_ok=True)

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ViTForImageClassification.from_pretrained(args.vit)
n_layers = model.vit.encoder.layer.__len__()

# dataset class
class MyDataset(Dataset):
    def __init__(self, file_path, layer_name):
        self.data = []
        with h5py.File(file_path, 'r') as f:
            for key in f.keys():
                activations = f[key][layer_name][:]
                images = f[key]['image'][:]
                self.data.extend(zip(images, activations))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# initialize tne encoder and encoder_bias for TiedSAE
activation_size = 768
n_dict_components = args.ratio * activation_size

encoder = torch.randn((n_dict_components, activation_size), device = device) # encoder
nn.init.xavier_uniform_(encoder)
encoder_bias = torch.zeros(n_dict_components, device = device) # encoder bias

# Create a TiedSAE instance
tied_sae = TiedSAE(encoder, encoder_bias)
tied_sae.to_device(device)

optimizer = torch.optim.Adam([tied_sae.encoder, tied_sae.encoder_bias], lr=args.lr)
original_bias = tied_sae.encoder_bias.clone().detach()

for layer in tqdm(range(n_layers)):
    dataset = []
    file_paths = glob.glob(f'{args.actpath}/*.h5')
    for file_path in file_paths:
        class_dataset = MyDataset(file_path, f'vit.encoder.layer.{layer}.output')
        dataset.extend(class_dataset)

    # split dataset into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # create dataloader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

    # create summary writer
    writer = SummaryWriter(f'runs/{args.modelpath}/SAE_ratio{args.ratio}_epochs{args.epochs}_lr{args.lr}_layer{layer}')
    dead_features = torch.zeros(tied_sae.encoder.shape[0])

    best_loss = np.inf
    patience = 10  # Number of epochs to wait for improvement
    counter = 0  # Counter for epochs with no improvement

    # train
    for epoch in tqdm(range(args.epochs)):
        for i, (images, activations) in enumerate(train_loader):
            tied_sae.encoder.requires_grad = True
            tied_sae.encoder_bias.requires_grad = True

            with torch.no_grad():
                activations = activations.to(device)
            
            c = tied_sae.encode(activations)
            x_hat = tied_sae.decode(c)

            reconstruction_loss = torch.mean((activations - x_hat) ** 2)
            l1_loss = torch.norm(c,1,dim=-1).mean()
            total_loss = reconstruction_loss + args.alpha * l1_loss

            dead_features += c.sum(dim=(0,1)).cpu() 

            with torch.no_grad():
                sparsity = (c != 0).float().mean(dim=(0,1)).sum().cpu().item()
                num_dead_features = (dead_features == 0).sum().cpu().item()
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        # log values to tensorboard
        writer.add_scalar('Loss/train', total_loss, epoch)
        writer.add_scalar('Reconstruction Loss/train', reconstruction_loss, epoch)
        writer.add_scalar('L1 Loss/train', l1_loss, epoch)
        writer.add_scalar('Sparsity/train', sparsity, epoch)
        writer.add_scalar('Dead Features/train', num_dead_features, epoch)

        dead_features = torch.zeros(tied_sae.encoder.shape[0])
        
        # validation
        tied_sae.encoder.requires_grad = False
        tied_sae.encoder_bias.requires_grad = False

        with torch.no_grad():
            for i, (images, activations) in enumerate(val_loader):
                activations = activations.to(device)
                c = tied_sae.encode(activations)
                x_hat = tied_sae.decode(c)

                reconstruction_loss = torch.mean((activations - x_hat) ** 2)
                l1_loss = torch.norm(c,1,dim=-1).mean()
                total_loss = reconstruction_loss + args.alpha * l1_loss

                dead_features += c.sum(dim=(0,1)).cpu()

                sparsity = (c != 0).float().mean(dim=(0,1)).sum().cpu().item()
                num_dead_features = (dead_features == 0).sum().cpu().item()

        # log values to tensorboard
        writer.add_scalar('Loss/val', total_loss, epoch)
        writer.add_scalar('Reconstruction Loss/val', reconstruction_loss, epoch)
        writer.add_scalar('L1 Loss/val', l1_loss, epoch)
        writer.add_scalar('Sparsity/val', sparsity, epoch)
        writer.add_scalar('Dead Features/val', num_dead_features, epoch)

        dead_features = torch.zeros(tied_sae.encoder.shape[0])
        
        # Check if validation loss has improved
        if total_loss < best_loss:
            best_loss = total_loss
            counter = 0
            # Save the best model
            torch.save(tied_sae.state_dict, f'SAE_models/{args.modelpath}/SAE_ratio{args.ratio}_epoch{args.epochs}_lr{args.lr}_layer{layer}.pth')
        else:
            counter += 1

        # Check if early stopping criteria is met
        if counter >= patience:
            print("Early stopping. No improvement in validation loss.")
            break

    # close summary writer
    writer.close()
