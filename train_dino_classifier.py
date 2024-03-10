import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import ViTForImageClassification, ViTImageProcessor
import numpy as np
import torch.utils.tensorboard as tb
from tqdm.auto import tqdm
import argparse

# Create an argument parser
parser = argparse.ArgumentParser(description='Dino Classifier Training')

# Add arguments
parser.add_argument('--experiment', type=str, default='cifar', help='name of experiment')
parser.add_argument('--dataset', type=str, default='cifar10', help='name of dataset')

# Parse the arguments
args = parser.parse_args()

# Access the arguments
experiment = args.experiment
dataset_name = args.dataset

writer = tb.SummaryWriter(f'runs/{experiment}_dino-vitb16_classifier')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ViTForImageClassification.from_pretrained('facebook/dino-vitb16')
if experiment == 'cifar':
    model.classifier = torch.nn.Linear(model.config.hidden_size, 10)
processor = ViTImageProcessor.from_pretrained('facebook/dino-vitb16')
model.to(device)

train_dataset = load_dataset(dataset_name, split='train[:90%]')
val_dataset = load_dataset(dataset_name, split='train[90%:]')
test_dataset = load_dataset(dataset_name, split='test')

batch_size = 32

class HFDataset(Dataset):
    def __init__(self, hf_dataset, processor):
        self.hf_dataset = hf_dataset
        self.processor = processor

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        # Extract image and label from the dataset
        if experiment == 'catdog':
            image = self.hf_dataset[idx]['image']
            label = self.hf_dataset[idx]['labels']
        elif experiment == 'cifar':
            image = self.hf_dataset[idx]['img']
            label = self.hf_dataset[idx]['label']

        # Preprocess the image
        inputs = self.processor(images=image, return_tensors='pt')

        # Remove the batch dimension
        pixel_values = inputs['pixel_values'].squeeze()

        return pixel_values, label
    
train_data = HFDataset(train_dataset, processor)
val_data = HFDataset(val_dataset, processor)
test_data = HFDataset(test_dataset, processor)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)

for param in model.parameters():
    param.requires_grad = False

for param in model.classifier.parameters():
    param.requires_grad = True

# Define the optimizer and loss function
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

#Train
num_epochs = 1000
best_loss = np.inf
patience = 10
counter = 0

for epoch in tqdm(range(num_epochs)):
    running_loss = 0.0
    for i, (input, label) in enumerate(train_loader):
        input = input.to(device)
        label = label.to(device)

        optimizer.zero_grad()

        output = model(input)
        loss = criterion(output.logits, target=label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    average_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch} Loss/train: {average_loss}")
    writer.add_scalar('Loss/train', average_loss, epoch)

    # validation
    model.eval()
    running_loss = 0.0
    for i, (input, label) in enumerate(val_loader):
        input = input.to(device)
        label = label.to(device)

        with torch.no_grad():
            output = model(input)
            loss = criterion(output.logits,target=label)
            running_loss += loss.item()
        
    average_loss = running_loss / len(val_loader)
    print(f"Epoch {epoch} Loss/val: {average_loss}")
    writer.add_scalar('Loss/val', average_loss, epoch)

    # early stopping
    if average_loss < best_loss:
        best_loss = average_loss
        counter = 0
        torch.save(model.state_dict(), f"{experiment}_dino-vitb16_classifier.pth")
    else:
        counter += 1
        if counter > patience:
            break
    
# test
model.load_state_dict(torch.load(f"{experiment}_dino-vitb16_classifier.pth"))
model.eval()

predicted_labels = []
true_labels = []

for i, (input, label) in enumerate(test_loader):
    input = input.to(device)
    label = label.to(device)

    with torch.no_grad():
        output = model(input)
    predicted_labels.append(torch.argmax(output.logits, dim=1))
    true_labels.append(label)

predicted_labels = torch.cat(predicted_labels)
true_labels = torch.cat(true_labels)
accuracy = (predicted_labels == true_labels).float().mean()
print(f"Test Accuracy: {accuracy}")
writer.add_scalar('Accuracy/test', accuracy, epoch)

writer.close()