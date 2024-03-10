import torch
import h5py
from transformers import ViTForImageClassification, ViTImageProcessor, CLIPModel, CLIPProcessor
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset
import os
from tqdm import tqdm
from hugginglens.hugginglens import HookedHFTransformer
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--cls', type=int, help='Class of interest')
parser.add_argument('--model', type=str, help='used model', default = "akahana/vit-base-cats-vs-dogs")
args = parser.parse_args()

# Get the class of interest
cls = args.cls

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model

model = ViTForImageClassification.from_pretrained(
    args.model)
processor = ViTImageProcessor(args.model)
model = HookedHFTransformer(model, device)
model.eval()

# Prepare your dataset and dataloader
dataset = load_dataset("Bingsu/Cat_and_Dog")

to_tensor = transforms.Compose([
    transforms.Resize((224, 224)),  # Set the desired image size
    transforms.ToTensor()
])


class HFDataset(Dataset):
    def __init__(self, hf_dataset, processor):
        self.hf_dataset = hf_dataset
        self.processor = processor

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        # Extract image and label from the dataset
        image = self.hf_dataset[idx]['image']
        label = self.hf_dataset[idx]['labels']

        # Preprocess the image
        if 'clip' in args.model:
            inputs = self.processor(text = ['cat','dog'],images=image, return_tensors='pt')
        else:
            inputs = self.processor(images=image, return_tensors='pt')
        image = to_tensor(image)

        # Remove the batch dimension
        pixel_values = inputs['pixel_values'].squeeze()

        return image, pixel_values, label


# Define the labels of interest(storing all activation for all test data ended up with huge data size >1.5T)
class_of_interest= [cls]


def filter_labels(example):
    return example['labels'] in class_of_interest


dataset = HFDataset(
    dataset['train'].filter(filter_labels), processor)

dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

save_dir = f"activations_catdog_{args.model}"

# Create the directory if it doesn't exist
os.makedirs(save_dir, exist_ok=True)


# Create an HDF5 file to store the images, activations, and labels
with h5py.File(f'{save_dir}/catdog_activations_{cls}.h5', 'w') as h5f:
    with torch.no_grad():
        for idx, (image, pixel_values, label) in enumerate(tqdm(dataloader)):
            inputs = pixel_values.to(device)
            labels = label.to(device)
            outputs, activations = model.run_with_cache(inputs)
            
            # Create a group for each instance with ID as the group name
            grp = h5f.create_group(str(idx))
            grp.create_dataset(
                'label', data=labels.detach().clone().cpu())
            grp.create_dataset('image', data=image)

            # Store activations for each layer in the group
            for name, act in activations.items():
                if "embeddings" in name or "query" in name or "key" in name or "value" in name or name.endswith("output"):
                    grp.create_dataset(name, data=act.detach().clone().cpu())

            # Clear the activations dictionary for the next batch
            activations.clear()
