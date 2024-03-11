import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from learned_dict import TiedSAE, UntiedSAE
from transformers import ViTForImageClassification, ViTImageProcessor
import h5py
from torch.utils.data import Dataset
import scipy.ndimage as ndimage
from torchvision import transforms
import matplotlib.gridspec as gridspec

class ActivationDataset(Dataset):
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


def plot_activation(vit_model, sae_model, input_image, layer_number, ratio, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), activation_size=768):

    # input image
    if isinstance(input_image, torch.Tensor):
        input_image = transforms.ToPILImage()(input_image)

    # transformer model
    model = ViTForImageClassification.from_pretrained(vit_model, output_hidden_states = True)
    model.to(device)
    model.eval()
    feature_extractor = ViTImageProcessor.from_pretrained(vit_model)
    inputs = feature_extractor(images=input_image, return_tensors="pt")
    inputs.to(device)
    
    # get the activations
    input_activation = model(**inputs)['hidden_states'][layer_number+1].to(device)

    activation_size = activation_size
    n_dict_components = ratio*activation_size

    # initiate the encoder and encoder_bias for SAE
    # torch.manual_seed(0)
    encoder = torch.randn((n_dict_components, activation_size), device = device) # encoder
    nn.init.xavier_uniform_(encoder)
    encoder_bias = torch.zeros(n_dict_components, device = device) # encoder bias
    if 'untied' in sae_model:
        decoder = torch.randn((activation_size, n_dict_components), device = device) # decoder
        nn.init.xavier_uniform_(decoder)

        sae = UntiedSAE(encoder, decoder, encoder_bias)
        sae.load_state_dict(torch.load(sae_model))
        sae.to_device(device)
    else:
        sae = TiedSAE(encoder, encoder_bias)
        sae.load_state_dict(torch.load(sae_model))
        sae.to_device(device)

    activation = sae.encode(torch.Tensor(input_activation)).squeeze(0)[1:,:]

    mean_hidden = activation.mean(dim=0).detach().cpu().numpy() #average of each neurone's activations across patches

    mean_activation = activation.mean(dim=1) #average of each patch's activations

    # Define heatmap
    heatmap = mean_activation.view(14, -1).detach().cpu().numpy()
    plt.imshow(heatmap, cmap='jet')
    input_image = transforms.ToTensor()(input_image)
    # Calculate the zoom factor
    zoom_factor = input_image.shape[1] / heatmap.shape[0], input_image.shape[2] / heatmap.shape[1]

    # Resize the heatmap
    heatmap_resized = ndimage.zoom(heatmap, zoom_factor)

    # Create a gridspec object
    gs = gridspec.GridSpec(2, 3)

    # Create the subplots
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])
    ax3 = plt.subplot(gs[0, 2])
    ax4 = plt.subplot(gs[1, :])  

    ax1.imshow(input_image.permute(1,2,0))
    ax1.set_title('Original Image')
    ax1.axis('off')

    ax2.imshow(heatmap_resized, cmap='jet')
    ax2.set_title('Heatmap')
    ax2.axis('off')

    ax3.imshow(input_image.permute(1,2,0))
    ax3.imshow(heatmap_resized, cmap='jet', alpha=0.5)
    ax3.set_title('Overlay')

    ax4.bar(range(len(mean_hidden)), mean_hidden, ec='blue')
    ax4.set_ylabel('Activation level')
    max_height = mean_hidden.max()*1.1
    ax4.set_ylim(0, max_height)
    plt.tight_layout()
    
    plt.show()
