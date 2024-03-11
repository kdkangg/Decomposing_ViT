import torch
import torch.optim as optim
import torch.nn as nn
from transformers.models.vit.modeling_vit import ViTOutput
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import random

def sae_feature_vis(model, neuron_index, input_size, sae_layer=-1, ratio = 10, num_iterations=1000, lr=0.1, device = 'cuda',augment:nn.Module = None, image=None, lambda_tv = 0.0005, show_intermediate = False):
    image = torch.randn(1, 3, input_size, input_size, device=device)
    image = image.clone().detach().requires_grad_(True)
    model.to(device).eval()
    model.requires_grad = False

    activation = []
    def hook_fn(module, input, output):
        activation.append(output)

    try:
        handle = model.vit.encoder.layer[sae_layer].encode.register_forward_hook(hook_fn)
    except:
        return "Layer not found"

    def loss_fn(hidden):
        hidden = hidden.view(-1,768*ratio)
        hidden = hidden[:, neuron_index]
        return -hidden.mean()

    optimizer = optim.Adam([image], lr=lr, betas = (0.5,0.99),eps=1e-8)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    image.requires_grad = True

    for i in tqdm(range(num_iterations)):
        optimizer.zero_grad()
        if augment:
            augmented = augment(image)
        else:
            augmented = image
        
        output = model(augmented)
        h = activation[0]
        
        loss = loss_fn(h)
        loss += TotalVariation(2)(h) * lambda_tv
        
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        activation = [] # Clear the list
        if show_intermediate:
            if i % 100 == 0:
                print(f'Iteration: {i}')
                print(f'Loss: {loss.item()}')
                plt.imshow(image[0].permute(1, 2, 0).detach().cpu().numpy())
                plt.show()
        image.data = Clip()(image)
        torch.cuda.empty_cache()
    handle.remove()
    return image.detach().cpu()

def feature_vis(model,layer, neuron_index, input_size, num_iterations=1000, lr=0.1, device = 'cuda',augment:nn.Module = None, image=None, lambda_tv = 0.00005, show_intermediate = False):
    # Initialize random input image
    image = torch.randn(1, 3, input_size, input_size, device=device)
    image = image.clone().detach().requires_grad_(True)
    model.to(device).eval()
    model.requires_grad = False
    
    # Define loss function
    def loss_fn(hidden):
        hidden = hidden.view(-1,768)
        hidden = hidden[:, neuron_index]
        return -hidden.mean()

    # Create optimizer
    optimizer = optim.Adam([image], lr=lr, betas = (0.5,0.99),eps=1e-8)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    image.requires_grad = True
    
    # Optimization loop
    for i in tqdm(range(num_iterations)):
        optimizer.zero_grad()
        if augment:
            augmented = augment(image)
        else:
            augmented = image
        # Forward pass through the model
        h = model(augmented, output_hidden_states=True)['hidden_states'][layer+1]

        # Compute loss
        loss = loss_fn(h)
        loss += TotalVariation(2)(h) * lambda_tv
        
        # Backward pass and optimization step
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        if show_intermediate:
            if i % 100 == 0:
                print(f'Iteration: {i}')
                print(f'Loss: {loss.item()}')
                plt.imshow(image[0].permute(1, 2, 0).detach().cpu().numpy())
                plt.show()
        image.data = Clip()(image)
        torch.cuda.empty_cache()
    return image.detach()

def new_init(size, batch_size):
    output = torch.randn(size=(batch_size,3,size,size))
    output.requires_grad = True
    return output

class Clip(nn.Module):
    @torch.no_grad()
    def forward(self, x: torch.tensor) -> torch.tensor:
        return x.clamp(min=0, max=1)
class Jitter(nn.Module):
    def __init__(self, lim: int = 32):
        super().__init__()
        self.lim = lim

    def forward(self, x: torch.tensor) -> torch.tensor:
        off1 = random.randint(-self.lim, self.lim)
        off2 = random.randint(-self.lim, self.lim)
        return torch.roll(x, shifts=(off1, off2), dims=(2, 3))
    
class TotalVariation(nn.Module):
    def __init__(self, p: int = 2):
        super().__init__()
        self.p = p

    def forward(self, x: torch.tensor) -> torch.tensor:
        x_wise = x[:, :, 1:] - x[:, :, :-1]
        y_wise = x[:, 1:, :] - x[:, :-1, :]
        diag_1 = x[:, 1:, 1:] - x[:, :-1, :-1]
        diag_2 = x[:, 1:, :-1] - x[:, :-1, 1:]
        return x_wise.norm(p=self.p, dim=(1, 2)).mean() + y_wise.norm(p=self.p, dim=(1, 2)).mean() + \
               diag_1.norm(p=self.p, dim=(1, 2)).mean() + diag_2.norm(p=self.p, dim=(1, 2)).mean()

class ColorJitter(nn.Module):
    def __init__(self, batch_size: int, mean: float = 1., std: float = 1.):
        super().__init__()
        self.batch_size = batch_size
        self.mean_p = mean
        self.std_p = std
        self.mean = self.std = None
        self.shuffle()

    def shuffle(self):
        self.mean = (torch.rand((self.batch_size, 3, 1, 1,)).cuda() - 0.5) * 2 * self.mean_p
        self.std = ((torch.rand((self.batch_size, 3, 1, 1,)).cuda() - 0.5) * 2 * self.std_p).exp()

    def forward(self, img: torch.tensor) -> torch.tensor:
        self.shuffle()
        return (img - self.mean) / self.std

class Tile(nn.Module):
    def __init__(self, rep: int = 224 // 16):
        super().__init__()
        self.rep = rep

    def forward(self, x: torch.tensor) -> torch.tensor:
        dim = x.dim()
        if dim < 3:
            raise NotImplementedError
        elif dim == 3:
            x.unsqueeze(0)
        final_shape = x.shape[:2] + (x.shape[2] * self.rep, x.shape[3] * self.rep)
        return x.unsqueeze(2).unsqueeze(4).repeat(1, 1, self.rep, 1, self.rep, 1).view(final_shape)
    
class GausianNoise(nn.Module):
    def __init__(self, batch_size: int, mean: float = 0., std: float = 1., max_iter=400):
        super().__init__()
        self.batch_size = batch_size
        self.mean = mean
        self.std = std

    def forward(self, img: torch.tensor) -> torch.tensor:
        noise = torch.rand((self.batch_size, 3, 1, 1)).cuda() * self.std + self.mean
        return img + noise     

class RepeatBatch(nn.Module):
    def __init__(self, num_repeats: int):
        super().__init__()
        self.num_repeats = num_repeats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.repeat(self.num_repeats, 1, 1, 1)


class ModifiedLayer(ViTOutput):
    def __init__(self, config, original_weight, encoder_weight, decoder_weight):
        super().__init__(config)
        self.original_weight = original_weight
        self.encoder_weight = encoder_weight
        self.decoder_weight = decoder_weight

        self.dense.weight = torch.nn.Parameter(original_weight)
        self.dense2.weight = torch.nn.Parameter(encoder_weight)
        self.dense3.weight = torch.nn.Parameter(decoder_weight)
    
    def forward(self, hidden_states, input_tensor):
        h = self.dense(hidden_states)
        h = self.dense2(h)
        h = self.dense3(h)
        h = self.dropout(h)
        h = h + input_tensor
        return h

# Create the sae_layer module
class sae_encoder(nn.Module):
    def __init__(self):
        super(sae_encoder, self).__init__()
        self.encode = nn.Linear(768,7680)
        
    def forward(self, x):
        x = self.encode(x)
        return x

class sae_decoder(nn.Module):
    def __init__(self):
        super(sae_decoder, self).__init__()
        self.decode = nn.Linear(7680, 768, bias=False)
    
    def forward(self, x):
        x = self.decode(x)
        return x

class Added_layer(nn.Module): 
    def __init__(self, original_layer, encoder, decoder):
        super(Added_layer, self).__init__()
        self.original_layer = original_layer
        self.encode = encoder
        self.decode = decoder
    
    def forward(self, x, head_mask = None, output_attentions = None):
        x = self.original_layer(x)
        x = self.encode(x[0])
        x = self.decode(x)
        return tuple(x.unsqueeze(0))
    