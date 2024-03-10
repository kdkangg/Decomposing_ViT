from abc import ABC, abstractmethod
import torch
from torchtyping import TensorType

_n_dict_components, _activation_size, _batch_size = None, None, None


class LearnedDict(ABC):
    n_feats: int
    activation_size: int

    @abstractmethod
    def get_learned_dict(self) -> TensorType["_n_dict_components", "_activation_size"]:
        pass

    @abstractmethod
    def encode(self, batch: TensorType["_batch_size", "_activation_size"]) -> TensorType["_batch_size", "_n_dict_components"]:
        pass

    @abstractmethod
    def to_device(self, device):
        pass

    def decode(self, code: TensorType["_batch_size","patch", "_n_dict_components"]) -> TensorType["_batch_size", "_activation_size"]:
        learned_dict = self.get_learned_dict()
        x_hat = torch.einsum("nd,bpn->bpd", learned_dict, code) #added patch dimension
        return x_hat

    def predict(self, batch: TensorType["_batch_size", "_activation_size"]) -> TensorType["_batch_size", "_activation_size"]:
        c = self.encode(batch)
        x_hat = self.decode(c)
        return x_hat

    def n_dict_components(self):
        return self.get_learned_dict().shape[0]


class TiedSAE(LearnedDict):
    def __init__(self, encoder, encoder_bias, norm_encoder=False):
        self.encoder = encoder
        self.encoder_bias = encoder_bias
        self.norm_encoder = norm_encoder
        self.n_feats, self.activation_size = self.encoder.shape

        self.state_dict = {
            'encoder': self.encoder,
            'encoder_bias': self.encoder_bias,
        }
    def get_learned_dict(self):
        norms = torch.norm(self.encoder, 2, dim=-1)
        return self.encoder / torch.clamp(norms, 1e-8)[:, None]

    def to_device(self, device):
        self.encoder = self.encoder.to(device)
        self.encoder_bias = self.encoder_bias.to(device)

    def encode(self, batch):
        if self.norm_encoder:
            norms = torch.norm(self.encoder, 2, dim=-1)
            encoder = self.encoder / torch.clamp(norms, 1e-8)[:, None]
        else:
            encoder = self.encoder

        c = torch.einsum("nd,bpd->bpn", encoder, batch) # added patch dimension
        c = c + self.encoder_bias[None, None, :]
        c = torch.clamp(c, min=0.0) #ReLu
        return c
    
    def load_state_dict(self,state_dict):
        self.encoder = state_dict['encoder']
        self.encoder_bias = state_dict['encoder_bias']


class UntiedSAE(LearnedDict):
    def __init__(self, encoder, decoder, encoder_bias):
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_bias = encoder_bias
        self.n_feats, self.activation_size = self.encoder.shape

        self.state_dict = {
            'encoder': self.encoder,
            'encoder_bias': self.encoder_bias,
            'decoder': self.decoder,
        }
    def get_learned_dict(self):
        norms = torch.norm(self.decoder, 2, dim=-1)
        return self.decoder / torch.clamp(norms, 1e-8)[:, None]

    def to_device(self, device):
        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)
        self.encoder_bias = self.encoder_bias.to(device)

    def encode(self, batch):
        c = torch.einsum("nd,bpd->bpn", self.encoder, batch) # added patch dimension
        c = c + self.encoder_bias
        c = torch.clamp(c, min=0.0)
        return c

    def load_state_dict(self,state_dict):
        self.encoder = state_dict['encoder']
        self.encoder_bias = state_dict['encoder_bias']
        self.decoder = state_dict['decoder']
    
    def decode(self, code: TensorType["_batch_size","patch", "_n_dict_components"]) -> TensorType["_batch_size", "_activation_size"]:
        learned_dict = self.get_learned_dict()
        x_hat = torch.einsum("dn,bpn->bpd", learned_dict, code) #added patch dimension
        return x_hat