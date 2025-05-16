# Vanilla encoder and it's topological extention
import torch
import torch.optim as optim

import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances

from tqdm import tqdm

from torch.utils.data import DataLoader

# from torch_topological.nn import SignatureLoss
# from torch_topological.nn import VietorisRipsComplex
from modified_components.VR_complex import VietorisRipsComplex
from modified_components.WeightedSignatureLoss import SignatureLoss
from modified_components.DistancesSignatureLoss import DistancesSignatureLoss


class LinearAutoencoder(torch.nn.Module):


    def __init__(self, input_dim, latent_dim=2, layers_dims=None, activation='elu'):
        """Create new autoencoder with pre-defined latent dimension.
        
        Args:
            input_dim: Dimension of input data
            latent_dim: Dimension of latent space (default: 2)
            layers_dims: List of integers representing hidden layer dimensions
                         (e.g., [512, 256, 64] for decreasing dimensions)
                         If None, uses default architecture from original code.
            activation: Either 'relu' or 'elu' (default: 'relu')
        """
        super().__init__()
    
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        if layers_dims is None:
            layers_dims = [515, 466, 392]
        
        if activation.lower() not in ['relu', 'elu']:
            raise ValueError("activation must be either 'relu' or 'elu'")
        self.activation = activation.lower()
        
        encoder_layers = []
        encoder_layers.append(torch.nn.Linear(input_dim, layers_dims[0]))
        encoder_layers.append(torch.nn.ReLU() if self.activation == 'relu' else torch.nn.ELU())
        
        for i in range(len(layers_dims)-1):
            encoder_layers.append(torch.nn.Linear(layers_dims[i], layers_dims[i+1]))
            encoder_layers.append(torch.nn.ReLU() if self.activation == 'relu' else torch.nn.ELU())
        
        encoder_layers.append(torch.nn.Linear(layers_dims[-1], latent_dim))
        self.encoder = torch.nn.Sequential(*encoder_layers)
        
        decoder_layers = []        
        decoder_layers.append(torch.nn.Linear(latent_dim, layers_dims[-1]))
        decoder_layers.append(torch.nn.ReLU() if self.activation == 'relu' else torch.nn.ELU())
        
        for i in range(len(layers_dims)-1, 0, -1):
            decoder_layers.append(torch.nn.Linear(layers_dims[i], layers_dims[i-1]))
            decoder_layers.append(torch.nn.ReLU() if self.activation == 'relu' else torch.nn.ELU())
        
        decoder_layers.append(torch.nn.Linear(layers_dims[0], input_dim))
        self.decoder = torch.nn.Sequential(*decoder_layers)
        
        self.loss_fn = torch.nn.MSELoss()

    def encode(self, x):
        """Embed data in latent space."""
        return self.encoder(x)

    def decode(self, z):
        """Decode data from latent space."""
        return self.decoder(z)

    def forward(self, x):
        """Embeds and reconstructs data, returning a loss."""
        z = self.encode(x)
        x_hat = self.decode(z)
        reconstruction_error = self.loss_fn(x, x_hat)
        return reconstruction_error


class TopologicalAutoencoder(torch.nn.Module):
    """Wrapper for a topologically-regularised autoencoder.

    This class uses another autoencoder model and imbues it with an
    additional topology-based loss term.
    """
    def __init__(self, model, complex_max_dim=2, min_persistence=1, top_loss_dims=[0], dims_weights=None, lam=1.0):
        super().__init__()

        self.lam = lam
        self.model = model
        self.loss = SignatureLoss(p=2, dimensions=top_loss_dims, weights=dims_weights)

        self.vr = VietorisRipsComplex(dim=complex_max_dim, min_persistence=min_persistence, keep_infinite_features=False)

    def forward(self, x):
        z = self.model.encode(x)

        pi_x = self.vr(x)
        pi_z = self.vr(z)

        geom_loss = self.model(x)
        topo_loss = self.loss([x, pi_x], [z, pi_z])

        loss = geom_loss + self.lam * topo_loss
        return loss

class DistancesTopologicalAutoencoder(torch.nn.Module):
    """Wrapper for a topologically-regularised autoencoder.

    This class uses another autoencoder model and imbues it with an
    additional topology-based loss term.
    """
    def __init__(self, model, complex_max_dim=2, top_loss_dims=[0], lam=1.0):
        super().__init__()

        self.lam = lam
        self.model = model
        self.loss = DistancesSignatureLoss(dimensions=top_loss_dims)
        self.vr = VietorisRipsComplex(dim=complex_max_dim)

    def forward(self, x):
        z = self.model.encode(x)

        x_distances = torch.tensor(pairwise_distances(x.detach().cpu()))
        z_distances =  torch.tensor(pairwise_distances(z.detach().cpu()))
        
        pi_x = self.vr(x_distances, treat_as_distances=True)
        pi_z = self.vr(z_distances, treat_as_distances=True)

        geom_loss = self.model(x)
        topo_loss = self.loss([x_distances, pi_x], [z_distances, pi_z])

        loss = geom_loss + self.lam * topo_loss
        return loss