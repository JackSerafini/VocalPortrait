import torch
from torch import nn

class MLPAudioToLatent(nn.Module):
    """
    Takes an audio embedding vector of dim. 768 and outputs mean and variance
    associated with a corresponding face
    """
    def __init__(self, input_dim=768, hidden_dims=[512, 256, 128], output_dim=2):
        super(MLPAudioToLatent, self).__init__()
        
        layers = []
        last_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, h_dim))
            layers.append(nn.ReLU())
            last_dim = h_dim
        
        layers.append(nn.Linear(last_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

    # KL divergence loss between the audio's predicted (mu, logvar) and VAE's (mu, logvar).
    def kl_divergence_loss(mu1, logvar1, mu2, logvar2) -> torch.Tensor:
        return 0.5 * torch.sum(
            logvar2 - logvar1 + (torch.exp(logvar1) + (mu1 - mu2) ** 2) / torch.exp(logvar2) - 1, dim=-1
        ).mean()