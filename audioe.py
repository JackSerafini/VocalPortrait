from torch import nn

class AudioEncoder(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super(AudioEncoder, self).__init__()
        self.latent_dim = latent_dim

        # TODO: Get input shape -> eventually change architecture
        self.encoder = nn.Sequential(
            # Layer 1: ???
            nn.Conv1d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Layer 2: ...
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Layer 3: ...
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),            
        )
        self.fc = nn.Linear(256 * 8 * 8, latent_dim)

    def forward(self, audio):
        h = self.encoder(audio)
        h = h.view(h.size(0), -1)
        return self.fc(h)