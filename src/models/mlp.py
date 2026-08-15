from torch import nn

class MLPRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=3, dropout=0.2):
        super().__init__()
        layers = []
        size = input_size
        for _ in range(num_layers):
            layers += [nn.Linear(size, hidden_size), nn.GELU(), nn.Dropout(dropout)]
            size = hidden_size
        layers += [nn.Linear(size, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)
