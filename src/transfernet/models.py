from transfernet.utils import freeze
from torch import nn, relu


class AppendModel(nn.Module):

    def __init__(self, pretrained_model, new_model):

        super(AppendModel, self).__init__()

        # Expose last layer
        pretrained_model = pretrained_model.children()
        pretrained_model = nn.Sequential(*list(pretrained_model)[:-1])

        # Count hidden layers
        count = 0
        for layer in pretrained_model.children():
            count += 1

        # Freeze layers
        pretrained_model = freeze(pretrained_model, count)

        self.pretrained_model = pretrained_model
        self.new_model = new_model

    def forward(self, x):

        x = self.pretrained_model(x)
        x = x.view(x.size(0), -1)
        x = self.new_model(x)

        return x


class ExampleNet(nn.Module):

    def __init__(self):

        super(ExampleNet, self).__init__()

        self.fc1 = nn.LazyLinear(24)
        self.fc2 = nn.LazyLinear(12)
        self.fc3 = nn.LazyLinear(6)
        self.fc4 = nn.LazyLinear(1)

    def forward(self, x):

        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = relu(self.fc3(x))
        x = self.fc4(x)

        return x


class GeneralNet(nn.Module):

    def __init__(self, arch={24: 1, 12: 1, 6: 1}, batch_norm=False):

        super(GeneralNet, self).__init__()

        self.layers = nn.ModuleList()
        for neurons, layers in arch.items():
            for i in range(layers):
                self.layers.append(nn.LazyLinear(neurons))

                if batch_norm:
                    self.layers.append(nn.BatchNorm1d(neurons))
                self.layers.append(nn.ReLU())

        self.layers.append(nn.LazyLinear(1))

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)

        return x


class ElemNet(GeneralNet):

    def __init__(self):

        # Input parameters
        arch = {1024: 4, 512: 3, 256: 3, 128: 3, 64: 2, 32: 1}
        batch_norm = False

        super(ElemNet, self).__init__(arch, batch_norm)
