from transfernet.utils import freeze
from torch import nn, relu


class AppendModel(nn.Module):

    def __init__(self, pretrained_model, new_model):

        super(AppendModel, self).__init__()

        # Expose last layer
        pretrained_model = pretrained_model.children()
        pretrained_model = nn.Sequential(*list(pretrained_model)[:-1])

        # Count hidden layers except last
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
        self.fc2 = nn.Linear(24, 12)
        self.fc3 = nn.Linear(12, 6)
        self.fc4 = nn.Linear(6, 1)

    def forward(self, x):

        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = relu(self.fc3(x))
        x = self.fc4(x)

        return x


class ElemNet(nn.Module):

    def __init__(self):

        super(ElemNet, self).__init__()

        bf = 1024
        self.fc01 = nn.LazyLinear(bf)
        self.fc02 = nn.Linear(bf, bf)
        self.fc03 = nn.Linear(bf, bf)
        self.fc04 = nn.Linear(bf, bf)
        self.dropout04 = nn.Dropout(0.8)

        af = 512
        self.fc05 = nn.Linear(bf, af)
        self.fc06 = nn.Linear(af, af)
        self.fc07 = nn.Linear(af, af)
        self.dropout07 = nn.Dropout(0.9)

        bf = 256
        self.fc08 = nn.Linear(af, bf)
        self.fc09 = nn.Linear(bf, bf)
        self.fc10 = nn.Linear(bf, bf)
        self.dropout10 = nn.Dropout(0.7)

        af = 128
        self.fc11 = nn.Linear(bf, af)
        self.fc12 = nn.Linear(af, af)
        self.fc13 = nn.Linear(af, af)
        self.dropout13 = nn.Dropout(0.8)

        bf = 64
        self.fc14 = nn.Linear(af, bf)
        self.fc15 = nn.Linear(bf, bf)

        af = 32
        self.fc16 = nn.Linear(bf, af)

        bf = 1
        self.fc17 = nn.Linear(af, bf)

    def forward(self, x):

        x = relu(self.fc01(x))
        x = relu(self.fc02(x))
        x = relu(self.fc03(x))
        x = relu(self.fc04(x))
        x = self.dropout04(x)

        x = relu(self.fc05(x))
        x = relu(self.fc06(x))
        x = relu(self.fc07(x))
        x = self.dropout07(x)

        x = relu(self.fc08(x))
        x = relu(self.fc09(x))
        x = relu(self.fc10(x))
        x = self.dropout10(x)

        x = relu(self.fc11(x))
        x = relu(self.fc12(x))
        x = relu(self.fc13(x))
        x = self.dropout13(x)

        x = relu(self.fc14(x))
        x = relu(self.fc15(x))

        x = relu(self.fc16(x))

        x = self.fc17(x)

        return x
