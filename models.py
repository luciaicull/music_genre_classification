'''
model_archive.py
A file that contains neural network models.
You can also implement your own model here.
'''
import torch
import torch.nn as nn

class VGGNet_time(nn.Module):
    def __init__(self, hparams):
        super(VGGNet_time, self).__init__()
        complexity = 8
        self.layer0 = nn.Sequential(
            nn.Conv2d(1, complexity, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(complexity),
            nn.LeakyReLU(),
            nn.Conv2d(complexity, complexity, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(complexity),
            nn.LeakyReLU(),
            nn.MaxPool2d((8,4), stride=(8,4))
        )

        self.layer1 = nn.Sequential(
            nn.Conv2d(complexity, complexity*2, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(complexity*2),
            nn.LeakyReLU(),
            nn.MaxPool2d((16,8), stride=(16,8))
        )

        self.fc0 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(32 * complexity*2, complexity*4),
            nn.ReLU()
        )

        self.fc1 = nn.Sequential(
            nn.Linear(complexity*4, complexity*4),
            nn.ReLU()
        )

        self.fc2 = nn.Linear(complexity*4, len(hparams.genres))

    def forward(self, x):
        x = x.transpose(2, 3)
        x = self.layer0(x)
        x = self.layer1(x)
        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
        x = self.fc0(x)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

class VGGNet_frequency(nn.Module):
    def __init__(self, hparams):
        super(VGGNet_frequency, self).__init__()
        complexity = 8
        self.layer0 = nn.Sequential(
            nn.Conv2d(1, complexity, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(complexity),
            nn.LeakyReLU(),
            nn.Conv2d(complexity, complexity, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(complexity),
            nn.LeakyReLU(),
            nn.MaxPool2d((4,16), stride=(4,16))
        )

        self.layer1 = nn.Sequential(
            nn.Conv2d(complexity, complexity*2, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(complexity*2),
            nn.LeakyReLU(),
            nn.MaxPool2d((4,64), stride=(4,1))
        )
        self.fc0 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(8 * complexity*2, complexity*4),
            nn.ReLU()
        )

        self.fc1 = nn.Sequential(
            nn.Linear(complexity*4, complexity*4),
            nn.ReLU()
        )

        self.fc2 = nn.Linear(complexity*4, len(hparams.genres))

    def forward(self, x):
        x = x.transpose(2, 3)
        x = self.layer0(x)
        x = self.layer1(x)

        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
        x = self.fc0(x)
        x = self.fc1(x)
        x = self.fc2(x)

        return x


class VGGNet(nn.Module):
    def __init__(self, hparams):
        super(VGGNet, self).__init__()

        self.layer0 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.MaxPool2d((4,4), stride=(4,4))
        )

        self.layer1 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.MaxPool2d((4,4), stride=(4,4))
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d((4,4), stride=(4,4))
        )

        self.fc0 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1024, 64),
            nn.LeakyReLU()
        )

        self.fc1 = nn.Sequential(
            nn.Linear(64, 64),
            nn.LeakyReLU()
        )

        self.fc2 = nn.Linear(64, len(hparams.genres))

    def forward(self, x):
        x = x.transpose(2, 3)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
        x = self.fc0(x)
        x = self.fc1(x)
        x = self.fc2(x)

        return x


class Baseline2D_integrated(nn.Module):
    def __init__(self, hparams):
        super(Baseline2D_integrated, self).__init__()

        self.conv0_t = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.MaxPool2d(8, stride=8)
        )

        self.conv0_f = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.MaxPool2d((4,8), stride=(4,8))
        )

        self.conv1_t = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(8, stride=8)
        )

        self.conv1_f = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d((2,16), stride=(2,16))
        )

        self.conv2_t = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.conv2_f = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d((2,8), stride=(2,8))
        )


        self.fc0_t = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, 64)
        )

        self.fc0_f = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, 64)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(128, len(hparams.genres)),
        )

    def forward(self, input):
        input = input.transpose(2, 3)

        x_f = self.conv0_f(input)
        x_f = self.conv1_f(x_f)
        x_f = self.conv2_f(x_f)
        x_f = x_f.view(x_f.size(0), x_f.size(1) * x_f.size(2) * x_f.size(3))
        x_f = self.fc0_f(x_f)

        x_t = self.conv0_t(input)
        x_t = self.conv1_t(x_t)
        x_t = self.conv2_t(x_t)
        x_t = x_t.view(x_t.size(0), x_t.size(1) * x_t.size(2) * x_t.size(3))
        x_t = self.fc0_t(x_t)

        x = torch.cat((x_f, x_t), dim=1)
        x = self.fc1(x)
        return x


class Baseline2D_freq(nn.Module):
    def __init__(self, hparams):
        super(Baseline2D_freq, self).__init__()

        self.conv0 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.MaxPool2d((4,8), stride=(4,8))
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d((2,16), stride=(2,16))
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d((2,8), stride=(2,8))
        )


        self.fc0 = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, 64)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(64, len(hparams.genres)),
        )

    def forward(self, input):
        input = input.transpose(2, 3)
        x = self.conv0(input)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
        x = self.fc0(x)
        x = self.fc1(x)

        return x

class Baseline2D_time(nn.Module):
    def __init__(self, hparams):
        super(Baseline2D_time, self).__init__()

        self.conv0 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.MaxPool2d(8, stride=8)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(8, stride=8)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, stride=2)
        )


        self.fc0 = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, 64)
        )

        self.fc1 = nn.Linear(64, len(hparams.genres))

    def forward(self, input):
        input = input.transpose(2, 3)
        x = self.conv0(input)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
        x = self.fc0(x)
        x = self.fc1(x)

        return x

class Spotify(nn.Module):
    def __init__(self, hparams):
        super(Spotify, self).__init__()

        self.conv0 = nn.Sequential(
            nn.Conv1d(hparams.num_mels, 32, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.MaxPool1d(8, stride=8)
        )

        self.conv1 = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.MaxPool1d(4, stride=4)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.MaxPool1d(4, stride=4)
        )

        self.pool0 = nn.AvgPool1d(4, stride=4)
        self.pool1 = nn.MaxPool1d(4, stride=4)
        self.pool2 = nn.LPPool1d(2, kernel_size=4, stride=4)
        self.pool3 = nn.LPPool1d(1, kernel_size=4, stride=4)

        self.fc1 = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(192, 64)
        )
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, len(hparams.genres))

    def forward(self, x):
        x = x.transpose(1,2)
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)

        x = torch.cat((self.pool0(x), self.pool1(x), self.pool2(x)),1)
        x = x.view(x.size(0), x.size(1) * x.size(2))
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


class Baseline(nn.Module):
    def __init__(self, hparams):
        super(Baseline, self).__init__()

        self.conv0 = nn.Sequential(
            nn.Conv1d(hparams.num_mels, 32, kernel_size=8, stride=1, padding=0),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.MaxPool1d(8, stride=8)
        )

        self.conv1 = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=8, stride=1, padding=0),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.MaxPool1d(8, stride=8)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.MaxPool1d(4, stride=4)
        )

        self.linear = nn.Linear(192, len(hparams.genres))

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), x.size(1) * x.size(2))
        x = self.linear(x)

        return x

