'''
hparams.py
A file sets hyper parameters for feature extraction and training.
You can change parameters using argument.
For example:
 $ python train_test.py --device=1 --batch_size=32.
'''

import argparse
import time

class HParams(object):
    def __init__(self):
        # Dataset Settings
        self.dataset_path = './gtzan'
        self.feature_path=  './feature'
        self.genres = ['classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae']

        # Feature Parameters
        self.sample_rate = 22050
        self.fft_size = 1024
        self.win_size = 1024
        self.hop_size = 512
        self.num_mels = 128
        self.feature_length = 1024 # audio length = feature_length*hop_size/sample_rate (s)

        # Training Parameters
        self.device = 1 # 0: CPU, 1: GPU0, 2:GPU1, ... # check with nvidia-smi in server
        self.batch_size = 20
        self.num_epochs = 100
        self.learning_rate = 1e-3
        self.stopping_rate = 1e-5
        self.weight_decay = 1e-3
        self.momentum = 0.9
        self.factor = 0.2
        self.patience = 10

    # Function for parsing argument and set hyper parameters
    def parse_argument(self, print_argument=True):
        cur = time.localtime()
        cur_time = "%04d-%02d-%02d_%02d:%02d" % (cur.tm_year, cur.tm_mon, cur.tm_mday, cur.tm_hour, cur.tm_min)
        f = open('./experiments/' + cur_time + '.txt', 'a')
        f.write("========================================================================\n")

        parser = argparse.ArgumentParser()

        for var in vars(self):              # vars = dict object.
            value = getattr(hparams, var)   # value = device, batch_size, ..
            argument = '--' + var           # argument = --device, --batch_sie, ..
            parser.add_argument(argument, type=type(value), default=value)

        args = parser.parse_args()      # args = set-upped namespace

        for var in vars(self):
            setattr(hparams, var, getattr(args, var))   # hparams.(var) = getattr(args, var)

        if print_argument:
            print('-------------------------')
            print('Hyper Parameter Settings')
            print('-------------------------')
            f.write('-------------------------\n')
            f.write('Hyper Parameter Settings\n')
            f.write('-------------------------\n')
            for var in vars(self):
                value = getattr(hparams, var)
                print(var + ': ' + str(value))
                f.write(var + ': ' + str(value) + '\n')
            print('-------------------------')
            f.write('-------------------------\n')

        f.close()

hparams = HParams()
hparams.parse_argument()

