'''
train_test.py
A file for training model for genre classification.
Please check the device in hparams.py before you run this code.
'''
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

import data_manager
import models
from hparams import hparams

import random
import numpy as np

import sys
from torch.nn.modules.module import _addindent
from functools import reduce
import time

import matplotlib.pyplot as plt

# Wrapper class to run PyTorch model
class Runner(object):
    def __init__(self, hparams):
        self.model = models.Spotify(hparams)
        #self.model = models.Baseline2D_integrated(hparams)
        #self.model = models.Baseline2D_time(hparams)
        #self.model = models.Baseline2D_freq(hparams)
        #self.model = models.VGGNet(hparams)
        #self.model = models.VGGNet_frequency(hparams)
        #self.model = models.VGGNet_time(hparams)
        #self.model = models.Baseline(hparams)
        print(self.model)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=hparams.weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=hparams.factor, patience=hparams.patience, verbose=True)
        self.learning_rate = hparams.learning_rate
        self.stopping_rate = hparams.stopping_rate
        self.device = torch.device("cpu")

        if hparams.device > 0:
            device_num = hparams.device - 1
            self.model.cuda(hparams.device - 1)
            torch.cuda.set_device(device_num)
            self.criterion.cuda(device_num)
            self.device = torch.device("cuda:"+str(device_num))

    # Accuracy function works like loss function in PyTorch
    def accuracy(self, source, target, mode='train'):
        source = source.max(1)[1].long().cpu()
        target = target.cpu()
        if mode =='test':
            print('source', source)
            print('target', target)
        correct = (source == target).sum().item()

        return correct / float(source.size(0))

    # Running model for train, test and validation. mode: 'train' for training, 'eval' for validation and test
    def run(self, dataloader, mode='train'):
        self.model.train() if mode is 'train' else self.model.eval()

        epoch_loss = 0
        epoch_acc = 0
        class_acc = np.zeros((8,8))
        for batch, (x,y) in enumerate(dataloader):
            x = x.to(self.device)
            y = y.to(self.device)

            prediction = self.model(x)
            '''
            prediction_result = prediction.max(1)[1].long().cpu()
            print(prediction_result[2])
            print(y[2].cpu())
            print(prediction_result[2] == y[2].cpu())
            if prediction_result[2] == y[2].cpu():
                print('!!!!')
                print(class_acc[prediction_result[2]][y[2].cpu()])
            '''
            loss = self.criterion(prediction, y)
            acc = self.accuracy(prediction, y, mode)
            if mode is 'train':
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            epoch_loss += prediction.size(0) * loss.item()
            epoch_acc += prediction.size(0) * acc

        epoch_loss = epoch_loss / len(dataloader.dataset)
        epoch_acc = epoch_acc / len(dataloader.dataset)

        return epoch_loss, epoch_acc

    def test_run(self, dataloader):
        epoch_loss = 0
        epoch_acc = 0
        prediction = 0
        class_acc = np.zeros((8,8))

        for batch, (x,y) in enumerate(dataloader):
            x = x.to(self.device)
            y = y.to(self.device)

            prediction = self.model(x)
            prediction_result = prediction.max(1)[1].long().cpu()
            loss = self.criterion(prediction, y)
            acc = self.accuracy(prediction, y, 'test')
            print('acc', acc)
            for i in range(0,len(y)):
                class_acc[prediction_result[i]][y[i].cpu()] += 1
            print(class_acc)
            epoch_loss += prediction.size(0) * loss.item()
            epoch_acc += prediction.size(0) * acc

        epoch_loss = epoch_loss / len(dataloader.dataset)
        epoch_acc = epoch_acc / len(dataloader.dataset)
        print(class_acc)
        return epoch_loss, epoch_acc, prediction


    # Early stopping function for given validation loss
    def early_stop(self, loss, epoch):
        self.scheduler.step(loss, epoch)
        self.learning_rate = self.optimizer.param_groups[0]['lr']
        stop = self.learning_rate < self.stopping_rate

        return stop


def device_name(device):
    if device == 0:
        device_name = 'CPU'
    else:
        device_name = 'GPU:' + str(device - 1)

    return device_name

def summary(model, file=sys.stdout):
    def repr(model):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = model.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        total_params = 0
        for key, module in model._modules.items():
            mod_str, num_params = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)
            total_params += num_params
        lines = extra_lines + child_lines

        for name, p in model._parameters.items():
            if hasattr(p, 'shape'):
                total_params += reduce(lambda x, y: x * y, p.shape)

        main_str = model._get_name() + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        if file is sys.stdout:
            main_str += ', \033[92m{:,}\033[0m params'.format(total_params)
        else:
            main_str += ', {:,} params'.format(total_params)
        return main_str, total_params

    string, count = repr(model)
    if file is not None:
        if isinstance(file, str):
            file = open(file, 'w')
        print(string, file=file)
        file.flush()

    return count

def get_img(cur_time, train, valid):
    plt.figure(figsize=(10,6))
    plt.plot(train, label='train loss')
    plt.plot(valid, label='valid loss')
    plt.xlabel('epoch', size=10)
    plt.ylabel('loss', size=10)
    plt.legend(loc='upper right')

    plt.savefig('./experiments/' + cur_time + '.png')
    plt.show()

def main():
    cur = time.localtime()
    cur_time = "%04d-%02d-%02d_%02d:%02d" % (cur.tm_year, cur.tm_mon, cur.tm_mday, cur.tm_hour, cur.tm_min)
    f = open('./experiments/' + cur_time + '.txt', 'a')    #f.write("========================================================================\n")

    train_loader, valid_loader, test_loader = data_manager.get_dataloader(hparams)
    runner = Runner(hparams)
    summary(runner.model)
    f.write(str(runner.model))
    f.write('\n')

    print('Training on ' + device_name(hparams.device))
    f.write('Training on ' + device_name(hparams.device) + '\n')
    train_loss_list = []
    valid_loss_list = []

    for epoch in range(hparams.num_epochs):
        train_loss, train_acc = runner.run(train_loader, 'train')
        valid_loss, valid_acc = runner.run(valid_loader, 'eval')
        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)
        print("[Epoch %d/%d] [Train Loss: %.4f] [Train Acc: %.4f] [Valid Loss: %.4f] [Valid Acc: %.4f]" %
              (epoch + 1, hparams.num_epochs, train_loss, train_acc, valid_loss, valid_acc))
        f.write("[Epoch %d/%d] [Train Loss: %.4f] [Train Acc: %.4f] [Valid Loss: %.4f] [Valid Acc: %.4f]" %
              (epoch + 1, hparams.num_epochs, train_loss, train_acc, valid_loss, valid_acc))
        f.write('\n')

        if runner.early_stop(valid_loss, epoch + 1):
            break

    get_img(cur_time, train_loss_list, valid_loss_list)
    test_loss, test_acc, prediction = runner.test_run(test_loader)

    print("Training Finished")
    print("Test Accuracy: %.2f%%" % (100 * test_acc))
    f.write("Test Accuracy: %.2f%%" % (100 * test_acc))

    f.close()

if __name__ == '__main__':
    #random.seed(1234)
    #torch.manual_seed(1234)
    #torch.cuda.manual_seed(1234)
    #np.random.seed(1234)
    #torch.backends.cudnn.enabled = False

    main()