'''
feature_extraction.py
A file related with extracting feature.
For the baseline code it loads audio files and extract mel-spectrogram using Librosa.
Then it stores in the './feature' folder.
'''
import os
import numpy as np
import librosa
import random

from hparams import hparams

def load_list(list_name, hparams):
    with open(os.path.join(hparams.dataset_path, list_name)) as f:
        file_names = f.read().splitlines()

    return file_names

def add_noise(data):
    noise = np.random.randn(len(data))
    data_noise = data + 0.005 * noise
    return data_noise

def shift(data):
    return np.roll(data, 1600)

def stretch(data, rate=0.8):
    input_length = hparams.feature_length
    data = librosa.effects.time_stretch(data, rate)
    if len(data) > input_length:
        data = data[:input_length]
    else:
        data = np.pad(data, (0, max(0, input_length - len(data))), "constant")

    return data


def melspectrogram(file_name, hparams, augment=False):
    y, sr = librosa.load(os.path.join(hparams.dataset_path, file_name), hparams.sample_rate)
    '''
    if augment:
        #print(y.shape)
        num = np.random.randint(0, 3)
        #print(num)
        if num == 0:
            y = add_noise(y)
        elif num == 1:
            y = shift(y)
        elif num == 2:
            rate = np.random.random() * 2  # 0.0 ~ 2.0 float
            y = stretch(y, rate)
        #print(y.shape)
    '''
    '''
    if augment == 'noise':
        y = add_noise(y)
    elif augment == 'shift':
        y = shift(y)
    elif augment == 'stretch':
        rate = np.random.random() * 2.5  # 0.0 ~ 2.5 float
        y = stretch(y, rate)
    '''

    S = librosa.stft(y, n_fft=hparams.fft_size, hop_length=hparams.hop_size, win_length=hparams.win_size)

    mel_basis = librosa.filters.mel(hparams.sample_rate, n_fft=hparams.fft_size, n_mels=hparams.num_mels)
    mel_S = np.dot(mel_basis, np.abs(S))
    mel_S = np.log10(1 + 10 * mel_S)
    mel_S = mel_S.T

    return mel_S

def resize_array(arr, length):
    re_array = np.zeros((length, arr.shape[1]))

    if arr.shape[0] >= length:
        re_array = arr[:length]

    else:
        re_array[:arr.shape[0]] = arr

    return re_array

def main():
    print('Extracting Feature')
    list_names = ['train_list.txt', 'valid_list.txt', 'test_list.txt']

    for list_name in list_names:
        set_name = list_name.replace('_list.txt', '')
        file_names = load_list(list_name, hparams)

        for file_name in file_names:
            feature = melspectrogram(file_name, hparams)
            #print(feature.shape)
            feature = resize_array(feature, hparams.feature_length)
            #print(feature.shape)
            save_path = os.path.join(hparams.feature_path, set_name, file_name.split('/')[0])   # save_path = './feature/classical/'
            save_name = file_name.split('/')[1].replace('.wav', '.npy')                         # save_name = 'classical.00011.npy'

            if not os.path.exists(save_path):
                os.makedirs(save_path)

            np.save(os.path.join(save_path, save_name), feature.astype(np.float32))
            print(os.path.join(save_path, save_name))

        '''
        if set_name == 'train':
            for file_name in file_names:
                feature = melspectrogram(file_name, hparams, True)
                #print(feature.shape)
                feature = resize_array(feature, hparams.feature_length)
                #print(feature.shape)
                save_path = os.path.join(hparams.feature_path, set_name, file_name.split('/')[0])   # save_path = './feature/classical/'
                save_name = file_name.split('/')[1].replace('.wav', '.npy')                         # save_name = 'classical.00011.npy'
                tmp = save_name.split('.')
                save_name = tmp[0] + '.' + tmp[1] + '_a.' + tmp[2]

                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                np.save(os.path.join(save_path, save_name), feature.astype(np.float32))
                print(os.path.join(save_path, save_name))
        '''
    print('Finished Extracting Feature')

if __name__ == '__main__':
    random.seed(1234)
    np.random.seed(1234)
    main()