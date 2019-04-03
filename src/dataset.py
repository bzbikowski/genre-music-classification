import requests
import tarfile
import os
import numpy as np
import sys
from tqdm import tqdm
import pandas as pd
import math
import librosa as lb
import random


class Dataset(object):
    def __init__(self):
        self.file_labels = None
        self.data = {}
        self.index = {"train": 0, "validate": 0, "test": 0}

    def init_dataset(self):
        self.check_and_download_data()
        self.create_batches()

    def init_test(self):
        # todo check if model exists
        self.check_and_download_data()
        self.create_test()

    def __getitem__(self, item):
        return self.data[item]

    def create_batches(self, val_size=300):
        data, label = self.load_data()

        rand = list(zip(data, label))
        random.shuffle(rand)
        data, label = zip(*rand)

        train_size = 1000 - val_size
        train_data = data[0:train_size]
        train_labels = label[0:train_size]
        val_data = data[train_size:]
        val_labels = label[train_size:]

        self.data["train"] = {"data": train_data, "labels": train_labels, "size": len(train_labels)}
        self.data["validate"] = {"data": val_data, "labels": val_labels, "size": len(val_labels)}

    def create_test(self):
        data, label = self.load_data()

        rand = list(zip(data, label))
        random.shuffle(rand)
        data, label = zip(*rand)

        self.data["test"] = {"data": data, "labels": label, "size": len(label)}

    def check_and_download_data(self):
        # todo check if it's working
        allFilesExists = True
        if os.path.exists('data/labels.csv'):
            labels_file = 'data/labels.csv'
            self.file_labels = pd.read_csv(labels_file, header=0)
            root = os.path.dirname(sys.modules['__main__'].__file__)
            pathsToFiles = self.file_labels.iloc[:, 0]
            for path in pathsToFiles:
                if not os.path.exists(os.path.join(root, path)):
                    allFilesExists = False
                    break
        else:
            allFilesExists = False
        if allFilesExists:
            return
        # todo clean all data
        if not os.path.exists('./genres.tar.gz'):
            url = 'http://opihi.cs.uvic.ca/sound/genres.tar.gz'
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-lenght', 0))
            block_size = 1024
            with open("genres.tar.gz", "wb") as handle:
                for data in tqdm(response.iter_content(chunk_size=block_size), total=math.ceil(total_size//block_size),
                                 unit='kB', unit_scale=True):
                    handle.write(data)
        with tarfile.open("./genres.tar.gz", "r:gz") as file:
            file.extractall('data')

    def load_data(self):
        d = np.asarray([self.process_data(path) for path in self.file_labels.iloc[:, 0]])
        data = d.reshape(d.shape[0], d.shape[1], d.shape[2], 1)
        labels = self.file_labels.iloc[:, 1]
        labels = self.convert_to_one_hot(labels)
        return data, labels

    @staticmethod
    def convert_to_one_hot(labels):
        numbers = np.max(labels) + 1
        return np.eye(numbers)[labels]

    def next_batch(self, type, number):
        index = self.index[type]
        if index + number < self.data[type]["size"]:
            data = self.data[type]["data"][index:index+number]
            labels = self.data[type]["labels"][index:index+number]
            self.index[type] += number
        else:
            data = self.data[type]["data"][index:]
            labels = self.data[type]["labels"][index:]
            self.index[type] = 0
        return [np.array(data), np.array(labels)]

    def process_data(self, path):
        Fs = 11000
        N_FFT = 512
        N_MELS = 96
        N_OVERLAP = 256
        DURA = 30.0
        signal, sr = lb.load(path, sr=Fs)
        n_sample = signal.shape[0]
        n_sample_fit = int(DURA * Fs)

        if n_sample < n_sample_fit:
            signal = np.hstack((signal, np.zeros((int(DURA * Fs) - n_sample,))))
        elif n_sample > n_sample_fit:
            signal = signal[int((n_sample - n_sample_fit) / 2):int((n_sample + n_sample_fit) / 2)]

        melspect = lb.core.power_to_db(
            lb.feature.melspectrogram(y=signal, sr=Fs, hop_length=N_OVERLAP, n_fft=N_FFT, n_mels=N_MELS))

        return melspect
