import requests
import tarfile
import pickle
import os
import numpy as np
import sys
from tqdm import tqdm
import pandas as pd


class Dataset(object):
    def __init__(self):
        self.data = {}
        self.index = {"train": 0, "validate": 0, "test": 0}
        self.check_and_download_data()
        self.create_batches("data/cifar-10-batches-py")

    def __getitem__(self, item):
        return self.data[item]

    def create_batches(self, path, val_size=5000):
        data, label = self.load_data(path, "data")

        data = np.multiply(data, 1/255)

        val_data = data[:val_size]
        val_labels = label[:val_size]
        train_data = data[val_size:]
        train_labels = label[val_size:]

        test_data, test_labels = self.load_data(path, "test")

        test_data = np.multiply(test_data, 1/255)

        self.data["train"] = {"data": train_data, "labels": train_labels, "size": len(train_labels)}
        self.data["validate"] = {"data": val_data, "labels": val_labels, "size": len(val_labels)}
        self.data["test"] = {"data": test_data, "labels": test_labels, "size": len(test_labels)}

    def check_and_download_data(self):
        labels_file = 'data/labels.csv'
        labels = pd.read_csv(labels_file, header=0)
        # todo check if files under this labels works
        root = os.path.dirname(sys.modules['__main__'].__file__)
        # todo download and extract gztan dataset
        if not os.path.exists('./cifar-10-python.tar.gz'):
            url = 'http://opihi.cs.uvic.ca/sound/genres.tar.gz'
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-lenght', 0))
            block_size = 1024
            with open("cifar-10-python.tar.gz", "wb") as handle:
                for data in tqdm(response.iter_content(chunk_size=block_size), total=math.ceil(total_size//block_size),
                                 unit='kB', unit_scale=True):
                    handle.write(data)
        with tarfile.open("./cifar-10-python.tar.gz", "r:gz") as file:
            file.extractall('data')

    def load_data(self, path, name):
        data = None
        label = None
        files = [f for f in os.listdir(path) if name in f]
        for i, file in enumerate(files):
            name = os.path.join(path, file)
            with open(name, "rb") as f:
                dict = pickle.load(f, encoding="latin1")
                labels = self.convert_to_one_hot(dict["labels"])
                if data is None:
                    data = dict["data"]
                    label = labels
                else:
                    data = np.concatenate((data, dict["data"]))
                    label = np.concatenate((label, labels))
        return data, label

    @staticmethod
    def convert_to_one_hot(labels):
        numbers = np.max(labels) + 1
        return np.eye(numbers)[labels]

    def next_batch(self, type, number):
        # todo dodać losowe przertwarzenie obrazu ( lustrzane odbicie, przyciemnienie/rozjaśnienie )
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
