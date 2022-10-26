import requests
import tarfile
import os
import numpy as np
import sys
import pickle
from tqdm import tqdm
import pandas as pd
import librosa as lb
import random
import shutil


class Dataset(object):
    def __init__(self):
        self.file_labels = None
        self.data = {}
        self.index = {"train": 0, "validate": 0, "test": 0}

    def init_dataset(self):
        """
        Entry point during initialization of training dataset.
        :return: None
        """
        self.check_and_download_data()
        self.create_train()

    def init_test(self):
        """
        Entry point during initialization of testing dataset.
        :return: None
        """
        self.check_and_download_data()
        self.create_test()

    def __getitem__(self, item):
        """
        Return dataset of given type.
        :param item: Type of the dataset. Possible options -> [train, validate, test]
        :return: dataset of given type
        """
        return self.data[item]

    def create_batches(self, val_size=15, test_size=10):
        """
        Create ready datasets for training, validation and testing.
        :param val_size: number of tracks from each genre, which will be added to validation dataset
        :param test_size: number of tracks from each genre, which will be added to testing dataset
        :return: None
        """
        print("Creating datasets from audio files... Please, wait.")
        train_index = None
        val_index = None
        test_index = None
        train_size = 100 - val_size - test_size

        # split every track into datasets with even number tracks from each genre
        for genre in range(10):
            df = self.file_labels.loc[self.file_labels['label'] == genre]
            rows = df.sample(frac=1).reset_index(drop=True).values
            if train_index is None:
                train_index = rows[0:train_size, :]
                val_index = rows[train_size:train_size+val_size, :]
                test_index = rows[train_size+val_size:, :]
            else:
                train_index = np.vstack((train_index, rows[0:train_size, :]))
                val_index = np.vstack((val_index, rows[train_size:train_size+val_size, :]))
                test_index = np.vstack((test_index, rows[train_size+val_size:, :]))

        # create processed audio and store it
        print("Creating train dataset...")
        train_data, train_labels = self.load_data(train_index)
        print("Creating validation dataset...")
        val_data, val_labels = self.load_data(val_index)
        print("Creating test dataset...")
        test_data, test_labels = self.load_data(test_index)

        # pickle this data for later use
        with open("data/train.pkl", "wb+") as f:
            pickle.dump({"data": train_data, "labels": train_labels, "size": len(train_labels)}, f)
        with open("data/validate.pkl", "wb+") as f:
            pickle.dump({"data": val_data, "labels": val_labels, "size": len(val_labels)}, f)
        with open("data/test.pkl", "wb+") as f:
            pickle.dump({"data": test_data, "labels": test_labels, "size": len(test_labels)}, f)

        # cleanup downloaded files
        for file_path in self.file_labels.iloc[:, 0]:
            os.remove(file_path)
        shutil.rmtree('data/genres')
        print("Datasets ready. Returning to training/testing...")

    def create_train(self):
        """
        Load train and validate datasets to computer memory.
        :return: None
        """
        with open("data/train.pkl", "rb") as f:
            self.data["train"] = pickle.load(f)
        with open("data/validate.pkl", "rb") as f:
            self.data["validate"] = pickle.load(f)

    def create_test(self):
        """
        Load test dataset to computer memory.
        :return:
        """
        with open("data/test.pkl", "rb") as f:
            self.data["test"] = pickle.load(f)

    def check_and_download_data(self):
        """
        Download files and create train/validate/test sets if needed.
        :return: None
        """
        # check if datasets exist
        all_sets_exists = True
        dataset = ["train", "validate", "test"]
        for _set in dataset:
            if not os.path.exists(os.path.join("data", f"{_set}.pkl")):
                all_sets_exists = False
                break
        if all_sets_exists:
            return
        print("Datasets -> not found.")
        # if datasets don't exist, check if all audio files exist
        all_files_exists = True
        if os.path.exists('data/labels.csv'):
            labels_file = 'data/labels.csv'
            self.file_labels = pd.read_csv(labels_file, header=0)
            root = os.path.dirname(sys.modules['__main__'].__file__)
            paths_to_files = self.file_labels.iloc[:, 0]
            for path in paths_to_files:
                if not os.path.exists(os.path.join(root, path)):
                    all_files_exists = False
                    print("All needed audio files -> not found.")
                    break
        else:
            all_files_exists = False
            print("labels.csv file -> not found.")
        # if audio files exists and datasets are not created, create them
        if all_files_exists:
            if not all_sets_exists:
                self.create_batches()
            return

        # clean data folder and create new one
        if os.path.exists('data/genres'):
            shutil.rmtree('data/genres')
        os.mkdir("data/genres")

        # download compressed file with audio data
        url = 'http://opihi.cs.uvic.ca/sound/genres.tar.gz'
        file_path = './genres.tar.gz'
        self.download_from_url(url, file_path)

        print("Extracting data from ./genres.tar.gz file...")
        # todo: check if file is not corrupted
        # extract files from tar file
        with tarfile.open(file_path, "r:gz") as file:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(file, "data")
        self.create_batches()

    @staticmethod
    def download_from_url(url, path):
        """
        Download file from URL and save it under path.
        :param url: URL of the file, that needs to be downloaded.
        :param path: path to output file, where downloaded file will be saved.
        :return: None
        """
        if os.path.exists(path):
            first_byte = os.path.getsize(path)
            print("genres.tar.gz file -> found.")
        else:
            print("genres.tar.gz file -> not found.")
            first_byte = 0
        chunk_size = 1024
        total_size = int(requests.head(url).headers["Content-Length"])
        if first_byte >= total_size:
            return
        if os.path.exists(path):
            print("Continuing download of this file...")
        else:
            print("Starting download of this file...")

        header = {"Range": "bytes=%s-%s" % (first_byte, total_size)}
        response = requests.get(url, headers=header, stream=True)
        bar = tqdm(total=total_size, initial=first_byte, unit='B', unit_scale=True, desc=url.split('/')[-1])
        with open("genres.tar.gz", "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    bar.update(chunk_size)
        bar.close()

    def load_data(self, dataset):
        """
        Process audio, create labels and return ready dataset.
        :param dataset: matrix, in which rows consist of the path to audio and corresponding label.
        :return: shuffled pairs of processed audio and labels
        """
        d = np.asarray([self.process_data(path) for path in tqdm(dataset[:, 0])])
        data = d.reshape((d.shape[0], d.shape[1], d.shape[2], 1))
        labels = dataset[:, 1]
        labels = self.convert_to_one_hot(labels.tolist())
        rand = list(zip(data, labels))
        random.shuffle(rand)
        data, labels = zip(*rand)
        return data, labels

    @staticmethod
    def process_data(path):
        """
        Create spectrogram in mel scale from audio file.
        :param path: path to audio file
        :return: spectrogram in mel scale with processed audio
        """
        fs = 11000
        n_fft = 512
        n_mels = 96
        n_overlap = 256
        dura = 30.0
        signal, sr = lb.load(path, sr=fs)
        n_sample = signal.shape[0]
        n_sample_fit = int(dura * fs)

        if n_sample < n_sample_fit:
            signal = np.hstack((signal, np.zeros((int(dura * fs) - n_sample,))))
        elif n_sample > n_sample_fit:
            signal = signal[int((n_sample - n_sample_fit) / 2):int((n_sample + n_sample_fit) / 2)]

        melspect = lb.core.power_to_db(
            lb.feature.melspectrogram(y=signal, sr=fs, hop_length=n_overlap, n_fft=n_fft, n_mels=n_mels))

        return melspect

    @staticmethod
    def convert_to_one_hot(labels):
        """
        Convert vector of labels to one hot format.
        :param labels: vector with integers, which .. labels
        :return: matrix of one hot labels
        """
        numbers = np.max(labels) + 1
        return np.eye(numbers)[labels]

    def next_batch(self, type, number):
        """
        Get next batch of data from dataset.
        :param type: type of dataset, possible types [train, validate, test]
        :param number: batch size
        :return: batch with data and labels from given dataset
        """
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


