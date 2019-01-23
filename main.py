# Diagnostyka przemysłowa (fault detection and isolation)
# Klasyfikacja utworów muzyczych do różnych gatunków muzycznych
import tensorflow as tf
import os
from src.dataset import Dataset
from src.crnn import CRNN


def main():
    dataset = Dataset()
    net = CRNN(dataset)
    net.start()


if __name__ == "__main__":
    main()