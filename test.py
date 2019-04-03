# Diagnostyka przemysłowa (fault detection and isolation)
# Klasyfikacja utworów muzyczych do różnych gatunków muzycznych
from src.dataset import Dataset
from src.cnn import CNN


def main():
    dataset = Dataset()
    net = CNN(dataset)
    net.start_test(20300)


if __name__ == "__main__":
    main()
