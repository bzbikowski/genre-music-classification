from src.dataset import Dataset
from src.cnn import CNN


def main():
    dataset = Dataset()
    net = CNN(dataset)
    net.start_train()


if __name__ == "__main__":
    main()
