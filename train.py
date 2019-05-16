from src import Dataset, CNN


def main():
    dataset = Dataset()
    net = CNN(dataset)
    net.start_train()


if __name__ == "__main__":
    main()
