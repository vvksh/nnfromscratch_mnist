# This is a sample Python script.

def run_mnist_training():
    import network
    import mnist_loader

    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = network.Network([784, 30, 10])
    net.SGD(training_data, 30, 10, 3.0, test_data=test_data)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run_mnist_training()
