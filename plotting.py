import matplotlib.pyplot as plt


def generate_plot(data):
    """
    Plot data on y axis over epochs on x axis
    :param data: list[int]
    :return:
    """
    epochs = list(range(1, len(data) + 1))
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.plot(epochs, data)
    plt.savefig("images/loss.png")
    plt.show()
