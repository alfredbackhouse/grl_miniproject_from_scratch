import matplotlib.pyplot as plt

def plot_metrics(metrics):
    plt.plot(metrics['loss'], label='Loss')
    plt.legend()
    plt.show()