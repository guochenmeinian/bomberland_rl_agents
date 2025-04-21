import matplotlib.pyplot as plt

def plot_curve(data, ylabel="Reward", title="Training Progress", save_path="plot.png"):
    plt.figure()
    plt.plot(data)
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    plt.savefig(save_path)
    plt.close()
