import numpy as np
import matplotlib.pyplot as plt


def plotSinusoid(k, d, n):
    i = np.arange(0, d/2)
    denominator = np.power(n, 2*i/d)
    sin = np.sin(k/denominator)
    plt.plot(i, sin)
    return


def main():
    fig = plt.figure(figsize=(15,15))
    for j in range(4):
        for i in range(4):
            plt.subplot(4, 4, j*4+i+1)
            plotSinusoid(k=i * 4, d=(j+1)*1024, n=(j+1)*1000)
    plt.show()
    return

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
