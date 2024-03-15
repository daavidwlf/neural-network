from matplotlib import pyplot as plt
import numpy as np

def plotDigit(pixel, labels, estimated, element):
    grid = np.zeros([28, 28])

    k = 0
    for i in range(28):
        for j in range(28):
            grid[i][j]=pixel.T[element][k-1]
            k+=1
    title = "actual Label: ",labels.T[element]
    subtitle = "estiamted Label: ", estimated[element]
    plt.title(title)
    plt.suptitle(subtitle)
    plt.imshow(grid, cmap='gray', vmin=0, vmax=255)
    plt.show()