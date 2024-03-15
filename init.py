import numpy as np
import pandas as pd

from network import gradientDescend
from plot import plotDigit


### import and convert data
data  =  pd.read_csv('./mnist_train.csv')
data = np.array(data)

## randomize and slice the data
np.random.shuffle(data)
data = data[0:40000]
print(np.shape(data))

### some data parsing
amtData, amtPixel = np.shape(data)
label = data.T[0, :]
pixel = data.T[1:amtData, :]




####### SPECIFY SETTINGS #######
#learing rate
rate = 0.0001
#itereations
iterations = 1
#plot
plot = False
#index element to be plotted
element = 0


estimated = gradientDescend(iterations, rate, label, pixel, amtData, amtPixel)

if(plot):
    plotDigit(pixel, label, estimated, element)