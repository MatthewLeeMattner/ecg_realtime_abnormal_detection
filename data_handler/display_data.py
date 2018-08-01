'''
ecg_realtime_abnormal_detection
Created 1/08/18 by Matthew Lee
'''
from format_data import get_data
import config
import matplotlib.pyplot as plt
import numpy as np
from time import sleep


def plot_feature(X, y):
    label = config.data['annotations'][np.argmax(y)]
    plt.plot([x[0] for x in X], label="Difference")
    plt.plot([x[1] for x in X], label="Average Difference")
    plt.title(label)
    plt.legend(bbox_to_anchor=(1, 1),
               bbox_transform=plt.gcf().transFigure)
    plt.show()



if __name__ == "__main__":
    index = 1500

    X, y = get_data()
    plot_feature(X[index], y[index])
