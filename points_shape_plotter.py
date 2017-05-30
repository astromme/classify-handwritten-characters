#!/usr/bin/env python3

'''
This python program reads in all the source character data, determining the
length of each character and plots them in a histogram.
This is used to help determine the bucket sizes to be used in the main program.
'''

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
import sys
from utils import read_pot_in_directory

num_bins = 50

def main():
    h_w_ratios = []
    count = 0
    for strokes, tagcode in read_pot_in_directory('OLHWDB1.1tst_pot/'):
        xmin = sys.maxsize
        ymin = sys.maxsize

        xmax = -sys.maxsize
        ymax = -sys.maxsize

        for stroke in strokes:
            for point in stroke:
                if len(point) < 2:
                    continue

                x, y = point
                xmin = min(xmin, x)
                ymin = min(ymin, y)

                xmax = max(xmax, x)
                ymax = max(ymax, y)

        h_w_ratios.append((ymax-ymin) / (xmax-xmin))
        count += 1

        if count % 1000 == 0:
            print("processed {} samples".format(count))
        if count > 50000:
            break


    mu = np.std(h_w_ratios)
    sigma = np.mean(h_w_ratios)
    x = np.array(h_w_ratios)
    n, bins, patches = plt.hist(x,  num_bins, facecolor='green', alpha=0.5)
    y = mlab.normpdf(bins, mu, sigma)
    plt.plot(bins, y, 'r--')
    plt.title("Height to Width Ratios of Sequences")
    plt.xlabel("Height to Width Ratio")
    plt.ylabel("Number of Sequences")
    plt.xlim(0,5)
    plt.show()

if __name__=="__main__":
    main()
