#!/usr/bin/env python3

'''
This python program reads in all the source character data, determining the
length of each character and plots them in a histogram.
This is used to help determine the bucket sizes to be used in the main program.
'''

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
from utils import read_pot_in_directory

num_bins = 50

def main():
    lengths = []
    count = 0
    for strokes, tagcode in read_pot_in_directory('OLHWDB1.1tst_pot/'):
        lengths.append(sum([len(stroke) for stroke in strokes]))
        count += 1

        if count % 1000 == 0:
            print("processed {} samples".format(count))
        if count > 50000:
            break

    #mu = np.mean(lengths)
    #sigma = np.std(lengths)
    mu = np.std(lengths)
    sigma = np.mean(lengths)
    x = np.array(lengths)
    n, bins, patches = plt.hist(x,  num_bins, facecolor='green', alpha=0.5)
    y = mlab.normpdf(bins, mu, sigma)
    plt.plot(bins, y, 'r--')
    plt.title("Frequency of Sequence Lengths")
    plt.xlabel("Length")
    plt.ylabel("Number of Sequences")
    plt.xlim(0,300)
    plt.show()

if __name__=="__main__":
    main()
