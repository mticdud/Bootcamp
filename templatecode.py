"""
This is a template code for ASTR 503/703. It is intended to illustrate
standard imports and header information, while providing practice in
debugging, speed optimization, and spotting bad habits in programming.

This code runs but is deeply flawed. Perform the four tasks below to
learn something about both programming and the Central Limit Theorem.

Author: Sheila Kannappan
Created: August 2016
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def gaussfunc(xvals, mean, sigma):
    y = np.exp(-1.*(((xvals-mean)**2) / (2.* sigma**2)))
    norm = 1./np.sqrt(2. * sigma**2 * np.pi)
    y = norm * y
    return y

def poissonfunc(xvals, mean):
    prob = stats.poisson.pmf(xvals, mean)
    return prob

U = 8.
Nct = np.array([6, 36, 216, 1296])
nhr = Nct/U

def main():
    for i in xrange(0, len(Nct)):
        mean = Nct[i]
        maxval = 2*mean
        xvals=np.arange(0, maxval)
        prob = poissonfunc(xvals, mean)
        plt.plot(xvals, prob, 'r', lw=3)
        plt.xlabel("count value")
        plt.ylabel("probability")
        plt.xscale("log")
        sel = np.where(prob == max(prob))
        n = (xvals[sel])[0]
        probval = (prob[sel])[0]
        label = "count for %s hr" % (nhr[i])
        plt.text(n, probval, label)
        sigma=np.sqrt(mean)
        y = gaussfunc(xvals, mean, sigma)
        plt.plot(xvals, y, 'b')
main()