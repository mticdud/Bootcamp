"""
Tutorial on Histograms, KDE, and Hypothesis Tests for Comparing Distributions
Author: Sheila Kannappan
June 2017: heavily adapted from the original -- see 
https://github.com/galastrostats/general/blob/master/distributions.md
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy.random as npr
from astroML.plotting import hist
from sklearn.neighbors import KernelDensity

data = np.genfromtxt("ECO_dr1_subset.csv", delimiter=",", dtype=None, names=True)
name = data['NAME']
logmstar = data ['LOGMSTAR']
urcolor = data['MODELU_RCORR']
cz = data['CZ']

goodur = (urcolor > -99) & (logmstar > 10.) #>9
colors=urcolor[goodur]

plt.figure(1)
plt.clf()
hist(colors,bins='freedman',label='freedman',normed=1,histtype='stepfilled',color='green',alpha=0.5)
hist(colors,bins='scott',label='scott',normed=1,histtype='step',color='purple',alpha=0.5,hatch='///')
n0, bins0, patches0 = hist(colors,bins='knuth',label='knuth',normed=1,histtype='stepfilled',color='blue',alpha=0.25)
plt.xlim(0,3)
plt.xlabel("u-r color (mag)")
plt.title("Galaxy Color Distribution")
plt.legend(loc="best")

bw = 0.5*(bins0[2]-bins0[1])
kde = KernelDensity(kernel='epanechnikov',bandwidth=bw).fit(colors[:,np.newaxis])
xx = np.linspace(-2,16,10000)[:,np.newaxis]
logdens = kde.score_samples(xx)
plt.figure(1)
plt.plot(xx,np.exp(logdens),color='green',label='kde')
plt.legend(loc="best")

nearby = (cz[goodur] < 5500.) #>
selenvnear = np.where(nearby)
selenvfar = np.where(~nearby)

plt.figure(2)
plt.clf()
hist(colors[selenvnear],bins='knuth',label='near',normed=1,histtype='stepfilled',color='red',alpha=0.25)
plt.xlim(0,3)
kde = KernelDensity(kernel='epanechnikov',bandwidth=bw).fit(colors[selenvnear][:,np.newaxis])
logdens = kde.score_samples(xx)
plt.plot(xx,np.exp(logdens),'r--')
hist(colors[selenvfar],bins='knuth',label='far',normed=1,histtype='stepfilled',color='blue',alpha=0.25)
kde = KernelDensity(kernel='epanechnikov',bandwidth=bw).fit(colors[selenvfar][:,np.newaxis])
logdens = kde.score_samples(xx)
plt.plot(xx,np.exp(logdens),'b--')

DD, pnullks = stats.ks_2samp(colors[selenvnear],colors[selenvfar])
UU, pnullmw = stats.mannwhitneyu(colors[selenvnear],colors[selenvfar])
print('Kolmogorov-Smirnov (K-S) probability :')
print(1-pnullks)
print('Mann-Whitney U (M-W) probability:')
print(1-pnullmw)
plt.text(1.1, 1.7, "K-S pnull = %0.2g" % pnullks, size=14, color='b')
plt.text(1.1, 1.5, "M-W pnull = %0.2g" % pnullmw, size=14, color='b')
plt.xlabel("u-r color (mag)")
plt.legend()

namegood = name[goodur]

makenew = True
if makenew:
    sample2inds = npr.choice(len(namegood),size=int(round(0.5*len(namegood)-1)),replace=False)
    flag12 = np.zeros(len(namegood),dtype=int)
    flag12[sample2inds] = 1
    flag12 += 1
    #np.savez('samplesplitflag',flag12=flag12)   ???
    
else:
    input = np.load("samplesplitflag.npz")
    flag12 = input['flag12']
    
sample1inds = np.where(flag12 == 1)
sample2inds = np.where(flag12 == 2)

plt.figure(3)
plt.clf()
n, bins, patches = hist(colors[sample1inds],bins='knuth',label='1',histtype='stepfilled',color='red',alpha=0.25)
hist(colors[sample2inds],bins=bins,label='2',histtype='stepfilled',color='blue',alpha=0.25)
plt.xlim(0,3)
DD, pnullks = stats.ks_2samp(colors[sample1inds],colors[sample2inds])
plt.text(1.1, 200, "K-S pnull = %0.2g" % pnullks, size=14, color='b')
plt.xlabel("u-r color (mag)")
plt.legend()

"""
As in Fig. 5.20 (p. 227), Scott's rule makes broader bins.
pnull can jump around because of randomization (changes per run):
Trials:
1     .26
2     .78
3     .94
4     .25
5     .051
    
Knuth: Almost like Freedman, just shifted slightly to the left
Scott: Very similar to an average of Freedman and Knuth, but with a wide peak
Freedman: Almost like the Knuth, just shifted slightly to the right
KDE: Best overall. An average of all but with a smooth curve to show precise peak

Making goodur in a different range (other than >10) emphasizes different peaks
As you go lower, the peak on the left is emphasized over the one on the right.

The near and far regions of ECO
# have different large-scale galaxy densities ("cosmic variance"), affecting 
# galaxy colors. So instead of dividing by distance, let's try dividing randomly.
"""