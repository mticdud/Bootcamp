import numpy.random as npr
import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt

nran = 1000
xvals = npr.random(nran)
intp_g = xvals
lkupuvals = (np.array([range(0,10000)])-5000)/1000.
lkupintvals = 0.5+0.5*erf(lkupuvals/np.sqrt(2.0))
uvals = 0.*intp_g
for i in range(0,nran):
    diffs = abs(lkupintvals-intp_g[i])
    uvals[i] = lkupuvals[np.where(diffs == diffs.min())]

n1, bins1, patches1 = plt.hist(uvals,bins=50,normed=1,histtype='stepfilled')
plt.setp(patches1,'facecolor','g','alpha',0.75)