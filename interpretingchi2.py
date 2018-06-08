"""
Interpreting Chi^2
Author: Sheila Kannappan
excerpted/adapted from CAP REU tutorial September 2016
"""
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as npr

i=0
#if i == 0:
for i in xrange(2):

  narr=300*(1+9*i) #30 first time, 300 second time
  chisqs=[]
  iters=1000  # experiment with rerunning the plots many times
             # to see how the random variations change the chi^2
             # start w/ iters=100 then change iters to 1000 and repeat

  for j in xrange(iters):
    xvals = np.zeros(narr)
    yvals = np.zeros(narr)
    xvals = np.arange(narr)/(1.+9.*i) + 1.
    errs=0.1
    yvals = 1./xvals + npr.normal(0,errs,narr)
    
    """
    #npr.normal draws samples from a normal Gaussian distribution
    """

    resids = np.abs(yvals - 1./xvals)
    #residual = observed y-value - predicted y-value
    
    """
    We are subtracting 1./xvals because we are finding residuals
    of the randomized data
    """

    chisq = np.sum(resids**2 / errs**2)
    chisqs.append(chisq)
    
    """
    chisq is related to N by:
    If we didn't know the errors and overestimated then the chi^2 value
    would be much less than one
    """

  if i==0:
      redchisqdist1 = np.array(chisqs)/narr
      
      """
      Reduced chi^2 is the chi^2 per degree of freedom.
      We are dividing by narr because narr is the degrees of freedom 
      for the system, and when divided out, yields the reduced chi^2 statistic
      """

  if i==1:
      redchisqdist2 = np.array(chisqs)/narr
      
plt.figure(1)
plt.clf()
n1, bins1, patches1 = plt.hist(redchisqdist1,bins=round(0.05*iters),normed=1,histtype='stepfilled')
n2, bins2, patches2 = plt.hist(redchisqdist2,bins=round(0.05*iters),normed=1,histtype='step')
plt.setp(patches1,'facecolor','g','alpha',0.75)
plt.xlim(0,2.5)
plt.setp(patches2,'hatch','///','alpha',0.75,color='blue')

"""
The good plotting strategies being used here are the ability to clearly see
both plots (one is a outline and the lower plot can be seen )
"""

plt.title('comparison of reduced chi-squared distributions')
#plt.title('reduced chi-squared distribution') 

#plt.text(1.4,1,"N=30",size=11,color='g')
plt.text(1.2,3,"N=300",size=11,color='b')

"""
# Q: how can we determine the confidence level associated with
#    a certain deviation of reduced chi^2 from 1?
# A: we have to do the integral under the distribution --
#    e.g. for 3sigma (99.8%) confidence find x such that
#    the integral up to x contains 99.8% of the area
# how can you approximate this using np.argsort?
# make sure to set iters=1000 for this exercise....
"""


inds=np.argsort(redchisqdist1)
dist1whereright = .998*len(inds)
dist1whereleft = (1-.998)*len(inds)
final1r = inds[dist1whereright]
final1l = inds[dist1whereleft]
print('left bound of interval dist1:')
l1bound = redchisqdist1[final1l]
print(l1bound)
print('right bound of interval dist1:')
r1bound = redchisqdist1[final1r]
print(r1bound)

inds2 = np.argsort(redchisqdist2)
dist2whereright = .998*len(inds2)
dist2whereleft = (1-.998)*len(inds2)
final2r = inds2[dist2whereright]
final2l = inds2[dist2whereleft]
print('left bound of interval dist2:')
l2bound = redchisqdist2[final2l]
print(l2bound)
print('right bound of interval dist2:')
r2bound = redchisqdist2[final2r]
print(r2bound)