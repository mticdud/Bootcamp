"""
Code for Tutorial on Bayesian Parameter Estimation
Modified by Kathleen Eckert from an activity written by Sheila Kannappan
June 2015
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy.random as npr

plt.close("all")
alphatrue=2.
betatrue=5. 
errs=2.5 

narr=10
xvals = np.arange(narr) + 1.
yvals = alphatrue*xvals + betatrue+ npr.normal(0,errs,narr)

plt.figure(1)
plt.clf()
plt.plot(xvals,yvals,'b*',markersize=10)
plt.xlabel('xvalues')
plt.ylabel('yvalues')

gridsize1=1000
gridsize2=100
alphaposs=np.arange(gridsize1) / 100.
betaposs=np.arange(gridsize2) / 10.

"""
-The values being considered are: slope 0:10 and y-int 0:9.9
    
-The priors are zero if not from 0 to 10 and are flat between 0-10.
"""

print("min slope is %f and max slope is %f" % (np.min(alphaposs), np.max(alphaposs)))
print("min y-intercept is %f and max y-intercept is %f" % (np.min(betaposs), np.max(betaposs)))

xx=np.arange(0,1,0.1)

plt.figure(2) 
plt.clf()
plt.subplot(121)
for i in range(len(betaposs)):
    plt.plot(xx,xx+betaposs[i],'b-')

plt.xlim(0,1)
plt.ylim(0,1)
plt.xlabel("x values")
plt.ylabel("y values for several values of y-intercept (y=x+beta)")
plt.title("test y-intercept prior")

plt.subplot(122)
for i in range(len(alphaposs)):
    plt.plot(xx,xx*alphaposs[i])

plt.xlim(0,1) 
plt.ylim(0,0.2)
plt.xlabel("x values")
plt.ylabel("y values for several values of slope (y=alpha*x)")
plt.title("test slope prior")

"""
-The model does evenly space sample intercept values.

-Slope values are on a weighted scale with steeper slopes being closer together.

-For a physical system, anything dealing with angles should proabably have a uniform prior in angle
and anything angle independent you would want a uniform prior for slope.
"""

prioronintercept_flat = 1.
prioronslope_flat = 1.
prioronslope_uninformative = (1.+alphaposs**2)**(-3./2)

lnpostprob_flat=np.zeros((gridsize1,gridsize2))
lnpostprob_comp=np.zeros((gridsize1,gridsize2))

for i in xrange(gridsize1):
    for j in xrange(gridsize2):
        modelvals = alphaposs[i]*xvals+betaposs[j]
        resids = (yvals - modelvals)
        chisq = np.sum(resids**2 / errs**2)
        priorval_flat = prioronintercept_flat * prioronslope_flat
        priorval_comp = prioronslope_flat * prioronslope_uninformative[i]
        lnpostprob_flat[i,j] = (-1./2.)*chisq + np.log(priorval_flat)      
        lnpostprob_comp[i,j] = (-1./2.)*chisq + np.log(priorval_comp)

postprob_flat=np.exp(lnpostprob_flat)
postprob_comp=np.exp(lnpostprob_comp)

marginalizedpprob_flat_slope = np.sum(postprob_flat,axis=1) / np.sum(postprob_flat)
marginalizedpprob_comp_slope = np.sum(postprob_comp,axis=1) / np.sum(postprob_comp)

plt.figure(3) 
plt.clf()
plt.plot(alphaposs,marginalizedpprob_flat_slope,'g.',markersize=10)
plt.plot(alphaposs,marginalizedpprob_comp_slope,'r.',markersize=10)
plt.xlabel("alpha")
plt.ylabel("marginalized posterior distribution of slope")

"""
-There is a differnce between the marginalized posterior distributions of the slope.
Depending on which direction you're reading the plot from you could say one\
either lags behind or is ahead of the other (i.e. green and red are slightly separate).

-The marginalized posterior distributions of the slope compare similarly 
with the MLE values. They are within .01 of each other (around .025)

-I estimate the uncertainty on the slope value as .025

-The uncertainty on the slope compares similarly with the MLE uncertainty value.

-The MPD for slope and intercept change based on the nuber of data points picked.
For N = 10 the distribution becomes broader i.e. larger uncertainties. There is
also a more obvious separation for the slope graph. For N = 100, the peak 
becomes much more narrow.

-If you change the grid spacing, the data points become more sparse and patterns
are harder to see. The separation between the sin waves and the y-int plot are 
far more noticable as well.
"""

marginalizedpprob_flat_yint = np.sum(postprob_flat,axis=0) / np.sum(postprob_flat)
marginalizedpprob_comp_yint = np.sum(postprob_comp,axis=0) / np.sum(postprob_comp)

plt.figure(4)
plt.clf()
plt.plot(betaposs,marginalizedpprob_flat_yint,'g',markersize='10.')
plt.plot(betaposs,marginalizedpprob_comp_yint,'r',markersize='10.')
plt.xlabel("beta")
plt.ylabel("marginalized posterior distribution of y-intercept")

"""
-In this case we do want a flat prior on the slope and intercept. Or we want a
prior that compensates for the unequal distribution in angles?

-Depends on what we're modeling (deals with angles or not)
"""