"""
Code for Tutorial on Parameter Estimation by Maximum Likelihood Fitting
Modified by Kathleen Eckert from an activity written by Sheila Kannappan
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy.random as npr

alphatrue=2. # slope
betatrue=5.  # intercept
errs=2.5 # sigma

narr=100
xvals = np.arange(narr) + 1.
yvals = alphatrue*xvals + betatrue + npr.normal(0,errs,narr)

"""
npr.normal draws samples from a normal Gaussian distribution, which means
here it emulates taking normalized data of a Gaussian nature.

The assumption made that is key to the least squares approach is
that all of the measurement uncetainties (sigma) in a data set are 
the same (i.e. follow the same Gaussian distribution)
"""

plt.figure(1) 
plt.clf()
plt.plot(xvals,yvals,'b*',markersize=10)
plt.xlabel("x-values")
plt.ylabel("y-values")

alphaest=(np.mean(xvals)*np.mean(yvals)-np.mean(xvals*yvals)) / \
   (np.mean(xvals)**2 -np.mean(xvals**2))
betaest= np.mean(yvals-alphaest*(np.mean(xvals)))

"""
We must use alphaest rather than alphatrue in the above formula because
your beta (slope) estimates are affected by the alpha values, so you
should definitely use the estimated alpha when calculating estimated beta
"""

print("analytical MLE slope = %0.7f" %alphaest)
print("analytical MLE y-intercept = %0.7f" %betaest)

yfitvals=xvals*alphaest+betaest
plt.plot(xvals,yfitvals,'r')

alphaunc = np.sqrt(np.sum((yvals - (alphaest*xvals+betaest))**2) / ((narr-2.)*(np.sum((xvals-np.mean(xvals))**2))))
betaunc = np.sqrt((np.sum((yvals - (alphaest*xvals+betaest))**2) / (narr-2.)) * ((1./narr) + (np.mean(xvals)**2)/np.sum((xvals-np.mean(xvals))**2)))

print("analytical MLE uncertainty on alpha is %0.7f" % (alphaunc))
print("analytical MLE uncertainty on beta is %0.7f" % (betaunc))
print("fractional uncertainty on alpha is %0.7f" % (alphaunc/alphaest))
print("fractional uncertainty on beta is %0.7f" % (betaunc/betaest))

"""
The beta parameter has the larger fractional uncertainty
"""

pfit1 = np.polyfit(xvals, yvals, 1)

print("               ")
print("np.polyfit MLE slope = %0.7f" %pfit1[0])
print("np.polyfit MLE y-intercept = %0.7f" %pfit1[1])

"""
You do get the same result as in the analytical case 
(roughly 2.1 for slope and roughly 4.6 for y-intercept)
"""

pfit, covp = np.polyfit(xvals, yvals, 1, cov='True')
print("slope is %0.7f +- %0.7f" % (pfit[0], np.sqrt(covp[0,0])))
print("intercept is %0.7f +- %0.7f" % (pfit[1], np.sqrt(covp[1,1])))

pfit_unc_slope = np.sqrt(covp[0,0])
pfit_unc_yint = np.sqrt(covp[1,1])
perc_diff_slope = ((pfit_unc_slope/alphaunc)-1)
perc_diff_yint = ((pfit_unc_yint/betaunc)-1)
print('Percent Difference between Analytical and Numerical Methods (Slope):')
print(perc_diff_slope)
print('Percent Difference between Analytical and Numerical Methods (Y-int):')
print(perc_diff_yint)

"""
The uncertainties are very close to the same as the analytical solutions
(i.e. .0256 to .0262 for alpha and .7508 to .767 for beta)

As you increase the number of points the uncertainties become smaller
As you decrease the number of points the uncertainties become larger

The percentage difference between the analytical and numerical methods decreases
as we increase the number of data points and increases as we decrease the number
of data points. Obviously slightly subjective to the randomized data.
"""