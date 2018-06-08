#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 11:39:25 2018

@author: mitcdud
"""
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def RMS(predy,obsvy):
    rms = np.sqrt(np.mean((predy-obsvy)**2))
    return rms

#==============================================================================
#Reduced Chi^2 for Deg1 and Deg2 Fits:
errs = 1.85
input = np.load('xydata.npz')
xx = input['xx']
yy = input['yy']
numcomp = np.size(yy)

plt.figure(1)
plt.scatter(xx,yy)
plt.title('Unknown Data with First and Second Degree Fits')
plt.xlabel('x')
plt.ylabel('y')

pfit1, covp1 = np.polyfit(xx, yy, 1, cov='True')
pfit2, covp2 = np.polyfit(xx, yy, 2, cov='True')

poly1 = np.polyval(pfit1,xx)
poly2 = np.polyval(pfit2,xx)
newy1, newy2 = (yy - poly1)**2, (yy - poly2)**2
sizexx = np.size(xx)
deg1 = sizexx - 2.
deg2 = sizexx - 3.
redchisq1 = np.sum(newy1)/(errs**2)/(deg1)
redchisq2 = np.sum(newy2)/(errs**2)/(deg2)
print('Reduced Chi^2 for Degree 1 Fit:')
print(redchisq1)
print('    ')
print('Reduced Chi^2 for Degree 2 Fit:')
print(redchisq2)

#==============================================================================
#Determining RMS Values and Sigma Variations:
xord = sorted(xx)
yord = sorted(yy)
xlimlow, xlimhigh = xord[0], xord[-1]
ylimlow, ylimhigh = yord[0], yord[-1]
xcomps = np.linspace(xlimlow,xlimhigh,100)
besty1 = np.polyval(pfit1,xcomps)
besty2 = np.polyval(pfit2,xcomps)
rms1 = RMS(yy,poly1)
rms2 = RMS(yy,poly2)
plt.plot(xcomps,besty1,label='Degree 1 Fit',color='r')
plt.plot(xcomps,besty2,label='Degree 2 Fit', color='g')
plt.legend()
print('     ')
print('RMS Error for Degree 1 Fit:')
print(rms1)
print('RMS Error for Degree 2 Fit:')
print(rms2)

"""
The reduced chi^2 values are wrong because they are too low. The error assumption
is too high. By making the errors 1.8 instead of 2 the redchi^2 become much closer
to 1 (.97 and .82) for degree 1 and degree 2 fitting (respectively). Based on this,
the first degree fit performs better and is preferred for this data set.
"""

print('   ')
print('Expected 1 Sigma Variation for Degree 1 Fit:')
sig1deg1 = stats.chi2.ppf(0.68, deg1) / (deg1)
print(sig1deg1)
print('Expected 2 Sigma Variation for Degree 1 Fit:')
sig2deg2 = stats.chi2.ppf(0.9545, deg1) / (deg1)
print(sig2deg2)

print('   ')
print('Expected 1 Sigma Variation for Degree 2 Fit:')
sig1deg1 = stats.chi2.ppf(0.68, deg2) / (deg2)
print(sig1deg1)
print('Expected 2 Sigma Variation for Degree 2 Fit:')
sig2deg2 = stats.chi2.ppf(0.9545, deg2) / (deg2)
print(sig2deg2)

"""
The reduced Chi^2 values for deg1 and deg2 fits are further apart, with the
deg1 fit being .97 and the deg2 fit being .82. However the sigma values are 
really close together (within .1 for both). I'd say just based soley on the 
reduced Chi^2 values I'm reasonably confident in that choice of fit. However, 
.97 still is not the best, so there is still room for improvement.
"""
#==============================================================================
#Determining Posterior Distributions and Marginalizing to Obtain Odds:
ndata=sizexx
nalpha, nbeta = 100, 100
alphaposs = np.linspace(pfit1[0]-4.*np.sqrt(covp1[0,0]),pfit1[0]+4.*np.sqrt(covp1[0,0]),nalpha)
betaposs = np.linspace(pfit1[1]-4.*np.sqrt(covp1[1,1]),pfit1[1]+4.*np.sqrt(covp1[1,1]),nbeta)
range_alpha = (8.*np.sqrt(covp1[0,0]))
range_beta = (8.*np.sqrt(covp1[1,1]))
prior_alpha = 1./range_alpha
prior_beta = 1./range_beta
prior_1storder = prior_alpha*prior_beta
modelgridterm1 = alphaposs.reshape(1,nalpha) * xx.reshape(ndata,1)
modelgrid = modelgridterm1.reshape(ndata,nalpha,1) + betaposs.reshape(nbeta,1,1).T
residgrid = yy.reshape(ndata,1,1) - modelgrid
chisqgrid = np.sum(residgrid**2/errs**2,axis=0)        
lnpostprob1 = (-1./2.)*chisqgrid + np.log(prior_1storder) 
ndata=sizexx
np0, np1, np2 = 100, 100, 100
p0poss = np.linspace(pfit2[0]-4.*np.sqrt(covp2[0,0]),pfit2[0]+4.*np.sqrt(covp2[0,0]),np0)
p1poss = np.linspace(pfit2[1]-4.*np.sqrt(covp2[1,1]),pfit2[1]+4.*np.sqrt(covp2[1,1]),np1)
p2poss = np.linspace(pfit2[2]-4.*np.sqrt(covp2[2,2]),pfit2[2]+4.*np.sqrt(covp2[2,2]),np2)
range_p0 = (8.*np.sqrt(covp2[0,0]))
range_p1 = (8.*np.sqrt(covp2[1,1]))
range_p2 = (8.*np.sqrt(covp2[2,2]))
prior_p0 = 1./range_p0
prior_p1 = 1./range_p1
prior_p2 = 1./range_p2
prior_2ndorder = prior_p0*prior_p1*prior_p2
modelgridterm1 = p0poss.reshape(1,np0) * xx.reshape(ndata,1)**2
modelgridterm2 = modelgridterm1.reshape(ndata,np0,1) + (p1poss.reshape(np1,1,1).T * xx.reshape(ndata,1,1))
modelgrid = modelgridterm2.reshape(ndata,np0,np1,1) + p2poss.reshape(np2,1,1,1).T
residgrid = yy.reshape(ndata,1,1,1) - modelgrid
chisqgrid = np.sum(residgrid**2/errs**2,axis=0)        
lnpostprob2 = (-1./2.)*chisqgrid + np.log(prior_2ndorder) 

postprob_deg1 = np.exp(lnpostprob1)
postprob_deg2 = np.exp(lnpostprob2)
marginalizeddeg1 = np.sum(postprob_deg1,axis=1)/np.sum(postprob_deg1)
marginalizeddeg2 = np.sum(postprob_deg2,axis=1) / np.sum(postprob_deg2)
altcomp = np.sum(marginalizeddeg2, axis = 1)

plt.figure(2)
plt.plot(alphaposs,marginalizeddeg1,'g.',markersize=5, label='Deg1 Fit - Alpha')
plt.plot(alphaposs,altcomp,'y.',markersize=5, label='Deg2 Fit - Alpha')
plt.plot(betaposs,marginalizeddeg1,'b.',markersize=5, label='Deg1 Fit - Beta')
plt.plot(betaposs,altcomp,'m.',markersize=5, label='Deg2 Fit - Beta')
plt.xlabel("Alpha and Beta")
plt.ylabel("Marginalized Posterior Distribution of Deg1 and Deg2 Fits")
plt.legend()

"""
I'm pretty sure the Bayesian favors the degree 1 fit over the degree 2 fit,
since the curve has a slightly lower y-max value for possible alpha and beta values.
"""
d_alpha, d_beta = range_alpha/nalpha, range_beta/nbeta
d_p0, d_p1, d_p2 = range_p0/np0, range_p1/np1, range_p2/np2
oddsdeg1_unsum = postprob_deg1*d_alpha*d_beta
oddsdeg2_unsum = postprob_deg2*d_p0*d_p1*d_p2
oddsdeg1 = np.sum(oddsdeg1_unsum)
oddsdeg2 = np.sum(oddsdeg2_unsum)
print('  ')
print('Odds for Deg1 Fit:')
print(oddsdeg1)
print('Odds for Deg2 Fit:')
print(oddsdeg2)
odds = oddsdeg1/oddsdeg2
print('    ')
print "Odds Favoring a 1st Order Over a 2nd Order Model: %0.2f" % odds

"""
Does your result agree with your earlier result based on chi^2 analysis?
Discuss the confidence levels in each analysis. (ch. 5.4)
"""


