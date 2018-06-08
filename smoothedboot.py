#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 14:17:30 2018

@author: mitcdud
"""
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as npr
from astroML.resample import bootstrap
from bootstrap_comp import smoothedbootstrap
from bootstrap_comp import smoothedbootstrap2

#==============================================================================
#Unsmoothed Section
plt.close('all')
sigma = 1
numdat = 5
mean = 0
rndsamp = npr.normal(mean,sigma,numdat)
dev = np.std(rndsamp)
print('Direct Method:')
print(dev)

boot1 = bootstrap(rndsamp, 5,  np.std, kwargs=dict(axis=1, ddof=1))
bootsort = sorted(boot1)
bootmed = np.median(bootsort)
print('Bootstrap Method:')
print(bootmed)

boot2 = smoothedbootstrap(rndsamp, numdat, 0, np.std, kwargs=dict(axis=1, ddof=1))
bootsort2 = sorted(boot2)
bootmed2 = np.median(bootsort2)
print('Smoothed Bootstrap Method:')
print(bootmed2)

runs = 1000
runsboot = 2000
dirdev = np.zeros(runs)
for i in range(runs):
    rndsamp2 = npr.normal(mean,sigma,numdat)
    boot2 = bootstrap(rndsamp2, runsboot,  np.std, kwargs=dict(axis=1, ddof=1))
    bootsort2 = sorted(boot2)
    dirdev[i] = np.median(bootsort2)
    mean1 = np.zeros(1)
    if i == 1:
        plt.figure(1)
        plt.title('Regular Bootstrap Results')
        plt.xlabel('Sigma Value')
        plt.ylabel('Counts')
        plt.hist(bootsort2, bins = 100, histtype = 'step', label = 'One Cycle')
        print('          ')
        print('Single Bootstrap Mean:')
        print(np.mean(bootsort2))
        mean1 = (1-np.mean(bootsort2))
        
plt.hist(dirdev, bins = 40, histtype = 'step', color = 'r', label = '2000 Runs')
print('Total Bootstrap Mean:')
print(np.mean(dirdev))
plt.legend()

#==============================================================================
#Smoothed Section
dirdev2 = np.zeros(runs)
for i in range(runs):
    rndsamp2 = npr.normal(mean,sigma,numdat)
    boot2 = smoothedbootstrap(rndsamp2, runsboot, 0, np.std, kwargs=dict(axis=1, ddof=1))
    bootsort2 = sorted(boot2)
    dirdev2[i] = np.median(bootsort2)
    if i == 1:
        plt.figure(2)
        plt.title('Smoothed Bootstrap Results')
        plt.xlabel('Sigma Value')
        plt.ylabel('Counts')
        plt.hist(bootsort2, bins = 100, histtype = 'step', label = 'One Cycle')
        print('          ')
        print('Single Smoothed Bootstrap Mean:')
        print(np.mean(bootsort2))

plt.hist(dirdev2, bins = 40, histtype = 'step', color = 'r', label = '2000 Runs')
print('Total Smoothed Bootstrap Mean:')
print(np.mean(dirdev2))
plt.legend()

#==============================================================================
#Ratio Calculation
SBM = np.abs(1-np.mean(mean1))
TBM = np.abs(1-np.mean(dirdev))
SBSM = np.abs(1-np.mean(bootsort2))
TSBSM = np.abs(1-np.mean(dirdev2))

#==============================================================================
#Comparing to paramfit1.py results
alphatrue=2.
betatrue=5.
errs=2.5
narr=100
newsig1,newsig2,newsig3=.01,.5,.1
ndat = 2

xvals = np.arange(narr) + 1.
sub = npr.normal(newsig3,newsig1,1)
yvals = alphatrue*xvals + betatrue + npr.normal(0,errs,narr)
alphaest=(np.mean(xvals)*np.mean(yvals)-np.mean(xvals*yvals)) / \
   (np.mean(xvals)**2 -np.mean(xvals**2))
betaest= np.mean(yvals-alphaest*(np.mean(xvals)))
yfitvals=xvals*alphaest+betaest
alphaunc = np.sqrt(np.sum((yvals - (alphaest*xvals+betaest))**2) / ((narr-2.)*(np.sum((xvals-np.mean(xvals))**2))))
betaunc = np.sqrt((np.sum((yvals - (alphaest*xvals+betaest))**2) / (narr-2.)) * ((1./narr) + (np.mean(xvals)**2)/np.sum((xvals-np.mean(xvals))**2)))
print('               ')
print("Analytical MLE slope is: %0.7f +- %0.7f" % (alphaest,alphaunc))
print("Analytical MLE y-intercept is: %0.7f +- %0.7f" %(betaest,betaunc))

pfit, covp = np.polyfit(xvals, yvals, 1, cov='True')
print('               ')
print("Slope by Hessian is: %0.7f +- %0.7f" % (pfit[0], np.sqrt(covp[0,0])))
print("Intercept by Hessian is: %0.7f +- %0.7f" % (pfit[1], np.sqrt(covp[1,1])))

dirdev4 = np.zeros(runs)
dirdev3 = np.zeros(runs)
for i in range(runs):
    alphaest_boot = (np.mean(xvals)*np.mean(yvals)-np.mean(xvals*yvals)) / \
        (np.mean(xvals)**2. -np.mean(xvals**2.))
    betaest_boot = np.mean(yvals-alphaest*(np.mean(xvals)))
    alphaunc_boot = np.sqrt(np.sum((yvals - (alphaest_boot*xvals+betaest_boot))**2) / ((narr-2.)*(np.sum((xvals-np.mean(xvals))**2))))
    rndsampalpha = npr.normal(alphaunc_boot,newsig1,ndat)
    betaunc_boot = np.sqrt((np.sum((yvals - (alphaest_boot*xvals+betaest_boot))**2) / (narr-2.)) * ((1./narr) + (np.mean(xvals)**2)/np.sum((xvals-np.mean(xvals))**2)))
    rndsampbeta = npr.normal(betaunc_boot,newsig2,ndat)
    boot3 = smoothedbootstrap2(rndsampalpha, runsboot, 0, np.std, kwargs=dict(axis=1, ddof=1))
    bootsort3 = sorted(boot3)
    dirdev3[i] = np.median(bootsort3)
    boot4 = smoothedbootstrap2(rndsampbeta, runsboot, 0, np.std, kwargs=dict(axis=1, ddof=1))
    bootsort4 = sorted(boot4)
    dirdev4[i] = np.median(bootsort4)

alpha_boot = alphaest_boot-sub
beta_boot = betaest_boot-sub
unc_slope_boot = np.mean(dirdev3)
unc_int_boot = np.mean(dirdev4)
print('               ')
print("Uncertainty in slope by Bootstrap is: %0.7f +- %0.7f" % (alpha_boot, unc_slope_boot))
print("Uncertainty in intercept by Bootstrap is: %0.7f +- %0.7f" % (beta_boot, unc_int_boot))
    
    
    