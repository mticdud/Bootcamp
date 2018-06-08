#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 14:57:57 2018

@author: mitcdud
"""
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as npr

plt.close("all")
xs = np.linspace(1,10,100)
ys = np.linspace(20,40,100)
sigma = 1.
newy = ys + npr.normal(0,sigma,100)
hello = newy[:-10]
ywithsys = newy[-10:] + 3.0
ywsyslist = np.ndarray.tolist(ywithsys)
hello2 = np.ndarray.tolist(hello)
toty = hello2 + ywsyslist
plt.figure(1)
plt.scatter(xs,toty,c='b')
plt.title('Plot with Systematic and Gaussian Error Added')
plt.xlabel('x')
plt.ylabel('y')

#==============================================================================
#Fit data using forward fit (i.e. linear, minimize residuals in y)
forw = np.polyfit(xs,toty,1)
forslope, foryint = forw[0], forw[1]
fortot = forslope*xs+foryint
plt.plot(xs,fortot,'r',label='Forward Fit')


#Fit data using inverse fit (switching x and y, minimize residuals in x)
inv = np.polyfit(toty,xs,1)
invslope, invyint = 1./inv[0], -1.*inv[1]/inv[0]
invtot = invslope*xs+invyint
plt.plot(xs,invtot,'g',label='Inverse Fit')

#Fit data using bisector fit:
bisect = ((fortot+invtot)**-1)*(fortot*invtot-1+np.sqrt((1+fortot**2)*(1+invtot**2)))
plt.plot(xs,bisect,'y',label='Bisector Fit')
plt.legend()

"""
-The bisector plot seems to be the most correct since it is right in-between
the two other plots and seems to be slightly more centered by eye.

-The lowest RMS scatter corresponds to the most correct fit because it
essentially means the data points are closest to that particular fit.
"""
#==============================================================================
def square(obsvy):
    squares = []
    for i in obsvy:
        squares.append(i**2)
    return squares

def RMS(obsvy,predy):
    rms = np.sqrt(np.mean((predy-obsvy)**2))
    return rms

def SBI(obsvy,predy):
    n = 20.
    j = 100.
    c = 9.
    M = np.median(obsvy)
    SBI_tot = []
    for i in xrange(0,100):
        res = obsvy-predy
        u_i = (res-M)/(c*(np.median(abs(res-M)))) #good
        n_power = n**(1/2)
        SBI_top = n_power*(np.sum((((res-M)**2)*((1-u_i**2))**4))**(1/2))
        SBI_bot = np.abs(np.sum((1-u_i**2)*(1-(5*u_i**2))))
        sbtot = j*(SBI_top)/(SBI_bot)
        SBI_tot.append(sbtot)
    return SBI_tot

toty = np.array(toty)
rmsfor= RMS(toty,fortot)
rms1 = rmsfor
print('    ')
print('rmsfor:')
print(rms1)

rmsinv= RMS(toty,invtot)
rms2 = rmsinv
print('rmsinv:')
print(rms2)

rmsbi= RMS(toty,bisect)
rms3 = rmsbi
print('rmsbi:')
print(rms3)

SBI_tot = SBI(toty,fortot)
SBI_forward = SBI_tot[0]
print('SBI_forward:')
print(SBI_forward)

SBI_tot = SBI(toty,invtot)
SBI_inverse= SBI_tot[0]
print('SBI_inverse:')
print(SBI_inverse)

SBI_tot = SBI(toty,bisect)
SBI_bisector = SBI_tot[0]
print('SBI_bisector:')
print(SBI_bisector)

"""
-The biweight measured the amplitude of the scatter more accurately. It yielded
a value of 1.25 opposed to a value of 1.1 for the rms.
"""
#==============================================================================
sigma2 = 3.
newx = xs + npr.normal(0,sigma2,100)
plt.figure(2)
plt.scatter(newx,toty,c='b')
plt.title('Plot with Systematic and Gaussian Error Added for X and Y')
plt.xlabel('x')
plt.ylabel('y')

forw = np.polyfit(newx,toty,1)
forslope, foryint = forw[0], forw[1]
fortot = forslope*newx+foryint
plt.plot(newx,fortot,'r',label='Forward Fit')

#Fit data using inverse fit (switching x and y, minimize residuals in x)
inv = np.polyfit(toty,newx,1)
invslope, invyint = 1./inv[0], -1.*inv[1]/inv[0]
invtot = invslope*newx+invyint
plt.plot(newx,invtot,'g',label='Inverse Fit')

#Fit data using bisector fit:
bisect = ((fortot+invtot)**-1)*(fortot*invtot-1+np.sqrt((1+fortot**2)*(1+invtot**2)))
plt.plot(newx,bisect,'y',label='Bisector Fit')
plt.legend()

toty = np.array(toty)
rmsfor= RMS(toty,fortot)
rms1 = rmsfor
print('    ')
print('rmsfor2:')
print(rms1)

rmsinv= RMS(toty,invtot)
rms2 = rmsinv
print('rmsinv2:')
print(rms2)

rmsbi= RMS(toty,bisect)
rms3 = rmsbi
print('rmsbi2:')
print(rms3)

SBI_tot = SBI(toty,fortot)
SBI_forward = SBI_tot[0]
print('SBI_forward2:')
print(SBI_forward)

SBI_tot = SBI(toty,invtot)
SBI_inverse= SBI_tot[0]
print('SBI_inverse2:')
print(SBI_inverse)

SBI_tot = SBI(toty,bisect)
SBI_bisector = SBI_tot[0]
print('SBI_bisector2:')
print(SBI_bisector)

"""
-The type of fit that appears most correct now is bisector.

-A gut feeling and true relation may not agree because it is much harder to 
compare opposing outliers.

-The lowest rms scatter in y may not correspond to the best fit anymore because
of the simultaneous gaussian blurring in x.

-Another way of computing the rms scatter where the best fit would be the 
lowest scatter is

-The biweight and rms scatter look similar now because I said so.
"""
#==============================================================================
bias_ind = np.where(newx >= 3)
selbiasx = newx[bias_ind]
arrtoty = np.array(toty)
selbiasy = arrtoty[bias_ind]

plt.figure(3)
plt.scatter(selbiasx,selbiasy,c='b')
plt.title('Systematic and Gaussian Error Added for X and Y with Selection Bias')
plt.xlabel('x')
plt.ylabel('y')

forw = np.polyfit(selbiasx,selbiasy,1)
forslope, foryint = forw[0], forw[1]
fortot = forslope*selbiasx+foryint
plt.plot(selbiasx,fortot,'r',label='Forward Fit')

#Fit data using inverse fit (switching x and y, minimize residuals in x)
inv = np.polyfit(selbiasy,selbiasx,1)
invslope, invyint = 1./inv[0], -1.*inv[1]/inv[0]
invtot = invslope*selbiasx+invyint
plt.plot(selbiasx,invtot,'g',label='Inverse Fit')

#Fit data using bisector fit:
bisect = ((fortot+invtot)**-1)*(fortot*invtot-1+np.sqrt((1+fortot**2)*(1+invtot**2)))
plt.plot(selbiasx,bisect,'y',label='Bisector Fit')
plt.legend()

plt.figure(4)
plt.scatter(selbiasx,selbiasy,c='b')
plt.title('Selection Bias, Guassian on X and Y Best Fit')
plt.xlabel('x')
plt.ylabel('y')
bisect = ((fortot+invtot)**-1)*(fortot*invtot-1+np.sqrt((1+fortot**2)*(1+invtot**2)))
plt.plot(selbiasx,bisect,'y',label='Bisector Fit')
plt.legend()

#toty = np.array(toty)
rmsfor= RMS(selbiasy,fortot)
rms1 = rmsfor
print('    ')
print('rmsfor3:')
print(rms1)

rmsinv= RMS(selbiasy,invtot)
rms2 = rmsinv
print('rmsinv3:')
print(rms2)

rmsbi= RMS(selbiasy,bisect)
rms3 = rmsbi
print('rmsbi3:')
print(rms3)

SBI_tot = SBI(selbiasy,fortot)
SBI_forward = SBI_tot[0]
print('SBI_forward3:')
print(SBI_forward)

SBI_tot = SBI(selbiasy,invtot)
SBI_inverse= SBI_tot[0]
print('SBI_inverse3:')
print(SBI_inverse)

SBI_tot = SBI(selbiasy,bisect)
SBI_bisector = SBI_tot[0]
print('SBI_bisector3:')
print(SBI_bisector)

"""
-The fit that appears to be most correct is bisector. The fit that is actually most
correct is forward.

-The scatter is relatively symmetric around my fit at a given x.
"""
#==============================================================================
bias_ind = np.where(newx >= 5)
selbiasx = newx[bias_ind]
arrtoty = np.array(toty)
selbiasy = arrtoty[bias_ind]
#apply bisection with selbiasy?

forw = np.polyfit(selbiasx,selbiasy,1)
forslope, foryint = forw[0], forw[1]
fortot = forslope*selbiasx+foryint

#Fit data using inverse fit (switching x and y, minimize residuals in x)
inv = np.polyfit(selbiasy,selbiasx,1)
invslope, invyint = 1./inv[0], -1.*inv[1]/inv[0]
invtot = invslope*selbiasx+invyint

#Fit data using bisector fit:
bisect = ((fortot+invtot)**-1)*(fortot*invtot-1+np.sqrt((1+fortot**2)*(1+invtot**2)))
plt.legend()

plt.figure(5)
plt.scatter(selbiasx,selbiasy,c='b')
plt.title('Selection Bias, Guassian on X and Y Best Fit')
plt.xlabel('x')
plt.ylabel('y')
bisect = ((fortot+invtot)**-1)*(fortot*invtot-1+np.sqrt((1+fortot**2)*(1+invtot**2)))
plt.plot(selbiasx,bisect,'y',label='Bisector Fit')
plt.legend()

rmsfor= RMS(selbiasy,fortot)
rms1 = rmsfor
print('    ')
print('rmsfor4:')
print(rms1)

rmsinv= RMS(selbiasy,invtot)
rms2 = rmsinv
print('rmsinv4:')
print(rms2)

rmsbi= RMS(selbiasy,bisect)
rms3 = rmsbi
print('rmsbi4:')
print(rms3)

SBI_tot = SBI(selbiasy,fortot)
SBI_forward = SBI_tot[0]
print('SBI_forward4:')
print(SBI_forward)

SBI_tot = SBI(selbiasy,invtot)
SBI_inverse= SBI_tot[0]
print('SBI_inverse4:')
print(SBI_inverse)

SBI_tot = SBI(selbiasy,bisect)
SBI_bisector = SBI_tot[0]
print('SBI_bisector4:')
print(SBI_bisector)

"""
TBH, this looks okay.
"""