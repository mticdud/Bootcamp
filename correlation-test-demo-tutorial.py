"""
Correlation Test Demo-Tutorial
Author: Sheila Kannappan
adapted for ASTR 503/703 from CAP REU version September 2016
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pylab
pylab.ion()

data=np.loadtxt("anscombe.txt")
x1=data[:,0]
y1=data[:,1]
x2=data[:,2]
y2=data[:,3]
x3=data[:,4]
y3=data[:,5]
x4=data[:,6]
y4=data[:,7]

plt.figure(1)
plt.clf()

plt.subplot(221)
plt.plot(x1, y1,'g.',markersize=10)
testxvals=np.array([3.,7.,11.,15.,19.])
plt.plot(testxvals,3.+0.5*testxvals,'r',linestyle=':',linewidth=2.)
rms=np.sqrt(np.mean((y1-(3.+0.5*x1))**2))
plt.text(3,12,'rms %0.2f' % rms,size=11,color='b')
plt.xlim(2,20)
plt.ylim(2,14)
plt.title('Standard')
plt.ylabel('y')

plt.subplot(222)
plt.plot(x2, y2,'g.',markersize=10)
plt.plot(testxvals,3.+0.5*testxvals,'r',linestyle=':',linewidth=2.)
rms=np.sqrt(np.mean((y2-(3.+0.5*x2))**2))
plt.text(3,12,'rms %0.2f' % rms,size=11,color='b')
plt.xlim(2,20)
plt.ylim(2,14)
plt.title('Curved')

plt.subplot(223)
plt.plot(x3, y3,'g.',markersize=10)
plt.plot(testxvals,3.+0.5*testxvals,'r',linestyle=':',linewidth=2.)
rms=np.sqrt(np.mean((y3-(3.+0.5*x3))**2))
plt.text(3,12,'rms %0.2f' % rms,size=11,color='b')
plt.xlim(2,20)
plt.ylim(2,14)
plt.title('Outlier')
plt.xlabel('x')
plt.ylabel('y')

plt.subplot(224)
plt.plot(x4, y4,'g.',markersize=10)
plt.plot(testxvals,3.+0.5*testxvals,'r',linestyle=':',linewidth=2.)
rms=np.sqrt(np.mean((y4-(3.+0.5*x3))**2))
plt.text(3,12,'rms %0.2f' % rms,size=11,color='b')
plt.xlim(2,20)
plt.ylim(2,14)
plt.title('Garbage')
plt.xlabel('x')

ax=plt.subplot(221)
plt.setp(ax.get_xticklabels(), visible=False)
ax=plt.subplot(222)
plt.setp(ax.get_xticklabels(), visible=False)
plt.setp(ax.get_yticklabels(), visible=False)
ax=plt.subplot(224)
plt.setp(ax.get_yticklabels(), visible=False)

sigmasym=r'$\sigma$'

#========================================================================

plt.subplot(221)
cc,pnull=stats.spearmanr(x1,y1)
print(" ")
print("Standard:")
print("Spearman rank correlation coefficient %f" % cc)
print("Spearman rank probability of no correlation %f" % pnull)
confidence=stats.norm.interval(1-pnull)
leveltext='Spearman rank %0.1f' % confidence[1]
plt.text(8.5,3,leveltext+sigmasym, size=11, color='b')

cc,pnull=stats.pearsonr(x1,y1)
print("Pearson correlation coefficient %f" % cc)
print("Pearson probability of no correlation %f" % pnull)
confidence=stats.norm.interval(1-pnull)
leveltext='Pearson %0.1f' % confidence[1]
plt.text(8.5,4.5,leveltext+sigmasym, size=11, color='b')

cc, pnull = stats.kendalltau(x1,y1)
print("Kendall correlation coefficient %f" % cc)
print("Kendall probability of no correlation %f" % pnull)
confidence=stats.norm.interval(1-pnull)
leveltext='Kendall %0.1f' % confidence[1]
plt.text(8.5,6,leveltext+sigmasym, size=11, color='b')

#=======================================================================

plt.subplot(222)
cc,pnull=stats.spearmanr(x2,y2)
print(" ")
print("Curved:")
print("Spearman rank correlation coefficient %f" % cc)
print("Spearman rank probability of no correlation %f" % pnull)
confidence=stats.norm.interval(1-pnull)
leveltext='Spearman rank %0.1f' % confidence[1]
plt.text(8.5,3,leveltext+sigmasym, size=11, color='b')

cc,pnull=stats.pearsonr(x2,y2)
print("Pearson correlation coefficient %f" % cc)
print("Pearson probability of no correlation %f" % pnull)
confidence=stats.norm.interval(1-pnull)
leveltext='Pearson %0.1f' % confidence[1]
plt.text(8.5,4.5,leveltext+sigmasym, size=11, color='b')

cc, pnull = stats.kendalltau(x2,y2)
print("Kendall correlation coefficient %f" % cc)
print("Kendall probability of no correlation %f" % pnull)
confidence=stats.norm.interval(1-pnull)
leveltext='Kendall %0.1f' % confidence[1]
plt.text(8.5,6,leveltext+sigmasym, size=11, color='b')

#=======================================================================

plt.subplot(223)
cc,pnull=stats.spearmanr(x3,y3)
print(" ")
print("Outlier:")
print("Spearman rank correlation coefficient %f" % cc)
print("Spearman rank probability of no correlation %f" % pnull)
confidence=stats.norm.interval(1-pnull)
leveltext='Spearman %0.1f' % confidence[1]
plt.text(8.5,3,leveltext+sigmasym, size=11, color='b')

cc,pnull=stats.pearsonr(x3,y3)
print("Pearson correlation coefficient %f" % cc)
print("Pearson probability of no correlation %f" % pnull)
confidence=stats.norm.interval(1-pnull)
leveltext='Pearson %0.1f' % confidence[1]
plt.text(8.5,4.5,leveltext+sigmasym, size=11, color='b')

cc, pnull = stats.kendalltau(x3,y3)
print("Kendall correlation coefficient %f" % cc)
print("Kendall probability of no correlation %f" % pnull)
confidence=stats.norm.interval(1-pnull)
leveltext='Kendall %0.1f' % confidence[1]
plt.text(8.5,6,leveltext+sigmasym, size=11, color='b')

#======================================================================

plt.subplot(224)
cc, pnull = stats.spearmanr(x4,y4)
print(" ")
print("Garbage:")
print("Spearman rank correlation coefficient %f" % cc)
print("Spearman rank probability of no correlation %f" % pnull)
confidence=stats.norm.interval(1-pnull)
leveltext='Spearman %0.1f' % confidence[1]
plt.text(8.5,3,leveltext+sigmasym, size=11, color='b')

cc, pnull = stats.pearsonr(x4,y4)
print("Pearson rank correlation coefficient %f" % cc)
print("Pearson rank probability of no correlation %f" % pnull)
confidence=stats.norm.interval(1-pnull)
leveltext='Pearson %0.1f' % confidence[1]
plt.text(8.5,4.5,leveltext+sigmasym, size=11, color='b')

cc, pnull = stats.kendalltau(x4,y4)
print("Kendall correlation coefficient %f" % cc)
print("Kendall probability of no correlation %f" % pnull)
confidence=stats.norm.interval(1-pnull)
leveltext='Kendall %0.1f' % confidence[1]
plt.text(8.5,6,leveltext+sigmasym, size=11, color='b')