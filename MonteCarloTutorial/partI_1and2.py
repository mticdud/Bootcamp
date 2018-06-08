import random
import matplotlib.pyplot as plt
import numpy as np

x = []
y = []
u = []

random.seed(555)
for i in range(0,10):
   x.append(random.random())

random.seed(555)
for i in range(0,10):
   y.append(random.random())

for i in range(1,1000):
   u.append(random.gauss(0,1))

mean,sigma = 0,1
plt.figure(1)
n1, bins1, patches1 = plt.hist(u, bins=50, normed=1)
gaussfunc= np.exp((-1.*bins1**2)/(2.*sigma**2))/(sigma*np.sqrt(2*np.pi))
plt.plot(bins1,gaussfunc)

plt.figure(2)
plt.plot(x,y,'g*')
plt.xlabel('x')
plt.ylabel('y')

uarr = np.array(u)
len_u = len(uarr)
one_sig_nums = np.where((uarr > -1*sigma) & (uarr < sigma))
len_one_sig = np.size(one_sig_nums)
percent_onesig = float(len_one_sig)/float(len_u)
print (percent_onesig)

"""
If you have an array of data with error bars equal to u, how often
should the fit line go through the error bars?

Answer: 68% of the time
"""