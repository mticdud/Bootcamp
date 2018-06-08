import numpy.random as npr
import numpy as np
import matplotlib.pyplot as plt
import random

nran = 1000
xvals = npr.random(nran)
uniform = xvals
radius = uniform
radvals = np.sqrt(radius)

n1, bins1, patches1 = plt.hist(radvals,bins=50,normed=1,histtype='stepfilled')
plt.setp(patches1,'facecolor','g','alpha',0.75)

tries=100000
hits = 0
throws = 0
good_radius = []
for i in range (0, tries):
    throws += 1
    x = random.uniform(-1,1)
    y = random.uniform(-1,1)
    distsquared = x**2 + y**2
    if distsquared <= 1.0:
        hits = hits + 1.0
        good_radius.append(np.sqrt(distsquared))

n1, bins1, patches1 = plt.hist(good_radius,bins=50,normed=1,histtype='stepfilled')
plt.setp(patches1,'facecolor','r','alpha',0.75)

"""
How the histograms compare:
The hits within the radius (good_radius) are roughly 70% of the original radii
created and plotted with the first histogram. The more times you run the full
code and overplot the histograms the more this 70% becomes apparent.
Also of note is that the good_radius histogram is much more consistent
(i.e. there is a more stable slope than the jumpy original radius histogram).
"""
