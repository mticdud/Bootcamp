import numpy.random as npr
import numpy as np

sigma=1.
throws=10000
xvals=(npr.random(throws) * 2.*sigma - 1.*sigma)
yvals=(npr.random(throws) / (sigma*np.sqrt(2.*3.14159)))
gaussfunct= np.exp((-1.*xvals**2)/(2.*sigma**2))/(sigma*np.sqrt(2.*3.14159))

hits=np.size(np.where(yvals <= gaussfunct))

rectarea = 2.* sigma / (sigma*np.sqrt(2.*3.14159))
area = (float(hits)/float(throws))*rectarea
print("area is %s" % area)

"""
Why is this area equal to the percentage of u values between +_ one sigma 
even though u was created with random.gauss?

Answer: Most likely because everything has been normalized, and one sigma of a
normalized gaussian (even randomized) corresponds to roughly 68%, which is our
current output with this number of throws.
"""
