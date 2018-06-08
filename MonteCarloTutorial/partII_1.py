import random

DARTS=1000000
hits = 0
throws = 0
for i in range (0, DARTS):
    throws += 1
    x = random.uniform(-1,1)
    y = random.uniform(-1,1)
    distsquared = x**2 + y**2
    if distsquared <= 1.0:
        hits = hits + 1.0

estpi = 4 * (hits / throws)
print("pi = %s" % estpi)

"""
Need near 1 million to get near thousandths-place accuracy.
Trials:
1      3.142288
2      3.140096
3      3.140864

with 10 million:
3.14167
"""