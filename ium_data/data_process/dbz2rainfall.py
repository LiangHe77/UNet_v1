from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from matplotlib import pyplot as plt



def dbz2rainfall():
    """
    Z-R relationship Z = a * (R ** b)
    R is rain rate in mm / hour
    Z is Not in Dbz, Dbz = 10 * log_10(Z)

    Dbz = 10 * b * lg(A)
    """
    a = 386
    b = 1.43
    dbz = np.arange(-10, 60, 1)
    tmp = np.power(10, dbz / 10) / a
    R = np.power(tmp, 1 / b)

    plt.plot(dbz, R)
    plt.show()

if __name__ == "__main__":
    dbz2rainfall()

