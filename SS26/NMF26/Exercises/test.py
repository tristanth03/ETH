

import numpy as np
import matplotlib.pyplot as plt
h = 1/(100+1)
b = lambda x_a,x_b,x_c,x : ((x-x_a)*int((x_a<x)*(x<=x_b))+(x_c-x)*int((x_b<x)*(x<=x_c)))/h
x = np.arange(0,10)
plt.plot(b(0,1,2,x))
plt.show()

