import numpy as np
import matplotlib.pyplot as plt
import math
import random

X1 = math.pi

X2 = 10**6

exponents = np.arange(-16,1,1)
x_deltas = 10.0**exponents

def my_func(x,delta):
    temp = -2*np.sin(x+(delta/2))*np.sin(delta/2)
    return temp

def original_func(x,delta):
    temp = np.cos(x+delta)-np.cos(x)
    return temp


# x = pi
y_pi_my = my_func(X1,x_deltas)
y_pi_orig = original_func(X1,x_deltas)
y_pi_diff = abs(y_pi_orig-y_pi_my)

# x = 10^6
y_10_my = my_func(X2,x_deltas)
y_10_orig = original_func(X2,x_deltas)
y_10_diff = abs(y_10_orig-y_10_my)

# plot for x=pi
plt.plot(x_deltas,y_pi_diff)
plt.xscale('log')
plt.yscale('log')
plt.title('Difference for x=pi')
plt.show()

#plot for x = 10^6
plt.plot(x_deltas,y_10_diff)
plt.xscale('log')
plt.yscale('log')
plt.title('Difference for x=10^6')
plt.show()


#Taylor Expansion
def taylor_func(x,delta):
    temp = delta*-np.sin(x)+(delta**2)/2*-np.cos(x)
    return temp

# Taylor examples vs the original function
taylor_pi = taylor_func(X1,x_deltas)
taylor_pi_diff = abs(y_pi_orig-taylor_pi)

taylor_10 = taylor_func(X2,x_deltas)
taylor_10_diff = abs(y_10_orig-taylor_10)


plt.plot(x_deltas,taylor_pi_diff)
plt.xscale('log')
plt.yscale('log')
plt.title('Difference for Taylor\'s and original at x=pi')
plt.show()

plt.plot(x_deltas,taylor_10_diff)
plt.xscale('log')
plt.yscale('log')
plt.title('Difference for Taylor\'s and original at x=10^6')
plt.show()