import numpy as np
import matplotlib.pyplot as plt

x = np.arange(1.92,2.08,.001)

Ya = x**9-(18*x**8)+(144*x**7)-(672*x**6)+(2016*x**5)-(4032*x**4)+(5376*x**3)-(4608*x**2)+(2304*x)-512

print(Ya)

plt.plot(x,Ya)

plt.show()

Yb = (x-2)**9

plt.plot(x,Yb)
plt.show()

