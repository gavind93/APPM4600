import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0,5,500)
f = lambda x: np.sin(x)


taylor = lambda x: x - (x**3)/6 + (x**5)/120

pade33 = lambda x: (x - (7/60)*x**3)/(1+(x**2)/20)
pade42 = lambda x: (x - (7/60)*x**3)/(1+(x**2)/20)
pade24 = lambda x: (x - (7/60)*x**3)/(1+(x**2)/20)

plt.semilogy(x, abs(f(x) - taylor(x)), lw = 2, label='Taylor $T_6$')
plt.semilogy(x, abs(f(x) - pade33(x)), lw = 4, label='Pade M=3, N=3')
plt.semilogy(x, abs(f(x) - pade42(x)), lw = 1.5, linestyle=(0, (5, 5)), label='Pade M=4, N=2')
plt.semilogy(x, abs(f(x) - pade24(x)), lw = 2, linestyle=(0, (1, 2)), label='Pade M=2, N=4')

plt.xlabel('xval')
plt.ylabel('absolute error')
plt.legend()
plt.grid(True)
plt.show()