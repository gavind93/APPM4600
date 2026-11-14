import numpy as np
from scipy.special import gamma
from scipy.integrate import quad

def trapezoid(f, a, b, n):
    h = (b - a) / n
    t = np.linspace(a, b, n + 1)
    y = f(t)
    return h * (0.5 * y[0] + y[1:-1].sum() + 0.5 * y[-1])

def gamma_trap(x, T, n):
    f = lambda t: np.where(t > 0, t**(x - 1) * np.exp(-t), 1.0 if x == 1 else 0.0)
    return trapezoid(f, 0.0, T, n)

def gauss_laguerre(g, n):
    x,w = np.polynomial.laguerre.laggauss(n)
    return np.sum(w * g(x))

def gamma_gl(x,n):
    func = lambda t: t**(x-1)
    return gauss_laguerre(func,n)

def relerr(approx, exact):
    return abs(approx - exact) / abs(exact)


h = 0.01
for x in [2, 4, 6, 8, 10]:
    T = x + 40 + 10 * np.sqrt(x)      # truncation point
    n = int(round(T / h))
    approx = gamma_trap(x, T, n)
    exact =  gamma(x)
    print(f"x={x:2d}  T={T:6.1f}  trap={approx:.10f}  scipy={exact:.1f}  "
          f"rel.err={abs(approx-exact)/exact:.2e} nfev={n+1}")

quad_res = {}
for x in [2, 4, 6, 8, 10]:
    T = x + 40 + 10 * np.sqrt(x)
    val, est_err, info = quad(lambda t: t**(x - 1) * np.exp(-t), 0, T,
                              full_output=1)
    exact = gamma(x)
    quad_res[x] = (relerr(val, exact), info['neval'])
    print(f"x={x:2d}  quad={val:.12g}  rel.err={quad_res[x][0]:.2e}  "
          f"nfev={info['neval']}")

for x in [2, 4, 6, 8, 10]:
    for n in [2, 4, 6, 8, 10]:
        v, ex = gamma_gl(x, n), gamma(x)
        print(f"x={x:2d} n={n:2d} approx={v:.10f} rel.err={abs(v-ex)/ex:.2e}")

'''
x_vals = [2 4 6 8 10];
tol = 1e-6;                         % quad's default absolute tolerance
for x = x_vals
    T = x + 40 + 10*sqrt(x);        % same truncation as part (a)
    nfev = 0;
    f = @(t) integrand(t, x);
    [val, nfev] = quad(f, 0, T, tol);   % 2nd output of quad = # function evals
    % (older versions: quad returns [Q, fcnt])
    ex = factorial(x-1);
    fprintf('x=%2d  quad=%.12g  relerr=%.2e  nfev=%d\n', x, val, abs(val-ex)/ex, nfev);
end

function y = integrand(t, x)        % must be vectorized: quad passes vectors
    y = t.^(x-1) .* exp(-t);
'''
