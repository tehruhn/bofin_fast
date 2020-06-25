from spectra import bath_correlation
from math import exp
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


################################################
#
# Part to compute correlation function
def function(x):
    A = 0.01 
    wc = 2
    J  = (A*x)*exp((-1*x*x)/wc**2)
    return J
# # fits well for n = 3
# tlist = np.linspace(0,10,10000)
tlist = np.linspace(0,10,10000)
params = [1, 1, 0, 2]
# plt.plot(tlist, y)
# plt.show()

print("Computing bath correlation function.")
temp = 0.1
ans1, ans2 = bath_correlation(tlist, function, 1/temp, wstart=0, wend=10)
print("Got bath correlation function.")
# print(ans1.shape, ans2.shape)

################################################

#
# Part to do fitting
#

# choose how many exponents and which data to fit
# I think k = 3 is sufficient for both real and imaginary parts
k = 3
# a = 1 for real part, a = 2 for imaginary part
a = 1


# selects real/imaginary data
ans = 0
if a==1:
    ans = ans1
    print("Fitting real part.")
else:
    ans = ans2
    print("Fitting imaginary part.")
    
# wrapper to fitting function because varibale number of params
def wrapper_fit_func(x, N, *args):
    d, a, b, c = list(args[0][:N]), list(args[0][N:2*N]), list(args[0][2*N:3*N]),list(args[0][3*N:4*N])
    # print("debug")
    return fit_func(x, d, a, b, c, N)


# actual fitting function
def fit_func(x, d, a, b, c, N):
    tot = 0
    for i in range(N):
        # print(i)
        tot += d[i]*(np.cos(a[i]*x + c[i])*np.exp(b[i]*x))
    return tot


# the actual computing of fit
popt = []
pcov = []



print("Doing actual fitting.")
# tries to fit for k exponents
for i in range(k):
    params_0 = [0]*(4*(i+1))

    #sets initial guess
    guess = []
    abguess = [-0.1]*(2*(i+1))
    cguess = [0]*(i+1)
    dguess = [0]*(i+1)
    guess.extend(abguess)
    guess.extend(cguess)
    guess.extend(dguess)

    # sets bounds
    # a's = negative, b's negative
    # c's = 0 to two pi
    # d's = -1 to 1

    # sets lower bound
    b_lower = []
    b_higher = []
    alower = [-np.inf]*(i+1)
    clower = [0]*(i+1)
    dlower = [-1]*(i+1)
    b_lower.extend(dlower)
    b_lower.extend(alower)
    b_lower.extend(alower)
    b_lower.extend(clower)

    # sets lower bound
    ahigher = [0]*(i+1)
    chigher = [2*np.pi]*(i+1)
    dhigher = [1]*(i+1)
    b_higher.extend(dhigher)
    b_higher.extend(ahigher)
    b_higher.extend(ahigher)
    b_higher.extend(chigher)
    param_bounds = (b_lower, b_higher)
    # print(params_0)
    p1, p2 = curve_fit(lambda x, *params_0: wrapper_fit_func(x, i+1, \
        params_0), tlist, ans, p0=guess, bounds = param_bounds)
    popt.append(p1)
    pcov.append(p2)
    print(i+1)
    # print(len(p1)/3)

print(popt)

# function that evaluates values with fitted params at\
# given inputs
def checker(tlist, vals):
    y = []
    for i in tlist:
        # print(i)
        y.append(wrapper_fit_func(i, int(len(vals)/4), vals))
    return y

# # show plots of real and imaginary parts
# plt.plot(tlist, ans1)
# plt.show()

# plt.plot(tlist, ans2)
# plt.show()


# plots vals

for i in range(k):
    y = checker(tlist, popt[i])
    plt.plot(tlist, ans, tlist, y)
    plt.show()

# plt.plot(tlist, ans1, tlist, y2)
# plt.show()

