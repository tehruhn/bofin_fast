import scipy.integrate as integrate
from scipy.integrate import quad
import numpy as np
from scipy.optimize import curve_fit, least_squares
from scipy.constants import k

def coth(x, beta):
    return 1/np.tanh(x)

def normalize(x):
    """
    Normalize x
    """
    return x/np.linalg.norm(x)

def J(w,a):
    """
    Mauro's definition
    """
    aa = np.conjugate(a)
    return w/((w-a)*(w+a)*(w-aa)*(w+aa))

def spectrum_neill(w, lam, gamma, w0):
    """
    Calculate the spectrum as defined by Neill
    """
    return ((lam**2)*gamma*w)/((w**2 - w0**2)**2 + gamma**2 * w**2)

def drude_lorrentz(w, lam, gamma, w0):
    """
    The lorrentz spectral density
    """
    num = lam*gamma
    den = ((w - w0)**2 + gamma**2)
    return(num/den)

def jk(w, *poles):
    """
    Calculate the jk function described in [1] given by:
    
    jk(w) = prod(1/[(w-a)*(w-a*)])
    
    Parameters
    ==========
    w: ndarray
        An array of values for the frequencies.
    
    wks: list
        A list of poles for the spectral density.
    """
    terms = np.array([1/((w-wjk)*(w-np.conjugate(wjk))) for wjk in poles])
    return np.prod(terms, 0)

# def spectrum(w, poles, pk=[1.], n=1):
#     """
#     Calculates the spectrum from the jk function specified with the poles.
    
#     Parameters
#     ==========
#     w: ndarray
#         A 1D array of frequencies
    
#     jk_list: list
#         A list of functions which takes in w and poles to compute jk
    
#     n: int
#         Odd integers for the power of the frequency term in the spectrum
    
#     pk: list
#         List of coefficients
    
#     poles: list
#         A list of poles for the function `jk_func`
#     """
#     jks = np.array([jk(w, p) - jk(-w, p) for p in poles])
#     return np.sum(np.multiply(pk, jks), 0)*(w**(n-1))

def bath_correlation(tlist, spectrum,beta, wstart = 0., wend = 1.):
    """
    The correlation function calculated for the specific spectrum at a given
    temperature of the bath. At zero temperature the coth(x) is set to 1 for
    the real part of the correlation function integral
    
    Parameters
    ==========
    tlist: ndarray
        A 1D array with the times to calculate the correlation at

    spectrum: callable
        A function of the form f(w, *params) which calculates the spectral
        densities for the given parameters. For example, this could be set
        to `lorrentz`
        
    params: ndarray
        A 1D array of parameters for the spectral density function.
        `[gamma, lam, w0, wc]`
    
    wstart, wend: float
        The starting and ending value of the angular frequencies for 
        integration. In general the intergration is for all values but
        since at higher frequencies, the spectral density is zero, we set
        a finite limit to the numberical integration
    
    temperature: float
        The absolute temperature of the bath in Kelvin. If the temperature
        is set to zero, we can replace the coth(x) term in the correlation
        function's real part with 1. At higher temperatures the coth(x)
        function behaves poorly at low frequencies.
    
    Returns
    =======
    corrR: ndarray
        A 1D array of the real part of the correlation function
    
    corrI: ndarray
        A 1D array of the imaginary part of the correlation function
    """ 
    corrR = []
    corrI = []
    integrandR = lambda w, t: spectrum(w)*(coth(beta*(w/2)))*np.cos(w*t)
    integrandI = lambda w, t: -spectrum(w)*np.sin(w*t)
    for i in tlist:
        corrR.append(quad(integrandR, wstart, wend, args=(i,))[0])
        corrI.append(quad(integrandI, wstart, wend, args=(i,))[0])
    return np.array(corrR)/np.pi , np.array(corrI)/np.pi

def _S(w, beta, a, lam, gamma):
    """
    Mauro's symmetric spectrum
    """
    aa = np.conjugate(a)
    prefactor = -(lam**2)*gamma/(a**2 - aa**2)

    t1 = coth(beta*(a/2))*(a/(a**2 - w**2))
    t2 = coth(beta*(aa/2))*(aa/(aa**2 - w**2))
    return prefactor*(t1 + t2)

def _A(w, beta, a):
    """
    Mauro's anti-symmetric spectrum
    """
    return J(w, a)*np.heaviside(w, 1) - J(-w, a)*np.heaviside(-w, 1)


def analytical_spectrum_matsubara(w, beta, a, lam, gamma):
    """
    The Matsubara part of the spectrum
    """
    return -_S(w, beta, a, lam, gamma) + _A(w, beta, a)*coth(beta*w/2)

def analytical_spectrum_non_matsubara(w, beta, a, lam, gamma):
    """
    The non-matsubara part of the spectrum
    """
    return _S(w, beta, a, lam, gamma) + _A(w, beta, a)

def corr_non_matsubara(tlist, beta, a, lam, gamma):
    """
    The auxiliary part of correlation which gives the non-matsubara terms
    """
    aa = np.conjugate(a)
    prefactor = (1j*gamma*lam**2/2)*(1/(a**2 - aa**2))
    t1 = (coth(beta*(a/2)) - 1)*np.exp(1j*a*tlist) + (coth(beta*(aa/2)) + 1)*np.exp(-1j*aa*tlist)
    t2 = (coth(beta*(aa/2)) - 1)*np.exp(1j*aa*tlist) + (coth(beta*(a/2)) + 1)*np.exp(-1j*a*tlist)
    return prefactor*(t1*np.heaviside(tlist, 1) + t2*np.heaviside(-tlist, 1))

def _matsubara_corr(t,beta,a,cut_off):
    """
    Mauro calculated correlation
    """
    aa = np.conjugate(a)    
    if t >= 0:
        temp = sum([n*np.exp(-2*np.pi/beta * n*t)/ ((a**2+(2*np.pi/beta*n)**2)*(aa**2+(2*np.pi/beta*n)**2)) for n in range(1, cut_off)])
    if t < 0:
        temp = sum([n * np.exp(2*np.pi/beta * n*t)/ ((a**2+(2*np.pi/beta*n)**2)*(aa**2+(2*np.pi/beta*n)**2)) for n in range(1, cut_off)])
    return -4*(np.pi/beta)**2*temp

def corr_matsubara(tlist, beta, a, cut_off, lam, gamma):
    prefactor = gamma*lam**2/np.pi
    return [prefactor*_matsubara_corr(i, beta, a, cut_off) for i in tlist]


def _exp_fit(tt, yy):
    """
    Fits a exponential decay to the signal.
    
                        A e^(-w t)
    
    Returns
    =======
    fitfunc: callable
        A callable function of the form f(t) which can be used to
        generate the data according to the fitted function and the
        obtained parameters.
    
    params: ndarray
        A 1D array of the fitted parameters A, w
    """
    ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing

    Fyy = abs(np.fft.rfft(yy))
    guess_freq = abs(ff[np.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(yy) * 2.**0.5
    guess_offset = np.mean(yy)
    calculated_guess = np.array([guess_amp, 2.*np.pi*guess_freq])
    
    p0 = calculated_guess

    expfit = lambda x, A, w: A * np.exp(-w*x)
    popt, pcov = curve_fit(expfit, xdata=tt, ydata=yy, p0=p0)
    A, w = popt
    fitfunc = lambda x: A * np.exp(-w*x)
    fitparams = np.array([A, w])
    return fitfunc, fitparams

def exp_fit(tt, yy, num_exp = 3):
    """
    Fits the y values to a sum of exponents given by num_exp
    by repeated subtraction and fitting
    """
    exponents = []
    for i in range(num_exp):
        f, _p = _exp_fit(tt, yy)
        yy = yy - f(tt)
        exponents.append(_p)
    return np.array(exponents)


def sum_of_exponentials(tt, ck, vk):
    """
    For a set of ck and vk return ck[i]e^{vk[i}
    """
    y = np.multiply(ck[0], np.exp(vk[0]*tt))
    for p in range(1, len(ck)):
        y += np.multiply(ck[p], np.exp(vk[p]*tt))
    return y

def analytical_exponentials_non_matsubara(gamma, lam, w0, beta):
    """
    Get the exponentials for the correlation function for non-matsubara
    terms. (t>=0)
    
    Parameters
    ----------
    gamma, lam, w0: float
    
    Returns
    -------
    ck: ndarray
        A 1D array with the prefactors for the exponentials

    vk: ndarray
        A 1D array with the frequencies
    """
    omega = np.sqrt(w0**2 - (gamma/2)**2)
    a = omega + 1j*gamma/2.
    aa = np.conjugate(a)
    coeff = (1j*gamma*lam**2/2)*(1/(a**2 - aa**2))
    
    vk = np.array([1j*a, -1j*aa])
    ck = np.array([coth(beta*(a/2))-1, coth(beta*(aa/2))+1])
    
    return coeff*ck, vk

def analytical_exponentials_matsubara(gamma, lam, w0, beta, cut_off):
    """
    Get the exponentials for the correlation function for matsubara
    terms. (t>=0)
    
    Parameters
    ----------
    gamma, lam, w0: float
    cut_off: int

    Returns
    -------
    ck: ndarray
        A 1D array with the prefactors for the exponentials

    vk: ndarray
        A 1D array with the frequencies
    """
    if beta == np.inf:
        raise ValueError("Use the function matsubara_analytical_zero_temp(tlist, gamma, lam)")
    omega = np.sqrt(w0**2 - (gamma/2)**2)
    a = omega + 1j*gamma/2.
    aa = np.conjugate(a)
    coeff = (-4*gamma*lam**2/np.pi)*((np.pi/beta)**2)
    vk = np.array([-2*np.pi*n/(beta) for n in range(1, cut_off)])
    ck = np.array([n/((a**2 + (2*np.pi*n/beta)**2)
                                   *(aa**2 + (2*np.pi*n/beta)**2)) for n in range(1, cut_off)])
    return -coeff*ck, vk


def matsubara_analytical_zero_temp(tlist, gamma, lam, w0):
    """
    Analytical zero temperature value for Matsubara when T = 0
    """
    omega = np.sqrt(w0**2 - (gamma/2)**2)
    a = omega + 1j*gamma/2.
    aa = np.conjugate(a)
    prefactor = -(lam**2*gamma)/np.pi
    integrand = lambda x: prefactor*((x*np.exp(-x*t))/((a**2 + x**2)*(aa**2 + x**2)))
    return quad(integrand, 0.0, np.inf)[0]

def fit_matsubara(t_train, y_train, bounds=([0, -np.inf, 0, -np.inf], [10, 0, 10, 0])):
    """
    Fit a bi-exponential as A1e^(-w1 t) + A2 e^(-w2 t)
    """
    fun = lambda x, t, y: np.power(x[0]*np.exp(x[1]*t) + x[2]*np.exp(x[3]*t) - y, 2)
    x0 = [0.5, -1, 0.5, -1]
    params = least_squares(fun, x0, loss='cauchy', args=(t_train, y_train), bounds=bounds)
    c1, v1, c2, v2 = params.x
    return np.array([c1, c2]), np.array([v1, v2])


def local_calc_matsubara_params(lam0, gam, beta, N_m):
    """
    Calculate the Matsubara coefficents and frequencies
    Returns
    -------
    c, nu: both list(float)
    """
    c = []
    nu = []
    hbar=1
    
    g = 2*np.pi / (beta)
    for k in range(N_m):
        if k == 0:
            nu.append(gam)
            c.append(lam0*gam*
                (1.0/np.tan(gam*hbar*beta/2.0) - 1j) / hbar)
        else:
            nu.append(k*g)
            c.append(4*lam0*gam*nu[k] /
                  ((nu[k]**2 - gam**2)*beta*hbar**2))

    return c, nu


def spectrum(w, lam, gamma, w0):
    """
    Calculate the spectrum as defined by Neill
    """
    return ((lam**2)*gamma*w)/((w**2 - w0**2)**2 + gamma**2 * w**2)

def coth(x):
    """
    coth function
    """
    return 1/np.tanh(x)

def integrand(w, lam, gamma, w0, beta, t):
    return (-4.*spectrum(w, lam, gamma, w0)/w**2)*(1. - np.cos(w*t))*(coth(beta * (w/2)))

def evolution(tlist, lam, gamma, w0, beta, wq):
    """
    Compute the evolution
    """
    integrated = lambda t: quad(integrand, 0.0, np.inf, args=(lam, gamma, w0, beta, t))
    evolution = np.array([np.exp(1j*wq*t + integrated(t)[0]/np.pi) for t in tlist])
    return evolution

def matsubara_analytical_zero_temp(t, gamma, lam, w0):
    """
    Analytical zero temperature value for Matsubara when T = 0
    """
    omega = np.sqrt(w0**2 - (gamma/2)**2)
    a = omega + 1j*gamma/2.
    aa = np.conjugate(a)
    prefactor = -(lam**2*gamma)/np.pi
    integrand = lambda x: prefactor*((x*np.exp(-x*t))/((a**2 + x**2)*(aa**2 + x**2)))
    return quad(integrand, 0.0, np.inf)[0]

def pure_dephasing_evolution_analytical(tlist, wq, ck, vk):
    """
    Computes the propagating function appearing in the pure dephasing model.
        
    Parameters
    ----------
    t: float
        A float specifying the time at which to calculate the integral.
    
    wq: float
        The qubit frequency in the Hamiltonian.

    ck: ndarray
        The list of coefficients in the correlation function.
        
    vk: ndarray
        The list of frequencies in the correlation function.
    
    Returns
    -------
    integral: float
        The value of the integral function at time t.
    """
    evolution = np.array([np.exp(-1j*wq*t - correlation_integral(t, ck, vk)) for t in tlist])
    return evolution

def correlation_integral(t, ck, vk):
    """
    Computes the integral sum function appearing in the pure dephasing model.
    
    If the correlation function is a sum of exponentials then this sum
    is given by:
    
    .. math:
        
        \int_0^{t}d\tau D(\tau) = \sum_k\frac{c_k}{\mu_k^2}e^{\mu_k t}
        + \frac{\bar c_k}{\bar \mu_k^2}e^{\bar \mu_k t}
        - \frac{\bar \mu_k c_k + \mu_k \bar c_k}{\mu_k \bar \mu_k} t
        + \frac{\bar \mu_k^2 c_k + \mu_k^2 \bar c_k}{\mu_k^2 \bar \mu_k^2}
        
    Parameters
    ----------
    t: float
        A float specifying the time at which to calculate the integral.
    
    ck: ndarray
        The list of coefficients in the correlation function.
        
    vk: ndarray
        The list of frequencies in the correlation function.
    
    Returns
    -------
    integral: float
        The value of the integral function at time t.
    """
    t1 = np.sum(np.multiply(np.divide(ck, vk**2), np.exp(vk*t) - 1))
    
    t2 = np.sum(np.multiply(np.divide(np.conjugate(ck), np.conjugate(vk)**2),
                            np.exp(np.conjugate(vk)*t) - 1))
    t3 = np.sum((np.divide(ck, vk) + np.divide(np.conjugate(ck), np.conjugate(vk)))*t)

    return 2*(t1+t2-t3)





















# ==================================================================================================
# Mauros code
# ==================================================================================================
def analytical_real(t,beta,a):
    aa = np.conjugate(a)
    if t >= 0:
        return np.real(np.pi * 1j / 2. * (1 / (a**2 - aa**2)) * ((coth(beta*a/2.) - 1) * np.exp(1j * a * t) + (coth(beta*aa/2.) + 1) * np.exp(-1j * aa * t)))
    if t < 0:
        return  np.real(np.pi * 1j / 2. * (1 / (a**2 - aa**2)) * ((coth(beta*aa/2.) - 1) * np.exp(1j * aa * t) + (coth(beta*a/2.) + 1) * np.exp(-1j * a * t)))
    
def analytical_imag(t,beta,a):
    aa = np.conjugate(a)
    if t >= 0:
        return np.imag(np.pi * 1j / 2. * (1 / (a**2 - aa**2)) * ((coth(beta*a/2.) - 1) * np.exp(1j * a * t) + (coth(beta*aa/2.) + 1) * np.exp(-1j * aa * t)))
    if t < 0:
        return  np.imag(np.pi * 1j / 2. * (1 / (a**2 - aa**2)) * ((coth(beta*aa/2.) - 1) * np.exp(1j * aa * t) + (coth(beta*a/2.) + 1) * np.exp(-1j * a * t)))

def toIntegrate_real(w, t, beta, a):
    return J(w,a)*coth(beta*w/2.)*np.cos(w*t)

def toIntegrate_imag(w,t,beta,a):
    return -J(w,a)*np.sin(w * t)

def analytical_real(t,beta,a):
    aa = np.conjugate(a)
    if t >= 0:
        return np.real(np.pi * 1j / 2. * (1 / (a**2 - aa**2)) * ((coth(beta*a/2.) - 1) * np.exp(1j * a * t) + (coth(beta*aa/2.) + 1) * np.exp(-1j * aa * t)))
    if t < 0:
        return  np.real(np.pi * 1j / 2. * (1 / (a**2 - aa**2)) * ((coth(beta*aa/2.) - 1) * np.exp(1j * aa * t) + (coth(beta*a/2.) + 1) * np.exp(-1j * a * t)))
    
def analytical_imag(t,beta,a):
    aa = np.conjugate(a)
    if t >= 0:
        return np.imag(np.pi * 1j / 2. * (1 / (a**2 - aa**2)) * ((coth(beta*a/2.) - 1) * np.exp(1j * a * t) + (coth(beta*aa/2.) + 1) * np.exp(-1j * aa * t)))
    if t < 0:
        return  np.imag(np.pi * 1j / 2. * (1 / (a**2 - aa**2)) * ((coth(beta*aa/2.) - 1) * np.exp(1j * aa * t) + (coth(beta*a/2.) + 1) * np.exp(-1j * a * t)))

def C_real(t,w,beta,a):
    return analytical_real(t,beta,a) * np.cos(w * t)  - analytical_imag(t,beta,a) * np.sin(w*t)

def C_imag(t,w,beta,a):
    return analytical_imag(t,beta,a) * np.cos(w * t)  - analytical_real(t,beta,a) * np.sin(w*t) 
    
def spectrum_real(w,beta,a,T):
    return integrate.quad(C_real,-T,T,args=(w,beta,a))[0]

def spectrum_analytical_real(w,beta,a):
    aa = np.conjugate(a)
    temp_1 = (coth(beta*a/2.)-1)*1j/(a+w)
    temp_2 = (coth(beta*aa/2.)+1)*1j/(-aa+w)
    temp_3 = (coth(beta*aa/2.)-1)*1j/(-aa-w)
    temp_4 = (coth(beta*a/2.)+1)*1j/(a-w)
    return 1j * np.pi / (2 * (a**2 - aa**2)) * (temp_1 +temp_2+temp_3+temp_4)

def spectrum_analytical_real(w,beta,a):
    aa = np.conjugate(a)
    return np.real(1j * np.pi / (2 * (a**2 - aa**2)) * ((coth(beta*a/2.)*(2*a*1j/(a**2-w**2))+coth(beta*aa/2.)*(-2*aa*1j/(aa**2-w**2)))\
    +2 * 1j * (aa**2 - a**2) * w / ((w-a)*(w+a)*(w-aa)*(w+aa))))

def spectrum_analytical_real_Matsubara(w,beta,a,cut_off):
    aa = np.conjugate(a)
    temp =  np.real(1j * np.pi / (2 * (a**2 - aa**2)) * ((coth(beta*a/2.)*(2*a*1j/(a**2-w**2))+coth(beta*aa/2.)*(-2*aa*1j/(aa**2-w**2)))\
    +2 * 1j * (aa**2 - a**2) * w / ((w-a)*(w+a)*(w-aa)*(w+aa))))
    temp = temp - 4 * (np.pi/beta)**2 * 4*np.pi*beta *sum([n**2/ ((a**2+(2*np.pi/beta*n)**2)*(aa**2+(2*np.pi/beta*n)**2)*((2*np.pi*n)**2+beta**2*w**2)) for n in np.linspace(0,cut_off,cut_off+1)])
    return temp

def spectrum_analytical_real_Matsubara_2(w,beta,a,cut_off):
    aa = np.conjugate(a)
    S =  np.real(1j * np.pi / (2 * (a**2 - aa**2)) * ((coth(beta*a/2.)*(2*a*1j/(a**2-w**2))+coth(beta*aa/2.)*(-2*aa*1j/(aa**2-w**2)))))
    A = np.real(1j * np.pi / (2 * (a**2 - aa**2)) * 2 * 1j * (aa**2 - a**2) * w / ((w-a)*(w+a)*(w-aa)*(w+aa)))
    temp = -(S + A * coth(-beta*w/2.))
    return S + A + temp


def spectrum_imag(w,beta,a,T):
    return integrate.quad(C_imag,-T,0,args=(w,beta,a))[0] + integrate.quad(C_imag,0,T,args=(w,beta,a))[0]

def spectrum_analytical_imag(w,beta,a):
    aa = np.conjugate(a)
    return np.imag(1j * np.pi / (2 * (a**2 - aa**2)) * ((coth(beta*a/2.)*(2*a*1j/(a**2-w**2))+coth(beta*aa/2.)*(-2*aa*1j/(aa**2-w**2)))\
    +2*1j*w/(a**2-w**2)-2*1j*w/(aa**2-w**2)))

def Matsubara(t,beta,a,cut_off):
    aa = np.conjugate(a)    
    if t >= 0:
        temp = sum([n * np.exp(-2*np.pi/beta * n*t)/ ((a**2+(2*np.pi/beta*n)**2)*(aa**2+(2*np.pi/beta*n)**2)) for n in np.linspace(0,cut_off,cut_off+1)])
    if t < 0:
        temp = sum([n * np.exp(2*np.pi/beta * n*t)/ ((a**2+(2*np.pi/beta*n)**2)*(aa**2+(2*np.pi/beta*n)**2)) for n in np.linspace(0,cut_off,cut_off+1)])
    return -4 * (np.pi/beta)**2 * temp

def Matsubara_2(w,beta,a,cut_off):
    aa = np.conjugate(a)
    S =  np.real(1j * np.pi / (2 * (a**2 - aa**2)) * ((coth(beta*a/2.)*(2*a*1j/(a**2-w**2))+coth(beta*aa/2.)*(-2*aa*1j/(aa**2-w**2)))))
    A = np.real(1j * np.pi / (2 * (a**2 - aa**2)) * 2 * 1j * (aa**2 - a**2) * w / ((w-a)*(w+a)*(w-aa)*(w+aa)))
    temp = -(S + A * coth(-beta*w/2.))
    return temp

def Matsubara_3(w,beta,a,cut_off):
    aa = np.conjugate(a)
    S =  np.real(1j * np.pi / (2 * (a**2 - aa**2)) * (2*a*1j/(a**2-w**2)-2*aa*1j/(aa**2-w**2)))
    A = np.real(1j * np.pi / (2 * (a**2 - aa**2)) * 2 * 1j * (aa**2 - a**2) * abs(w) / ((w-a)*(w+a)*(w-aa)*(w+aa)))
    temp = -(S - A)
    return temp

def spectrum_analytical_real_Matsubara(w,beta,a,cut_off):
    aa = np.conjugate(a)
    temp =  np.real(1j * np.pi / (2 * (a**2 - aa**2)) * ((coth(beta*a/2.)*(2*a*1j/(a**2-w**2))+coth(beta*aa/2.)*(-2*aa*1j/(aa**2-w**2)))\
    +2 * 1j * (aa**2 - a**2) * w / ((w-a)*(w+a)*(w-aa)*(w+aa))))
#     print(- 4 * (np.pi/beta)**2 * 4*np.pi*sum([n**2/ ((a**2+(2*np.pi/beta*n)**2)*(aa**2+(2*np.pi/beta*n)**2)*((2*np.pi*n)**2+beta**2*w**2)) for n in np.linspace(0,cut_off,cut_off+1)]))
    return temp


