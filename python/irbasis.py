from __future__ import print_function

import os
import numpy
import h5py
from numpy.polynomial.legendre import legval, legder
import scipy.special

# Get version string
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'version'), 'r', encoding='ascii') as f:
    __version__ = f.read().strip()

def _check_type(obj, *types):
    if not isinstance(obj, types):
        raise RuntimeError(
                "Passed the argument is type of %s, but expected one of %s"
                % (type(obj), str(types)))


def load(statistics, Lambda, h5file=""):
    assert statistics == "F" or statistics == "B"

    Lambda = float(Lambda)

    if h5file == "":
        name = os.path.dirname(os.path.abspath(__file__)) 
        file_name = os.path.normpath(os.path.join(name, './irbasis.h5'))
    else:
        file_name = h5file

    prefix = "basis_f-mp-Lambda"+str(Lambda) if statistics == 'F' else "basis_b-mp-Lambda"+str(Lambda)

    with h5py.File(file_name, 'r') as f:
        if not prefix in f:
            raise RuntimeError("No data available!")

        return basis(file_name, prefix)


class _PiecewiseLegendrePoly:
    """Piecewise Legendre polynomial.

    Models a function on the interval `[-1, 1]` as a set of segments on the
    intervals `S[i] = [a[i], a[i+1]]`, where on each interval the function
    is expanded in scaled Legendre polynomials.
    """
    def __init__(self, data, knots, dx):
        """Piecewise Legendre polynomial"""
        data = numpy.array(data)
        knots = numpy.array(knots)
        polyorder, nsegments = data.shape[:2]
        if knots.shape != (nsegments+1,):
            raise ValueError("Invalid knots array")
        if (numpy.diff(knots) < 0).any():
            raise ValueError("Knots must be monotonically increasing")
        if not numpy.allclose(dx, knots[1:] - knots[:-1]):
            raise ValueError("dx must work with knots")

        self.nsegments = nsegments
        self.polyorder = polyorder
        self.xmin = knots[0]
        self.xmax = knots[-1]

        self.knots = knots
        self.dx = dx
        self.data = data
        self._xm = .5 * (knots[1:] + knots[:-1])
        self._inv_xs = 2/dx
        self._norm = numpy.sqrt(self._inv_xs)

    def _split(self, x):
        """Split segment"""
        if (x < self.xmin).any() or (x > self.xmax).any():
            raise ValueError("x must be in [%g, %g]" % (self.xmin, self.xmax))

        i = self.knots.searchsorted(x, 'right').clip(None, self.nsegments)
        i -= 1
        xtilde = x - self._xm[i]
        xtilde *= self._inv_xs[i]
        return i, xtilde

    def __call__(self, x, l=None):
        """Evaluate polynomial at position x"""
        i, xtilde = self._split(numpy.asarray(x))

        if l is None:
            # Evaluate for all values of l.  xtilde and data array must be
            # broadcast'able against each other, so we append a dimension here
            xtilde = xtilde[(slice(None),) * xtilde.ndim + (None,)]
            data = self.data[:,i,:]
        else:
            numpy.broadcast(xtilde, l)
            data = self.data[:,i,l]

        res = legval(xtilde, data, tensor=False)
        res *= self._norm[i]
        return res

    def deriv(self, n=1):
        """Get polynomial for the n'th derivative"""
        ddata = legder(self.data, n)
        scale = self._inv_xs ** n
        ddata *= scale[None, :, None]
        return _PiecewiseLegendrePoly(ddata, self.knots, self.dx)


def _preprocess_irdata(data, knots, knots_corr=None):
    """Perform preprocessing of IR data"""
    data = numpy.array(data)
    dim, nsegments, polyorder = data.shape
    if knots_corr is None:
        knots_corr = numpy.zeros_like(knots)

    # First, the basis is given by *normalized* Legendre function,
    # so we have to undo the normalization here:
    norm = numpy.sqrt(numpy.arange(polyorder) + 0.5)
    data *= norm

    # The functions are stored for [0,1] only, since they are
    # either even or odd for even or odd orders, respectively. We
    # undo this here, because it simplifies the logic.
    mdata = data[:,::-1].copy()
    mdata[1::2,:,0::2] *= -1
    mdata[0::2,:,1::2] *= -1
    data = numpy.concatenate((mdata, data), axis=1)
    knots = numpy.concatenate((-knots[::-1], knots[1:]), axis=0)
    knots_corr = numpy.concatenate((-knots_corr[::-1], knots_corr[1:]), axis=0)
    dx = (knots[1:] - knots[:-1]) + (knots_corr[1:] - knots_corr[:-1])

    # Transpose following numpy polynomial convention
    data = data.transpose(2,1,0)
    return data, knots, dx


class basis(object):
    def __init__(self, file_name, prefix=""):
        
        with h5py.File(file_name, 'r') as f:
            self._Lambda = f[prefix+'/info/Lambda'][()]
            self._dim = f[prefix+'/info/dim'][()]
            self._statistics = 'B' if f[prefix+'/info/statistics'][()] == 0 else  'F'

            self._sl = f[prefix+'/sl'][()]

            ulx_data = f[prefix+'/ulx/data'][()] # (l, section, p)
            ulx_section_edges = f[prefix+'/ulx/section_edges'][()]
            ulx_section_edges_corr = f[prefix+'/ulx/section_edges_corr'][()]
            assert ulx_data.shape[0] == self._dim
            assert ulx_data.shape[1] == f[prefix+'/ulx/ns'][()]
            assert ulx_data.shape[2] == f[prefix+'/ulx/np'][()]

            vly_data = f[prefix+'/vly/data'][()]
            vly_section_edges = f[prefix+'/vly/section_edges'][()]
            assert vly_data.shape[0] == self._dim
            assert vly_data.shape[1] == f[prefix+'/vly/ns'][()]
            assert vly_data.shape[2] == f[prefix+'/vly/np'][()]

            # Reference data:
            # XXX: shall we move this to the tests?
            self._ulx_ref_max = f[prefix+'/ulx/ref/max'][()]
            self._ulx_ref_data = f[prefix+'/ulx/ref/data'][()]
            self._vly_ref_max = f[prefix+'/vly/ref/max'][()]
            self._vly_ref_data = f[prefix+'/vly/ref/data'][()]

            assert f[prefix+'/ulx/np'][()] == f[prefix+'/vly/np'][()]

            np = f[prefix+'/vly/np'][()]
            self._np = np

        self._ulx_ppoly = _PiecewiseLegendrePoly(
                *_preprocess_irdata(ulx_data, ulx_section_edges, ulx_section_edges_corr))
        self._vly_ppoly = _PiecewiseLegendrePoly(
                *_preprocess_irdata(vly_data, vly_section_edges))

        deriv_x1 = numpy.asarray(list(_derivs(self._ulx_ppoly, x=1)))
        moments = _power_moments(self._statistics, deriv_x1)
        self._ulw_model = _PowerModel(self._statistics, moments)

    @property
    def Lambda(self):
        """
        Dimensionless parameter of IR basis

        Returns
        -------
            Lambda : float
        """
        return self._Lambda

    @property
    def statistics(self):
        """
        Statistics

        Returns
        -------
        statistics : string
            "F" for fermions, "B" for bosons
        """
        return self._statistics

    def dim(self):
        """
        Return dimension of basis

        Returns
        -------
        dim : int
        """
        return self._dim

    def sl(self, l=None):
        """
        Return the singular value for the l-th basis function

        Parameters
        ----------
        l : int, int-like array or None
            index of the singular values/basis functions. If None, return all.

        Returns
        sl : float
            singular value
        -------

        """
        if l is None: l = Ellipsis

        return self._sl[l]
    
    def ulx(self, l, x):
        """
        Return value of basis function for x

        Parameters
        ----------
        l : int, int-like array or None
            index of basis functions. If None, return array with all l
        x : float or float-like array
            dimensionless parameter x (-1 <= x <= 1)

        Returns
        -------
        ulx : float
            value of basis function u_l(x)
        """
        return self._ulx_ppoly(x,l)

    def d_ulx(self, l, x, order, section=None):
        """
        Return (higher-order) derivatives of u_l(x)

        Parameters
        ----------
        l : int, int-like array or None
            index of basis functions. If None, return array with all l
        x : float or float-like array
            dimensionless parameter x
        order : int
            order of derivative (>=0). 1 for the first derivative.
        section : int
            index of the section where x is located.

        Returns
        -------
        d_ulx : float
            (higher-order) derivative of u_l(x)

        """
        return self._ulx_ppoly.deriv(order)(x,l)

    def vly(self, l, y):
        """
        Return value of basis function for y

        Parameters
        ----------
        l : int, int-like array or None
            index of basis functions. If None, return array with all l
        y : float or float-like array
            dimensionless parameter y (-1 <= y <= 1)

        Returns
        -------
        vly : float
            value of basis function v_l(y)
        """
        return self._vly_ppoly(y,l)

    def d_vly(self, l, y, order):
        """
        Return (higher-order) derivatives of v_l(y)

        Parameters
        ----------
        l : int, int-like array or None
            index of basis functions. If None, return array with all l
        y : float or float-like array
            dimensionless parameter y
        order : int
            order of derivative (>=0). 1 for the first derivative.
        section : int
            index of the section where y is located.

        Returns
        -------
        d_vly : float
            (higher-order) derivative of v_l(y)

        """
        return self._vly_ppoly.deriv(order)(y,l)


    def compute_unl(self, n, whichl=None):
        """
        Compute transformation matrix from IR to Matsubara frequencies

        Parameters
        ----------
        n : int or 1D ndarray of integers
            Indices of Matsubara frequncies

        whichl : vector of integers or None
            Indices of the l values

        Returns
        -------
        unl : 2D array of complex
            The shape is (niw, nl) where niw is the dimension of the input "n"
            and nl is the dimension of the basis

        """
        n = numpy.asarray(n)
        if not numpy.issubdtype(n.dtype, numpy.integer):
            RuntimeError("n must be integer")
        if whichl is None:
            whichl = slice(None)
        else:
            whichl = numpy.ravel(whichl)

        zeta = 1 if self._statistics == 'F' else 0
        wn_flat = 2 * n.ravel() + zeta

        # The radius of convergence of the asymptotic expansion is Lambda/2,
        # so for significantly larger frequencies we use the asymptotics,
        # since it has lower relative error.
        cond_inner = numpy.abs(wn_flat[:,None]) < 40 * self._Lambda
        result_inner = _compute_unl(self._ulx_ppoly, wn_flat, whichl)
        result_asymp = self._ulw_model.giw(wn_flat)[:,whichl]
        result_flat = numpy.where(cond_inner, result_inner, result_asymp)
        return result_flat.reshape(n.shape + result_flat.shape[-1:])


    def num_sections_x(self):
        """
        Number of sections of piecewise polynomial representation of u_l(x)

        Returns
        -------
        num_sections_x : int
        """
        return self._ulx_ppoly.nsegments

    @property
    def section_edges_x(self):
        """
        End points of sections for u_l(x)

        Returns
        -------
        section_edges_x : 1D ndarray of float
        """
        return self._ulx_ppoly.knots

    def num_sections_y(self):
        """
        Number of sections of piecewise polynomial representation of v_l(y)

        Returns
        -------
        num_sections_y : int
        """
        return self._vly_ppoly.nsegments

    @property
    def section_edges_y(self):
        """
        End points of sections for v_l(y)

        Returns
        -------
        section_edges_y : 1D ndarray of float
        """
        return self._vly_ppoly.knots

    def sampling_points_x(self, whichl):
        """
        Computes "optimal" sampling points in x space for given basis
    
        Parameters
        ----------
        b :
            basis object
        whichl: int
            Index of reference basis function "l"
    
        Returns
        -------
        sampling_points: 1D array of float
            sampling points in x space
        """

        return sampling_points_x(self, whichl)

    def sampling_points_y(self, whichl):
        """
        Computes "optimal" sampling points in y space for given basis
    
        Parameters
        ----------
        b :
            basis object
        whichl: int
            Index of reference basis function "l"
    
        Returns
        -------
        sampling_points: 1D array of float
            sampling points in y space
        """

        return sampling_points_y(self, whichl)

    def sampling_points_matsubara(self, whichl):
       """
       Computes "optimal" sampling points in Matsubara domain for given basis
   
       Parameters
       ----------
       b :
           basis object
       whichl: int
           Index of reference basis function "l"
   
       Returns
       -------
       sampling_points: 1D array of int
           sampling points in Matsubara domain
   
       """

       return sampling_points_matsubara(self, whichl)
   
    def _check_ulx(self):
        ulx_max = self._ulx_ref_max[2]
        ulx_ref = numpy.array([ (_data[0], _data[1], abs(self.ulx(int(_data[0]-1), _data[1])-_data[3])/ulx_max ) for _data in self._ulx_ref_data[self._ulx_ref_data[:,2]==0]])
        return(ulx_ref)

    def _get_d_ulx_ref(self):
        return self._ulx_ref_data

    def _check_vly(self):
        vly_max = self._vly_ref_max[2]
        vly_ref = numpy.array([ (_data[0], _data[1], abs(self.vly(int(_data[0]-1), _data[1])-_data[3])/vly_max ) for _data in self._vly_ref_data[ self._vly_ref_data[:,2]==0]])
        return(vly_ref)

    def _get_d_vly_ref(self):
        return self._vly_ref_data


class _PowerModel:
    """Model from a high-frequency series expansion:

        A(iw) = sum(A[n] / (iw)**(n+1) for n in range(1, N))

    where `iw == 1j * pi/2 * wn` is a reduced imaginary frequency, i.e.,
    `wn` is an odd/even number for fermionic/bosonic frequencies.
    """
    def __init__(self, statistics, moments):
        """Initialize model"""
        self.zeta = {'F': 1, 'B': 0}[statistics]
        self.moments = numpy.asarray(moments)
        self.nmom, self.nl = self.moments.shape

    @staticmethod
    def _inv_iw_safe(wn, result_dtype):
        """Return inverse of frequency or zero if freqency is zero"""
        result = numpy.zeros(wn.shape, result_dtype)
        wn_nonzero = wn != 0
        result[wn_nonzero] = 1/(1j * numpy.pi/2 * wn[wn_nonzero])
        return result

    def _giw_ravel(self, wn):
        """Return model Green's function for vector of frequencies"""
        result_dtype = numpy.result_type(1j, wn, self.moments)
        result = numpy.zeros((wn.size, self.nl), result_dtype)
        inv_iw = self._inv_iw_safe(wn, result_dtype)[:,None]
        for mom in self.moments[::-1]:
            result += mom
            result *= inv_iw
        return result

    def giw(self, wn):
        """Return model Green's function for reduced frequencies"""
        wn = numpy.array(wn)
        if (wn % 2 != self.zeta).any():
            raise ValueError("expecting 'reduced' frequencies")

        return self._giw_ravel(wn.ravel()).reshape(wn.shape + (self.nl,))


def _derivs(ppoly, x):
    """Evaluate polynomial and its derivatives at specific x"""
    yield ppoly(x)
    for _ in range(ppoly.polyorder-1):
        ppoly = ppoly.deriv()
        yield ppoly(x)


def _power_moments(stat, deriv_x1):
    """Return moments"""
    statsign = {'F': -1, 'B': 1}[stat]
    mmax, lmax = deriv_x1.shape
    m = numpy.arange(mmax)[:,None]
    l = numpy.arange(lmax)[None,:]
    coeff_lm = ((-1.0)**(m+1) + statsign * (-1.0)**l) * deriv_x1
    return -statsign/numpy.sqrt(2.0) * coeff_lm


def _imag_power(n):
    """Imaginary unit raised to an integer power without numerical error"""
    n = numpy.asarray(n)
    if not numpy.issubdtype(n.dtype, numpy.integer):
        raise ValueError("expecting set of integers here")
    cycle = numpy.array([1, 0+1j, -1, 0-1j], complex)
    return cycle[n % 4]


def _get_tnl(l, w):
    r"""Fourier integral of the l-th Legendre polynomial:

        T_l(w) = \int_{-1}^1 dx \exp(iwx) P_l(x)
    """
    i_pow_l = _imag_power(l)
    return 2 * numpy.where(
        w >= 0,
        i_pow_l * scipy.special.spherical_jn(l, w),
        (i_pow_l * scipy.special.spherical_jn(l, -w)).conj(),
        )


def _shift_xmid(knots, dx):
    r"""Return midpoint relative to the nearest integer plus a shift

    Return the midpoints `xmid` of the segments, as pair `(diff, shift)`,
    where shift is in `(0,1,-1)` and `diff` is a float such that
    `xmid == shift + diff` to floating point accuracy.
    """
    dx_half = dx / 2
    xmid_m1 = dx.cumsum() - dx_half
    xmid_p1 = -dx[::-1].cumsum()[::-1] + dx_half
    xmid_0 = knots[1:] - dx_half

    shift = numpy.round(xmid_0).astype(int)
    diff = numpy.choose(shift+1, (xmid_m1, xmid_0, xmid_p1))
    return diff, shift


def _phase_stable(poly, wn):
    """Phase factor for the piecewise Legendre to Matsubara transform.

    Compute the following phase factor in a stable way:

        numpy.exp(1j * numpy.pi/2 * wn[:,None] * poly.dx.cumsum()[None,:])
    """
    # A naive implementation is losing precision close to x=1 and/or x=-1:
    # there, the multiplication with `wn` results in `wn//4` almost extra turns
    # around the unit circle.  The cosine and sines will first map those
    # back to the interval [-pi, pi) before doing the computation, which loses
    # digits in dx.  To avoid this, we extract the nearest integer dx.cumsum()
    # and rewrite above expression like below.
    #
    # Now `wn` still results in extra revolutions, but the mapping back does
    # not cut digits that were not there in the first place.
    xmid_diff, extra_shift = _shift_xmid(poly.knots, poly.dx)
    phase_shifted = numpy.exp(1j * numpy.pi/2 * wn[None,:] * xmid_diff[:,None])
    corr = _imag_power((extra_shift[:,None] + 1) * wn[None,:])
    return corr * phase_shifted


def _compute_unl(poly, wn, whichl):
    """Compute piecewise Legendre to Matsubara transform."""
    posonly = slice(poly.nsegments//2, None)
    dx_half = poly.dx[posonly] / 2
    data_sc = poly.data[:,posonly,whichl] * numpy.sqrt(dx_half/2)[None,:,None]
    p = numpy.arange(poly.polyorder)

    wred = numpy.pi/2 * wn
    phase_wi = _phase_stable(poly, wn)[posonly]
    t_pin = _get_tnl(p[:,None,None], wred[None,:] * dx_half[:,None]) * phase_wi

    # Perform the following, but faster:
    #   resulth = einsum('pin,pil->nl', t_pin, data_sc)
    npi = poly.polyorder * poly.nsegments // 2
    resulth = t_pin.reshape(npi,-1).T.dot(data_sc.reshape(npi,-1))

    # We have integrated over the positive half only, so we double up here
    zeta = wn[0] % 2
    l = numpy.arange(poly.data.shape[-1])[whichl]
    return numpy.where(l % 2 != zeta, 2j * resulth.imag, 2 * resulth.real)

#
# The functions below are for sparse sampling
#

def _funique(x, tol=2e-16):
    """Removes duplicates from an 1D array within tolerance"""
    x = numpy.sort(x)
    unique = numpy.ediff1d(x, to_end=2*tol) > tol
    x = x[unique]
    return x


def _find_roots(ulx):
    """Find all roots in (-1, 1) using double exponential mesh + bisection"""
    Nx = 10000
    eps = 1e-14
    tvec = numpy.linspace(-3, 3, Nx)  # 3 is a very safe option.
    xvec = numpy.tanh(0.5 * numpy.pi * numpy.sinh(tvec))

    zeros = []
    for i in range(Nx - 1):
        if ulx(xvec[i]) * ulx(xvec[i + 1]) < 0:
            a = xvec[i + 1]
            b = xvec[i]
            u_a = ulx(a)
            while a - b > eps:
                half_point = 0.5 * (a + b)
                if ulx(half_point) * u_a > 0:
                    a = half_point
                else:
                    b = half_point
            zeros.append(0.5 * (a + b))
    return numpy.array(zeros)


def sampling_points_x(b, whichl):
    """
    Computes "optimal" sampling points in x space for given basis

    Parameters
    ----------
    b :
        basis object
    whichl: int
        Index of reference basis function "l"

    Returns
    -------
    sampling_points: 1D array of float
        sampling points in x space
    """
    _check_type(b, basis)

    xroots = _find_roots(lambda x: b.ulx(whichl, x))
    xroots_ex = numpy.hstack((-1.0, xroots, 1.0))
    return 0.5 * (xroots_ex[:-1] + xroots_ex[1:])


def sampling_points_y(b, whichl):
    """
    Computes "optimal" sampling points in y space for given basis

    Parameters
    ----------
    b :
        basis object
    whichl: int
        Index of reference basis function "l"

    
    -------
    sampling_points: 1D array of float
        sampling points in y space
    """
    _check_type(b, basis)

    roots_positive_half = 0.5 * _find_roots(lambda y: b.vly(whichl, (y + 1)/2)) + 0.5
    if whichl % 2 == 0:
        roots_ex = numpy.sort(numpy.hstack([-1, -roots_positive_half, roots_positive_half, 1]))
    else:
        roots_ex = numpy.sort(numpy.hstack([-1, -roots_positive_half, 0, roots_positive_half, 1]))

    return 0.5 * (roots_ex[:-1] + roots_ex[1:])

def _start_guesses(n=1000):
    "Construct points on a logarithmically extended linear interval"
    x1 = numpy.arange(n)
    x2 = numpy.array(numpy.exp(numpy.linspace(numpy.log(n), numpy.log(1E+8), n)), dtype=int)
    x = numpy.unique(numpy.hstack((x1, x2)))
    return x


def _get_unl_real(basis_xy, x, l):
    "Return highest-order basis function on the Matsubara axis"
    unl = basis_xy.compute_unl(x, l)
    result = numpy.zeros(unl.shape, float)

    # Purely real functions
    zeta = 1 if basis_xy.statistics == 'F' else 0
    if l % 2 == zeta:
        assert numpy.allclose(unl.imag, 0)
        return unl.real
    else:
        assert numpy.allclose(unl.real, 0)
        return unl.imag


def _sampling_points(fn):
    "Given a discretized 1D function, return the location of the extrema"
    fn = numpy.asarray(fn)
    fn_abs = numpy.abs(fn)
    sign_flip = fn[1:] * fn[:-1] < 0
    sign_flip_bounds = numpy.hstack((0, sign_flip.nonzero()[0] + 1, fn.size))
    points = []
    for segment in map(slice, sign_flip_bounds[:-1], sign_flip_bounds[1:]):
        points.append(fn_abs[segment].argmax() + segment.start)
    return numpy.asarray(points)


def _full_interval(sample, stat):
    if stat == 'F':
        return numpy.hstack((-sample[::-1]-1, sample))
    else:
        # If we have a bosonic basis and even order (odd maximum), we have a
        # root at zero. We have to artifically add that zero back, otherwise
        # the condition number will blow up.
        if sample[0] == 0:
            sample = sample[1:]
        return numpy.hstack((-sample[::-1], 0, sample))


def _get_mats_sampling(basis_xy, lmax=None):
    "Generate Matsubara sampling points from extrema of basis functions"
    if lmax is None:
        lmax = basis_xy.dim()-1

    x = _start_guesses()
    y = _get_unl_real(basis_xy, x, lmax)
    x_idx = _sampling_points(y)

    sample = x[x_idx]
    return _full_interval(sample, basis_xy.statistics)


def sampling_points_matsubara(b, whichl):
    """
    Computes "optimal" sampling points in Matsubara domain for given basis

    Parameters
    ----------
    b :
        basis object
    whichl: int
        Index of reference basis function "l"

    Returns
    -------
    sampling_points: 1D array of int
        sampling points in Matsubara domain

    """
    _check_type(b, basis)

    stat = b.statistics

    assert stat == 'F' or stat == 'B' or stat == 'barB'

    if whichl > b.dim()-1:
        raise RuntimeError("Too large whichl")

    return _get_mats_sampling(b, whichl)
