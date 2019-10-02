from __future__ import print_function
from builtins import range

import os
import numpy
import h5py
import bisect
from itertools import product
from numpy.polynomial.legendre import legval, legder
import scipy.special
import mpmath
from mpmath import mp, mpf

mp.dps = 30

def _check_type(obj, types):
    if isinstance(obj, type):
        raise RuntimeError("Passed the argument is type of" + str(type(obj)) + ", but expected to be one of " + " ".join(map(str, types)))

def _mpmath_op(array, func):
    func_v = numpy.vectorize(func)
    return func_v(array)

def _cos(array):
    return _mpmath_op(array, mpmath.cos)

def _sin(array):
    return _mpmath_op(array, mpmath.sin)

def _mk_mp_array(x_array, y_array):
    """
    Construct an array of mpf from two arrays of float type
    """
    assert x_array.shape == y_array.shape

    x = numpy.ravel(x_array)
    y = numpy.ravel(y_array)

    N = x.size
    res = numpy.empty((N,), dtype=mpf)
    for i in range(N):
        res[i] = mpf(x[i]) + mpf(y[i])

    return res.reshape(x_array.shape)

def _load_mp_array(h5, key):
    return _mk_mp_array(h5[key][()], h5[key + '_corr'][()])

def _even_odd_sign(l):
    return 1 if l%2==0 else -1

def _compute_tnl(wvec, n_legendre):
    num_w = len(wvec)
    tnl = numpy.zeros((num_w, n_legendre), dtype=complex)
    wvec_f = wvec.astype(float)

    # Positive part
    mask = wvec_f >= 0
    for il in range(n_legendre):
        tnl[mask, il] = 2 * (1J**il) * scipy.special.spherical_jn(il, wvec_f[mask])

    mask = wvec_f < 0
    for il in range(n_legendre):
        tnl[mask, il] = numpy.conj(2 * (1J**il) * scipy.special.spherical_jn(il, -wvec_f[mask]))

    return tnl

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

def _compute_unl_tail(w_vec, stastics, deriv_x1):
    sign_statistics = 1 if stastics == 'B' else -1

    n_iw = len(w_vec)
    Nl, num_deriv = deriv_x1.shape

    # coeffs_nm
    coeffs_nm = numpy.zeros((n_iw, num_deriv), dtype=complex)
    for i_iw in range(n_iw):
        if stastics == "B" and w_vec[i_iw] == 0:
            continue
        fact = 1J /w_vec[i_iw]
        coeffs_nm[i_iw, 0] = fact
        for m in range(1, num_deriv):
            coeffs_nm[i_iw, m] = fact * coeffs_nm[i_iw, m-1]

    # coeffs_lm
    coeffs_lm = numpy.zeros((Nl, num_deriv))
    for l, m in product(range(Nl), range(num_deriv)):
        coeffs_lm[l, m] = (1 - sign_statistics * _even_odd_sign(l+m)) * deriv_x1[l, m]

    return -(sign_statistics/numpy.sqrt(2.0)) * numpy.einsum('ij,kj->ik', coeffs_nm, coeffs_lm)


class _PiecewiseLegendrePoly:
    """Piecewise Legendre polynomial.

    Models a function on the interval `[-1, 1]` as a set of segments on the
    intervals `S[i] = [a[i], a[i+1]]`, where on each interval the function
    is expanded in scaled Legendre polynomials.
    """
    def __init__(self, data, knots):
        """Piecewise Legendre polynomial"""
        data = numpy.array(data)
        knots = numpy.array(knots)
        polyorder, nsegments = data.shape[:2]
        if knots.shape != (nsegments+1,):
            raise ValueError("Invalid knost array")
        if (numpy.diff(knots) < 0).any():
            raise ValueError("Knots must be monotonically increasing")

        self.nsegments = nsegments
        self.polyorder = polyorder
        self.xmin = knots[0]
        self.xmax = knots[-1]

        self.knots = knots
        self.data = data
        self._xm = .5 * (knots[1:] + knots[:-1])
        self._inv_xs = 2 / (knots[1:] - knots[:-1])
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

    def __call__(self, x, l=Ellipsis):
        """Evaluate polynomial at position x"""
        i, xtilde = self._split(x)
        res = legval(xtilde, self.data[:,i,l], tensor=False)
        res *= self._norm[i]
        return res

    def deriv(self, n=1):
        """Get polynomial for the n'th derivative"""
        ddata = legder(self.data, n)
        scale = self._inv_xs ** n
        ddata *= scale[(slice(None),) + (None,) * (ddata.ndim-2)]
        return _PiecewiseLegendrePoly(ddata, self.knots)


def _preprocess_irdata(data, knots):
    """Perform preprocessing of IR data"""
    data = numpy.array(data)
    dim, nsegments, polyorder = data.shape

    # First, the basis is given by *normalized* Legendre function
    # so we have to undo the normalization here:
    norm = numpy.sqrt(numpy.arange(polyorder) + 0.5)
    data *= norm

    # The functions are stored for [0,1] only, since they are
    # either even or odd for even or odd orders, respectively. We
    # undo this here, because it simplifies the logic.
    mknots = -knots[::-1]
    mdata = data[:,::-1].copy()
    mdata[1::2,:,0::2] *= -1
    mdata[0::2,:,1::2] *= -1
    data = numpy.concatenate((mdata, data), axis=1)
    knots = numpy.concatenate((mknots, knots[1:]), axis=0)

    # Transpose following numpy polynomial convention
    data = data.transpose(2,1,0)
    return data, knots



class basis(object):
    def __init__(self, file_name, prefix=""):
        
        with h5py.File(file_name, 'r') as f:
            self._Lambda = f[prefix+'/info/Lambda'][()]
            self._dim = f[prefix+'/info/dim'][()]
            self._statistics = 'B' if f[prefix+'/info/statistics'][()] == 0 else  'F'

            self._sl = f[prefix+'/sl'][()]

            self._ulx_data = f[prefix+'/ulx/data'][()] # (l, section, p)
            #self._ulx_data_mp = _load_mp_array(f, prefix + '/ulx/data')  # (l, section, p)

            self._ulx_data_for_vec = self._ulx_data.transpose((1, 2, 0)) # (section, p, l)
            self._ulx_ref_max = f[prefix+'/ulx/ref/max'][()]
            self._ulx_ref_data = f[prefix+'/ulx/ref/data'][()]
            self._ulx_section_edges = f[prefix+'/ulx/section_edges'][()]
            self._ulx_section_edges_mp = _load_mp_array(f, prefix + '/ulx/section_edges')
            assert self._ulx_data.shape[0] == self._dim
            assert self._ulx_data.shape[1] == f[prefix+'/ulx/ns'][()]
            assert self._ulx_data.shape[2] == f[prefix+'/ulx/np'][()]
            
            self._vly_data = f[prefix+'/vly/data'][()]
            self._vly_ref_max = f[prefix+'/vly/ref/max'][()]
            self._vly_ref_data = f[prefix+'/vly/ref/data'][()]
            self._vly_section_edges = f[prefix+'/vly/section_edges'][()]
            assert self._vly_data.shape[0] == self._dim
            assert self._vly_data.shape[1] == f[prefix+'/vly/ns'][()]
            assert self._vly_data.shape[2] == f[prefix+'/vly/np'][()]

            assert f[prefix+'/ulx/np'][()] == f[prefix+'/vly/np'][()]

            np = f[prefix+'/vly/np'][()]
            self._np = np

        self._ulx_ppoly = _PiecewiseLegendrePoly(
                *_preprocess_irdata(self._ulx_data, self._ulx_section_edges))
        self._vly_ppoly = _PiecewiseLegendrePoly(
                *_preprocess_irdata(self._vly_data, self._vly_section_edges))

        self._norm_coeff = numpy.sqrt(numpy.arange(np) + 0.5)
        self._norm_coeff_mp = numpy.array([mpmath.sqrt(n + mpmath.mpf('0.5')) for n in numpy.arange(np)], dtype=mpf)

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
        if l is None: l = Ellipsis
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
        if l is None: l = Ellipsis
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
        if l is None: l = Ellipsis
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
        if l is None: l = Ellipsis
        return self._vly_ppoly.deriv(order)(y,l)

    def compute_unl(self, n):
        """
        Compute transformation matrix from IR to Matsubara frequencies

        Parameters
        ----------
        n : int or 1D ndarray of integers
            Indices of Matsubara frequncies

        Returns
        -------
        unl : 2D array of complex
            The shape is (niw, nl) where niw is the dimension of the input "n" and nl is the dimension of the basis

        """
        from mpmath import mpf, pi

        if isinstance(n, int):
            num_n = 1
            o_vec = 2*numpy.array([n], dtype=mpf)
        elif (isinstance(n, numpy.ndarray) and numpy.issubdtype(n.dtype, numpy.integer)) or (isinstance(n, list) and numpy.all([type(e) == int for e in n])):
            num_n = len(n)
            o_vec = 2*numpy.array(n, dtype=mpf)
        else:
            raise RuntimeError("n is not an integer, list or a numpy array")

        if self._statistics == 'F':
            o_vec += 1

        w_vec = (pi * o_vec)/2

        num_deriv = self._ulx_data.shape[2]

        # Compute tail
        replaced_with_tail = numpy.zeros((num_n, self.dim()), dtype=int)
        deriv_x1 = numpy.zeros((self.dim(), num_deriv), dtype=float)
        for o in range(num_deriv):
            deriv_x1[:,o] = self.d_ulx(None, 1.0, o)
        unl_tail = _compute_unl_tail(w_vec.astype(float), self._statistics, deriv_x1)
        unl_tail_without_last_two = _compute_unl_tail(w_vec.astype(float), self._statistics, deriv_x1[:, :-2])
        for i in range(len(n)):
            if self._statistics == 'B' and n[i] == 0:
                continue
            for l in range(self.dim()):
                if numpy.abs((unl_tail[i, l] - unl_tail_without_last_two[i, l])/unl_tail[i, l]) < 1e-12:
                    replaced_with_tail[i, l] = 1

        unl = self._compute_tilde_unl_fast(w_vec)

        sign_shift = 1 if self._statistics == 'F' else 0
        for l in range(self.dim()):
            if (l + sign_shift) % 2 == 1:
                unl[:, l] = 2J * unl[:, l].imag
            else:
                unl[:, l] = 2 * unl[:, l].real

        # Overwrite by tail
        for i, l in product(range(len(n)), range(self.dim())):
            if replaced_with_tail[i, l] == 1:
                unl[i, l] = unl_tail[i, l]

        return unl

    def _compute_tilde_unl_fast(self, w_vec):
        num_n = len(w_vec)

        tilde_unl = numpy.zeros((num_n, self._dim), dtype=complex)
        for s in range(self.num_sections_x()):
            xs = self._ulx_section_edges_mp[s]
            xsp = self._ulx_section_edges_mp[s+1]
            dx = xsp - xs
            xmid = (xsp + xs)/2

            dx_f = float(dx)

            coeffs_lp = self._ulx_data[:, s, :] * numpy.sqrt(dx_f)/2

            phase = w_vec * (xmid+1)
            exp_n = _cos(phase) + mpmath.j * _sin(phase)
            tnp = _compute_tnl((dx * w_vec/2).astype(float), self._np)

            # n, np -> np
            # O(Nw * Np)
            tmp_np = exp_n[:, None].astype(complex) * tnp.astype(complex)

            assert tmp_np.dtype == complex

            # lp, p -> lp
            # O(Nl * Np)
            tmp_lp = coeffs_lp * self._norm_coeff[None, :]

            assert tmp_lp.dtype == float

            # O(Nw * Nl * Np)
            tilde_unl += numpy.tensordot(tmp_np, tmp_lp, axes=[[1], [1]])

        return tilde_unl


    def num_sections_x(self):
        """
        Number of sections of piecewise polynomial representation of u_l(x)

        Returns
        -------
        num_sections_x : int
        """
        return self._ulx_data.shape[1]

    @property
    def section_edges_x(self):
        """
        End points of sections for u_l(x)

        Returns
        -------
        section_edges_x : 1D ndarray of float
        """
        return self._ulx_section_edges

    def num_sections_y(self):
        """
        Number of sections of piecewise polynomial representation of v_l(y)

        Returns
        -------
        num_sections_y : int
        """
        return self._vly_data.shape[1]

    @property
    def section_edges_y(self):
        """
        End points of sections for v_l(y)

        Returns
        -------
        section_edges_y : 1D ndarray of float
        """
        return self._vly_section_edges

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
            u_b = ulx(b)
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

    Returns
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


def _get_unl_real(basis_xy, x):
    "Return highest-order basis function on the Matsubara axis"
    unl = basis_xy.compute_unl(x)
    result = numpy.zeros(unl.shape, float)

    # Purely real functions
    real_loc = 1 if basis_xy.statistics == 'F' else 0
    assert numpy.allclose(unl[:, real_loc::2].imag, 0)
    result[:, real_loc::2] = unl[:, real_loc::2].real

    # Purely imaginary functions
    imag_loc = 1 - real_loc
    assert numpy.allclose(unl[:, imag_loc::2].real, 0)
    result[:, imag_loc::2] = unl[:, imag_loc::2].imag
    return result


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
    if lmax is None: lmax = basis_xy.dim()-1

    x = _start_guesses()
    y = _get_unl_real(basis_xy, x)[:,lmax]
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
