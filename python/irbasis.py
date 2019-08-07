from __future__ import print_function
from builtins import range

import os
import numpy
import h5py
import bisect
from itertools import product
from numpy.polynomial.legendre import legval
import scipy.special
import mpmath
from mpmath import mp, mpf

mp.dps = 30

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

        # Differential operator for \tilde{P}_l[x]
        self._deriv_mat = numpy.zeros((self._np, self._np))
        for l in range(self._np):
            for m in range(l-1, -1, -2):
                self._deriv_mat[m, l] = 2*m + 1
            #print(l, self._deriv_mat[:, l])
        coeffs = numpy.sqrt(numpy.arange(self._np) + 0.5) # Conversion between \tilde{P}_l and P_l
        self._deriv_mat = numpy.einsum('i,ij,j->ij', 1/coeffs, self._deriv_mat, coeffs)

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

    def sl(self, l):
        """
        Return the singular value for the l-th basis function

        Parameters
        ----------
        l : int
            index of the singular values/basis functions

        Returns
        sl : float
            singular value
        -------

        """
        return self._sl[l]
    
    def ulx(self, l, x):
        """
        Return value of basis function for x

        Parameters
        ----------
        l : int
            index of basis functions
        x : float
            dimensionless parameter x (-1 <= x <= 1)

        Returns
        -------
        ulx : float
            value of basis function u_l(x)
        """
        if not -1 <= x <= 1:
            raise RuntimeError("x should be in [-1,1]!")

        if x >= 0:
            return self._eval(x, self._ulx_data[l, :, :], self._ulx_section_edges)
        else:
            return self._eval(-x, self._ulx_data[l, :, :], self._ulx_section_edges) * _even_odd_sign(l)

    def d_ulx(self, l, x, order, section=-1):
        """
        Return (higher-order) derivatives of u_l(x)

        Parameters
        ----------
        l : int
            index of basis functions
        x : int
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
        if not -1 <= x <= 1:
            raise RuntimeError("x should be in [-1,1]!")

        if x >= 0:
            return self._eval_derivative(x, order, self._ulx_data[l, :, :], self._ulx_section_edges, section)
        else:
            return self._eval_derivative(-x, order, self._ulx_data[l, :, :], self._ulx_section_edges, section) * _even_odd_sign(l + order)

    def vly(self, l, y):
        """
        Return value of basis function for y

        Parameters
        ----------
        l : int
            index of basis functions
        y : float
            dimensionless parameter y (-1 <= y <= 1)

        Returns
        -------
        vly : float
            value of basis function v_l(y)
        """
        if not -1 <= y <= 1:
            raise RuntimeError("y should be in [-1,1]!")

        if y >= 0:
            return self._eval(y, self._vly_data[l, :, :], self._vly_section_edges)
        else:
            return self._eval(-y, self._vly_data[l, :, :], self._vly_section_edges) * _even_odd_sign(l)


    def d_vly(self, l, y, order):
        """
        Return (higher-order) derivatives of v_l(y)

        Parameters
        ----------
        l : int
            index of basis functions
        y : int
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
        if not -1 <= y <= 1:
            raise RuntimeError("y should be in [-1,1]!")

        if y >= 0:
            return self._eval_derivative(y, order, self._vly_data[l, :, :], self._vly_section_edges)
        else:
            return self._eval_derivative(-y, order, self._vly_data[l, :, :], self._vly_section_edges) * _even_odd_sign(l+order)

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
        for l in range(self.dim()):
            deriv_x1[l, :] = numpy.array([self.d_ulx(l, 1.0, o) for o in range(num_deriv)])
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

    def _eval(self, x, data, section_edges):
        """
        data : (num_sections, np)
        """
        section_idx = min(bisect.bisect_right(section_edges, x)-1, len(section_edges)-2)

        return self._eval_impl(x, section_edges[section_idx], section_edges[section_idx+1], data[section_idx, :])

    def _eval_impl(self, x, x_s, x_sp, coeffs):
        """
        coeffs is 1D or 2D array containing expansion coefficients in terms of \tilde{P}_l
        """
        dx = x_sp - x_s
        tilde_x = (2*x - x_sp - x_s)/dx

        return legval(tilde_x, coeffs * self._norm_coeff) * numpy.sqrt(2/dx)

    def _eval_derivative(self, x, order, data, section_edges, section=-1):
        """
        If section = -1, the index is determined by binary search

        """
        section_idx = section if section >= 0 else min(bisect.bisect_right(section_edges, x)-1, len(section_edges)-2)
        coeffs = self._differentiate_coeff(data[section_idx, :], order)
        dx = section_edges[section_idx+1] - section_edges[section_idx]
        return self._eval_impl(x, section_edges[section_idx], section_edges[section_idx+1], coeffs) * ((2/dx)**order)

    def _differentiate_coeff(self, coeffs, order):
        """

        Parameters
        ----------
        coeffs : coefficients of piecewise Legendre polynomial (\tilde{P}_l)
        order : order of differentiation (1 is for the first derivative)

        Returns
        -------
        Coefficients representing derivatives

        """

        for i in range(order):
            coeffs = numpy.dot(self._deriv_mat, coeffs)
        return coeffs

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
