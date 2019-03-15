from __future__ import print_function
from builtins import range

import os
import numpy
import h5py
import bisect
from itertools import product


def _even_odd_sign(l):
    return 1 if l%2==0 else -1


def _compute_unl_tail(w_vec, stastics, deriv_x1):
    sign_statistics = 1 if stastics == 'B' else -1

    n_iw = len(w_vec)
    Nl, num_deriv = deriv_x1.shape
    result = numpy.zeros((n_iw, Nl), dtype=complex)

    # coeffs_nm
    coeffs_nm = numpy.zeros((n_iw, num_deriv,), dtype=complex)
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


def _compute_unl_high_freq(mask, w_vec_org, deriv0, deriv1, x0, x1, result):
    """
    Compute unl by high-frequency formula
    :param mask:
    :param w_vec_org:
    :param deriv0: derivatives at x0
    :param deriv1: derivatives at x1
    :param x0: smaller end point of x
    :param x1: larger end point of x
    :param result:
    :return:
    """
    w_vec = w_vec_org[mask]

    nw = len(w_vec)
    nl = deriv0.shape[0]
    n_deriv = deriv0.shape[1]

    iw = numpy.einsum('i,j->ij', 1J * w_vec, numpy.ones((nl)))  # (nw, nl)

    exp10_d1 = numpy.einsum('w,ld->dwl', numpy.exp(1J * w_vec * (x1 - x0)), deriv1)  # (n_deriv, nw, nl)
    d0_work = numpy.einsum('w,ld->dwl', numpy.ones((nw)), deriv0)  # (n_deriv, nw, nl)
    coeff = numpy.einsum('dwl,w->dwl', exp10_d1 - d0_work, numpy.exp(1J * w_vec * x0))  # (n_deriv, nw, nl)

    jk = numpy.zeros((nw, nl))
    for k in range(n_deriv-1, -1, -1):
        jk = (coeff[k, :, :] - jk) / iw

    result[mask, :] += jk

def load(statistics, Lambda, h5file=""):
    assert statistics == "F" or statistics == "B"

    Lambda = float(Lambda)

    if h5file == "":
        name = os.path.dirname(os.path.abspath(__file__)) 
        file_name = os.path.normpath(os.path.join(name, './irbasis.h5'))
    else:
        file_name = h5file

    prefix = "basis_f-mp-Lambda"+str(Lambda)+"_np8" if statistics == 'F' else "basis_b-mp-Lambda"+str(Lambda)+"_np8"

    with h5py.File(file_name, 'r') as f:
        if not prefix in f:
            raise RuntimeError("No data available!")

        return basis(file_name, prefix)


class basis(object):
    def __init__(self, file_name, prefix=""):
        
        with h5py.File(file_name, 'r') as f:
            self._Lambda = f[prefix+'/info/Lambda'][()]
            self._dim = f[prefix+'/info/dim'][()]
            self._statistics = 'B' if f[prefix+'/info/statistics'][()] == 0 else  'F'

            self._sl = f[prefix+'/sl'][()]

            self._ulx_data = f[prefix+'/ulx/data'][()] # (l, section, p)
            self._ulx_data_for_vec = self._ulx_data.transpose((1, 2, 0)) # (section, p, l)
            self._ulx_ref_max = f[prefix+'/ulx/ref/max'][()]
            self._ulx_ref_data = f[prefix+'/ulx/ref/data'][()]
            self._ulx_section_edges = f[prefix+'/ulx/section_edges'][()]
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
            return self._interpolate(x, self._ulx_data[l, :, :], self._ulx_section_edges)
        else:
            return self._interpolate(-x, self._ulx_data[l, :, :], self._ulx_section_edges) * _even_odd_sign(l)

    def ulx_all_l(self, x):
        """
        Return value of basis function for x

        Parameters
        ----------
        x : float
            dimensionless parameter x (-1 <= x <= 1)

        Returns
        -------
        ulx : 1D ndarray
            values of basis functions u_l(x) for all l at the given x
        """
        if not -1 <= x <= 1:
            raise RuntimeError("x should be in [-1,1]!")

        ulx_data = self._interpolate_all_l(numpy.abs(x), self._ulx_data_for_vec, self._ulx_section_edges)
        if x < 0:
            # Flip the sign for odd l
            ulx_data[1::2] *= -1
        return ulx_data


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
            return self._interpolate_derivative(x, order, self._ulx_data[l, :, :], self._ulx_section_edges, section)
        else:
            return self._interpolate_derivative(-x, order, self._ulx_data[l, :, :], self._ulx_section_edges, section) * _even_odd_sign(l + order)

    def _d_ulx_all(self, l, x, section=-1):
        assert x >= 0
        return self._interpolate_derivatives(x, self._ulx_data[l, :, :], self._ulx_section_edges, section)
        #else:
            #return self._interpolate_derivatives(-x, self._ulx_data[l, :, :], self._ulx_section_edges, section) * _even_odd_sign(l + order)

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
            return self._interpolate(y, self._vly_data[l,:,:], self._vly_section_edges)
        else:
            return self._interpolate(-y, self._vly_data[l,:,:], self._vly_section_edges) * _even_odd_sign(l)


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
            return self._interpolate_derivative(y, order, self._vly_data[l,:,:], self._vly_section_edges)
        else:
            return self._interpolate_derivative(-y, order, self._vly_data[l,:,:], self._vly_section_edges) * _even_odd_sign(l + order)


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
        if isinstance(n, int):
            num_n = 1
            o_vec = 2*numpy.array([n], dtype=float)
        elif (isinstance(n, numpy.ndarray) and numpy.issubdtype(n.dtype, numpy.integer)) or (isinstance(n, list) and numpy.all([type(e) == int for e in n]) ):
            num_n = len(n)
            o_vec = 2*numpy.array(n, dtype=float)
        else:
            raise RuntimeError("n is not an integer, list or a numpy array")

        if self._statistics == 'F':
            o_vec += 1
        w_vec = 0.5 * numpy.pi * o_vec

        num_deriv = self._ulx_data.shape[2]

        # Compute tail
        replaced_with_tail = numpy.zeros((num_n, self.dim()), dtype=int)
        deriv_x1 = numpy.zeros((self.dim(), num_deriv), dtype=float)
        for l in range(self.dim()):
            deriv_x1[l, :] = self._d_ulx_all(l, 1.0)
        unl_tail = _compute_unl_tail(w_vec, self._statistics, deriv_x1)
        unl_tail_without_last_two = _compute_unl_tail(w_vec, self._statistics, deriv_x1[:, :-2])
        for i in range(len(n)):
            if self._statistics == 'B' and n[i] == 0:
                continue
            for l in range(self.dim()):
                if numpy.abs((unl_tail[i, l] - unl_tail_without_last_two[i, l])/unl_tail[i, l]) < 1e-10:
                    replaced_with_tail[i, l] = 1
        mask_tail = numpy.prod(replaced_with_tail, axis=1) == 0

        # Numerical integration
        result = numpy.zeros((num_n, self.dim()), dtype=complex)
        deriv0 = numpy.zeros((self.dim(), num_deriv), dtype=float)
        deriv1 = numpy.zeros((self.dim(), num_deriv), dtype=float)
        deg = 2 * num_deriv
        x_smpl_org, weight_org = numpy.polynomial.legendre.leggauss(deg)
        for s in range(self.num_sections_x()):
            x0 = self._ulx_section_edges[s]
            x1 = self._ulx_section_edges[s+1]

            # Derivatives at end points
            for l in range(self.dim()):
                deriv0[l, :] = self._d_ulx_all(l, x0, s)
                deriv1[l, :] = self._d_ulx_all(l, x1, s)

            # Mask based on phase shift
            mask = numpy.logical_and(numpy.abs(w_vec) * (x1-x0) > 0.1 * numpy.pi, mask_tail)
            
            # High frequency formula
            _compute_unl_high_freq(mask, w_vec, deriv0, deriv1, x0, x1, result)

            # low frequency formula (Gauss-Legendre quadrature)
            x_smpl = 0.5 * (x_smpl_org + 1) * (x1 - x0) + x0
            weight = weight_org * (x1 - x0)/2

            smpl_vals = numpy.zeros((deg, self.dim()))
            for ix in range(deg):
                smpl_vals[ix, :] = self.ulx_all_l(x_smpl[ix])
            mask_not = numpy.logical_and(numpy.logical_not(mask), mask_tail)
            exp_iwx = numpy.exp(numpy.einsum('w,x->wx', 1J * w_vec[mask_not], x_smpl))
            result[mask_not, :] += numpy.einsum('wx,x,xl->wl', exp_iwx, weight, smpl_vals)

        for l in range(self.dim()):
            if l % 2 == 0:
                result[:, l] = result[:, l].real
            else:
                result[:, l] = 1J * result[:, l].imag
        result = numpy.einsum('w,wl->wl', numpy.sqrt(2.) * numpy.exp(1J * w_vec), result)

        # Overwrite by tail
        for i, l in product(range(len(n)), range(self.dim())):
            if replaced_with_tail[i, l] == 1:
                result[i, l] = unl_tail[i, l]

        return result

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

    def _interpolate(self, x, data, section_edges):
        section_idx = min(bisect.bisect_right(section_edges, x)-1, len(section_edges)-2)
        return self._interpolate_impl(x - section_edges[section_idx], data[section_idx, :])

    def _interpolate_all_l(self, x, data, section_edges):
        section_idx = min(bisect.bisect_right(section_edges, x)-1, len(section_edges)-2)
        return self._interpolate_all_l_impl(x - section_edges[section_idx], data[section_idx, :, :])

    def _interpolate_derivative(self, x, order, data, section_edges, section=-1):
        """
        If section = -1, the index is determined by binary search

        """
        section_idx = section if section >= 0 else min(bisect.bisect_right(section_edges, x)-1, len(section_edges)-2)
        coeffs = self._differentiate_coeff(data[section_idx, :], order)
        return self._interpolate_impl(x - section_edges[section_idx], coeffs)

    def _interpolate_derivatives(self, x, data, section_edges, section=-1):
        """
        if section = -1, the index is determined by binary search
        """
        section_idx = section if section >= 0 else min(bisect.bisect_right(section_edges, x)-1, len(section_edges)-2)

        coeffs = data[section_idx, :]
        k = len(coeffs)
        coeffs_deriv = numpy.array(coeffs)
        result = numpy.zeros((k,))
        for o in range(k):
            result[o] = self._interpolate_impl(x - section_edges[section_idx], coeffs_deriv)
            for p in range(k-1):
                coeffs_deriv[p] = (p+1) * coeffs_deriv[p+1]
            coeffs_deriv[k-1] = 0

        return result

    def _interpolate_impl(self, dx, coeffs):
        value = 0.0
        dx_power = 1.0
        for p in range(len(coeffs)):
            value += dx_power * coeffs[p]
            dx_power *= dx

        return value

    def _interpolate_all_l_impl(self, dx, coeffs):
        """
        Evaluate the value of a polynomial

        Parameters
        ----------
        dx : float
            coordinate where interpolated value is evalued
        coeffs : 2D ndarray
            expansion coefficients (p, l)

        Returns
        ----------
        interpolated value
        """
        np, nl = coeffs.shape
        value = numpy.zeros((nl))
        dx_power = 1.0
        for p in range(np):
            value += dx_power * coeffs[p, :]
            dx_power *= dx

        return value

    def _differentiate_coeff(self, coeffs, order):
        """

        Parameters
        ----------
        coeffs : coefficients of piecewise polynomial
        order : order of differentiation (1 is for the first derivative)

        Returns
        -------
        Coefficients representing derivatives

        """
        k = len(coeffs)
        coeffs_deriv = numpy.array(coeffs)
        for o in range(order):
            for p in range(k-1-o):
                coeffs_deriv[p] = (p+1) * coeffs_deriv[p+1]
            coeffs_deriv[k-1-o] = 0

        return coeffs_deriv

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
