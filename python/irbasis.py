from __future__ import print_function
from builtins import range

import os
import numpy
import h5py
import bisect
import platform
from itertools import product

is_python3 = int(platform.python_version_tuple()[0]) == 3

def _even_odd_sign(l):
    return 1 if l%2==0 else -1

def _compute_Tnl_tail(w_vec, stastics, deriv_x1):
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


def _compute_Tnl_high_freq(mask, w_vec_org, deriv0, deriv1, x0, x1, result):
    """
    Compute Tnl by high-frequency formula
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
            self._Lambda = f[prefix+'/info/Lambda'].value
            self._dim = f[prefix+'/info/dim'].value
            self._statistics = 'B' if f[prefix+'/info/statistics'].value == 0 else  'F'

            self._sl = f[prefix+'/sl'].value

            self._ulx_data = f[prefix+'/ulx/data'].value # (l, section, p)
            self._ulx_data_for_vec = self._ulx_data.transpose((1, 2, 0)) # (section, p, l)
            self._ulx_ref_max = f[prefix+'/ulx/ref/max'].value
            self._ulx_ref_data = f[prefix+'/ulx/ref/data'].value
            self._ulx_section_edges = f[prefix+'/ulx/section_edges'].value
            assert self._ulx_data.shape[0] == self._dim
            assert self._ulx_data.shape[1] == f[prefix+'/ulx/ns'].value
            assert self._ulx_data.shape[2] == f[prefix+'/ulx/np'].value
            
            self._vly_data = f[prefix+'/vly/data'].value
            self._vly_ref_max = f[prefix+'/vly/ref/max'].value
            self._vly_ref_data = f[prefix+'/vly/ref/data'].value
            self._vly_section_edges = f[prefix+'/vly/section_edges'].value
            assert self._vly_data.shape[0] == self._dim
            assert self._vly_data.shape[1] == f[prefix+'/vly/ns'].value
            assert self._vly_data.shape[2] == f[prefix+'/vly/np'].value
            
    def dim(self):
        return self._dim

    def sl(self, l):
        return self._sl[l]
    
    def ulx(self, l, x):
        if x >= 0:
            return self._interpolate(x, self._ulx_data[l, :, :], self._ulx_section_edges)
        else:
            return self._interpolate(-x, self._ulx_data[l, :, :], self._ulx_section_edges) * _even_odd_sign(l)

    def ulx_all_l(self, x):
        ulx_data = self._interpolate_all_l(numpy.abs(x), self._ulx_data_for_vec, self._ulx_section_edges)
        if x < 0:
            # Flip the sign for odd l
            ulx_data[1::2] *= -1
        return ulx_data

    def check_ulx(self):
        ulx_max = self._ulx_ref_max[2]
        ulx_ref = numpy.array([ (_data[0], _data[1], abs(self.ulx(int(_data[0]-1), _data[1])-_data[3])/ulx_max ) for _data in self._ulx_ref_data[self._ulx_ref_data[:,2]==0]])
        return(ulx_ref)
        
    def get_d_ulx_ref(self):
        return self._ulx_ref_data
        
    def d_ulx(self, l, x, order, section=-1):
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
        if y >= 0:
            return self._interpolate(y, self._vly_data[l,:,:], self._vly_section_edges)
        else:
            return self._interpolate(-y, self._vly_data[l,:,:], self._vly_section_edges) * _even_odd_sign(l)

    def check_vly(self):
        vly_max = self._vly_ref_max[2]
        vly_ref = numpy.array([ (_data[0], _data[1], abs(self.vly(int(_data[0]-1), _data[1])-_data[3])/vly_max ) for _data in self._vly_ref_data[ self._vly_ref_data[:,2]==0]])
        return(vly_ref)

    def get_d_vly_ref(self):
        return self._vly_ref_data
        
    def d_vly(self, l, y, order):
        if y >= 0:
            return self._interpolate_derivative(y, order, self._vly_data[l,:,:], self._vly_section_edges)
        else:
            return self._interpolate_derivative(-y, order, self._vly_data[l,:,:], self._vly_section_edges) * _even_odd_sign(l + order)


    def compute_Tnl(self, n):
        """
        Compute transformation matrix
        :param n: array-like or int   index (indices) of Matsubara frequencies
        :return: a 2d array of results
        """
        if isinstance(n, int):
            num_n = 1
            o_vec = 2*numpy.array([n], dtype=float)
        elif (isinstance(n, numpy.ndarray) and numpy.issubdtype(n.dtype, np.integer)) or (isinstance(n, list) and numpy.all([type(e) == int for e in n]) ):
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
        Tnl_tail = _compute_Tnl_tail(w_vec, self._statistics, deriv_x1)
        Tnl_tail_without_last_two = _compute_Tnl_tail(w_vec, self._statistics, deriv_x1[:, :-2])
        for i in range(len(n)):
            if self._statistics == 'B' and n[i] == 0:
                continue
            for l in range(self.dim()):
                if numpy.abs((Tnl_tail[i, l] - Tnl_tail_without_last_two[i, l])/Tnl_tail[i, l]) < 1e-10:
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
            _compute_Tnl_high_freq(mask, w_vec, deriv0, deriv1, x0, x1, result)

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
                result[i, l] = Tnl_tail[i, l]

        return result

    def num_sections_x(self):
        return self._ulx_data.shape[1]

    @property
    def section_edges_x(self):
        return self._ulx_section_edges

    def num_sections_y(self):
        return self._vly_data.shape[1]

    @property
    def section_edges_y(self):
        return self._vly_section_edges

    def _interpolate(self, x, data, section_edges):
        """

        :param x:
        :param data:
        :param section_edges:
        :return:
        """
        section_idx = min(bisect.bisect_right(section_edges, x)-1, len(section_edges)-2)
        return self._interpolate_impl(x - section_edges[section_idx], data[section_idx, :])

    def _interpolate_all_l(self, x, data, section_edges):
        """

        :param x:
        :param data:
        :param section_edges:
        :return:
        """
        section_idx = min(bisect.bisect_right(section_edges, x)-1, len(section_edges)-2)
        return self._interpolate_all_l_impl(x - section_edges[section_idx], data[section_idx, :, :])

    def _interpolate_derivative(self, x, order, data, section_edges, section=-1):
        """

        :param x:
        :param order:
        :param data:
        :param section_edges:
        :param section: the index of section. if section = -1, the index is determined by binary search
        :return:
        """
        section_idx = section if section >= 0 else min(bisect.bisect_right(section_edges, x)-1, len(section_edges)-2)
        coeffs = self._differentiate_coeff(data[section_idx, :], order)
        return self._interpolate_impl(x - section_edges[section_idx], coeffs)

    def _interpolate_derivatives(self, x, data, section_edges, section=-1):
        """

        :param x:
        :param data:
        :param section_edges:
        :param section: the index of section. if section = -1, the index is determined by binary search
        :return:
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
        """
        :param dx: coordinate where interpolated value is evalued
        :param coeffs: expansion coefficients
        :return:  interpolated value
        """
        value = 0.0
        dx_power = 1.0
        for p in range(len(coeffs)):
            value += dx_power * coeffs[p]
            dx_power *= dx

        return value

    def _interpolate_all_l_impl(self, dx, coeffs):
        """
        :param dx: coordinate where interpolated value is evalued
        :param coeffs: expansion coefficients (p, l)
        :return:  interpolated value
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
        Compute expansion coefficients after differentiations
        :param coeffs: coefficients
        :param order: order of differentiation (1 is the first derivative)
        :return:
        """
        k = len(coeffs)
        coeffs_deriv = numpy.array(coeffs)
        for o in range(order):
            for p in range(k-1-o):
                coeffs_deriv[p] = (p+1) * coeffs_deriv[p+1]
            coeffs_deriv[k-1-o] = 0

        return coeffs_deriv


def _composite_leggauss(deg, section_edges):
    """
    Composite Gauss-Legendre quadrature.

    :param deg: Number of sample points and weights. It must be >= 1.
    :param section_edges: array_like
                          1-D array of the two end points of the integral interval
                          and breaking points in ascending order.
    :return ndarray, ndarray: sampling points and weights
    """
    x_loc, w_loc = numpy.polynomial.legendre.leggauss(deg)

    ns = len(section_edges)-1
    x = []
    w = []
    for s in range(ns):
        dx = section_edges[s+1] - section_edges[s]
        x0 = section_edges[s]
        x.extend(((dx/2)*(x_loc+1)+x0).tolist())
        w.extend((w_loc*(dx/2)).tolist())

    return numpy.array(x), numpy.array(w)


class transformer(object):
    def __init__(self, basis, beta):
        section_edges_positive_half = numpy.array(basis.section_edges_x)
        section_edges = numpy.setxor1d(section_edges_positive_half, -section_edges_positive_half)
        self._dim = basis.dim()
        self._beta = beta
        self._x, self._w = _composite_leggauss(24, section_edges)

        nx = len(self._x)
        self._u_smpl = numpy.zeros((nx, self._dim))
        for ix in range(nx):
            self._u_smpl[ix, :] = self._w[ix] * basis.ulx_all_l(self._x[ix])

    def compute_gl(self, gtau, nl):
        assert nl <= self._dim

        nx = len(self._x)
        gtau_smpl = numpy.zeros((1, nx), dtype=complex)
        for ix in range(nx):
            gtau_smpl[0, ix] = gtau(0.5 * (self._x[ix] + 1) * self._beta)

        return numpy.sqrt(self._beta / 2) * numpy.dot(gtau_smpl[:, :], self._u_smpl[:, 0:nl]).reshape((nl))


if __name__ == '__main__':
    import argparse
    import os
    import matplotlib.pyplot as plt
    
    parser = argparse.ArgumentParser(
        prog='load.py',
        description='Read hdf5 file.',
        epilog='end',
        add_help=True,
    )

    parser.add_argument('-i', '--input', action='store', dest='inputfile',
                        type=str, choices=None,
                        required=True,
                        help=('Path to input file.'),
                        metavar=None)
    parser.add_argument('-p', '--prefix', action='store', dest='prefix',
                        type=str, choices=None,
                        default='',
                        help=('Data will be stored in this HF5 group.'),
                        metavar=None)
 

    args = parser.parse_args()

    if os.path.exists(args.inputfile):
        rb = basis(args.inputfile, args.prefix)
    else:
        print("Input file does not exist.")
        exit(-1)

    rb.check_ulx()
    exit()
        
    xvec = numpy.linspace(-1, 1, 1000)
    markers = ['o', 's', 'x', '+', 'v']
    ls = ['-', '--', ':']
    colors = ['r', 'b', 'g', 'k']
    idx=0

    plt.figure(1)
    plt.plot([rb.sl(l)/rb.sl(0) for l in range(rb.dim())], marker='+', linestyle='-', color=colors[idx])

    plt.figure(1)
    plt.xlabel('$l$')
    plt.ylabel('$s_l/s_0$')
    plt.legend()
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig('sl.pdf')

    plt.figure(2)
    for l in [rb.dim()-1]:
        print(rb.ulx(l,0.0))
        exit(0)  
        plt.plot(xvec, numpy.array([rb.ulx(l,x) for x in xvec]), marker='', linestyle='-', color='r')

    plt.figure(3)
    for l in [0, 1, 4, 6]:
        plt.plot(xvec, numpy.array([rb.vly(l,x) for x in xvec])/rb.vly(l,1), marker='', linestyle='-', color=colors[idx], label='l='+str(l))

    plt.figure(4)
    for l in [rb.dim()-1]:
        plt.plot(xvec, numpy.array([rb.d_vly(l,x,0) for x in xvec]), marker='', linestyle='-', color=colors[idx], label='l='+str(l))

    plt.figure(2)
    plt.xlabel('$x$')
    plt.ylabel('$u_l(x)$')
    #plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig('ulx.pdf')

    plt.figure(3)
    plt.xlabel('$y$')
    plt.ylabel('$v_l(y)$')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig('vly.pdf')

    plt.figure(4)
    plt.xlabel('$y$')
    plt.ylabel('$v^{(3)}_l(y)$')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig('d3_vly.pdf')
