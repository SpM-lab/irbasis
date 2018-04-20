from __future__ import print_function
from builtins import range

import numpy
import h5py
import bisect
import platform

is_python3 = int(platform.python_version_tuple()[0]) == 3

def _from_bytes_to_utf8(s):
    """
    from bytes to string
    :param s:
    :return:
    """
    if is_python3 and isinstance(s, bytes):
        return s.decode('utf-8')
    else:
        return s


def _even_odd_sign(l):
    return 1 if l%2==0 else -1

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

class basis(object):
    def __init__(self, file_name, prefix=""):
        
        with h5py.File(file_name, 'r') as f:
            self._Lambda = f[prefix+'/info/Lambda'].value
            self._dim = f[prefix+'/info/dim'].value
            self._statistics = 'B' if f[prefix+'/info/statistics'].value == 0 else  'F'

            self._sl = f[prefix+'/sl'].value

            self._ulx_data = f[prefix+'/ulx/data'].value
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

    def check_ulx(self):
        ulx_max = self._ulx_ref_max[2]
        ulx_ref = numpy.array([ (_data[0], _data[1], abs(self.ulx(int(_data[0]-1), _data[1])-_data[2])/ulx_max ) for _data in self._ulx_ref_data])
        return(ulx_ref)
        
        
    def d_ulx(self, l, x, order, section=-1):
        if x >= 0:
            return self._interpolate_derivative(x, order, self._ulx_data[l, :, :], self._ulx_section_edges, section)
        else:
            return -self._interpolate_derivative(-x, order, self._ulx_data[l, :, :], self._ulx_section_edges, section) * _even_odd_sign(l)


    def vly(self, l, y):
        if y >= 0:
            return self._interpolate(y, self._vly_data[l,:,:], self._vly_section_edges)
        else:
            return self._interpolate(-y, self._vly_data[l,:,:], self._vly_section_edges) * _even_odd_sign(l)

    def check_vly(self):
        vly_max = self._vly_ref_max[2]
        vly_ref = numpy.array([ (_data[0], _data[1], abs(self.vly(int(_data[0]-1), _data[1])-_data[2])/vly_max ) for _data in self._vly_ref_data])
        return(vly_ref)

        
    def d_vly(self, l, y, order):
        if y >= 0:
            return self._interpolate_derivative(y, order, self._vly_data[l,:,:], self._vly_section_edges)
        else:
            return -self._interpolate_derivative(-y, order, self._vly_data[l,:,:], self._vly_section_edges) * _even_odd_sign(l)


    def compute_Tnl(self, n):
        """
        Compute transformation matrix
        :param n: array-like or int   index (indices) of Matsubara frequencies
        :return: a 2d array of results
        """
        if isinstance(n, int):
            num_n = 1
            o_vec = 2*numpy.array([n], dtype=float)
        elif isinstance(n, numpy.ndarray) or isinstance(n, list):
            num_n = len(n)
            o_vec = 2*numpy.array(n, dtype=float)
        else:
            raise RuntimeError("n is not an integer, list or a numpy array")

        if self._statistics == 'F':
            o_vec += 1
        w_vec = 0.5 * numpy.pi * o_vec

        num_deriv = self._ulx_data.shape[2]

        deriv0 = numpy.zeros((self.dim(), num_deriv), dtype=float)
        deriv1 = numpy.zeros((self.dim(), num_deriv), dtype=float)

        result = numpy.zeros((num_n, self.dim()), dtype=complex)

        for s in range(self.num_sections_x()):
            x0 = self._ulx_section_edges[s]
            x1 = self._ulx_section_edges[s+1]

            # Derivatives at end points
            for k in range(num_deriv):
                for l in range(self.dim()):
                    deriv0[l, k] = self.d_ulx(l, x0, k, s)
                    deriv1[l, k] = self.d_ulx(l, x1, k, s)

            # Mask based on phase shift
            mask = numpy.abs(w_vec) * (x1-x0) > 0.1 * numpy.pi

            # High frequency formula
            _compute_Tnl_high_freq(mask, w_vec, deriv0, deriv1, x0, x1, result)

            # low frequency formula
            deg = 2 * num_deriv
            x_smpl, weight = numpy.polynomial.legendre.leggauss(deg)
            x_smpl = 0.5 * (x_smpl + 1) * (x1 - x0) + x0
            weight *= (x1 - x0)/2

            smpl_vals = numpy.zeros((deg, self.dim()))
            for l in range(self.dim()):
                for ix in range(deg):
                    smpl_vals[ix, l] = self.ulx(l, x_smpl[ix])
            mask_not = numpy.logical_not(mask)
            exp_iwx = numpy.exp(numpy.einsum('w,x->wx', 1J * w_vec[mask_not], x_smpl))
            result[mask_not, :] += numpy.einsum('wx,x,xl->wl', exp_iwx, weight, smpl_vals)

        for l in range(self.dim()):
            if l % 2 == 0:
                result[:, l] = result[:, l].real
            else:
                result[:, l] = 1J * result[:, l].imag

        result = numpy.einsum('w,wl->wl', numpy.sqrt(2.) * numpy.exp(1J * w_vec), result)

        return result

    def num_sections_x(self):
        return self._ulx_data.shape[1]

    def num_sections_y(self):
        return self._vly_data.shape[1]

    def _interpolate(self, x, data, section_edges):
        """

        :param x:
        :param data:
        :param section_edges:
        :return:
        """
        section_idx = min(bisect.bisect_right(section_edges, x)-1, len(section_edges)-2)
        return self._interpolate_impl(x - section_edges[section_idx], data[section_idx,:])

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
            for p in range(k-1):
                coeffs_deriv[p] = (p+1) * coeffs_deriv[p+1]
            coeffs_deriv[k-1] = 0

        return coeffs_deriv

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
