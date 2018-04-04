import numpy
import h5py

def _even_odd_sign(l):
    return 1 if l%2==0 else -1

class basis(object):
    def __init__(self, file_name):
        
        #with open(file_name, 'r') as f:
        f = h5py.File(file_name, "r")

        self._Lambda = f['info/Lambda'].value
        self._dim = f['info/dim'].value
        self._statistics = f['info/statistics'].value

        self._sl = f['sl'].value

        self._ulx_data = f['ulx/data'].value
        self._ulx_section_edges = f['ulx/section_edges'].value
        assert self._ulx_data.shape[0] == self._dim
        assert self._ulx_data.shape[1] == f['ulx/ns'].value
        assert self._ulx_data.shape[2] == f['ulx/np'].value

        self._vly_data = f['vly/data'].value
        self._vly_section_edges = f['vly/section_edges'].value
        assert self._vly_data.shape[0] == self._dim
        assert self._vly_data.shape[1] == f['vly/ns'].value
        assert self._vly_data.shape[2] == f['vly/np'].value


    def dim(self):
        return self._dim

    def sl(self, l):
        return self._sl[l]
    
    def ulx(self, l, x):
        if x >= 0:
            return self._interpolate(x, self._ulx_data[l,:,:], self._ulx_section_edges)
        else:
            return self._interpolate(-x, self._ulx_data[l,:,:], self._ulx_section_edges) * _even_odd_sign(l)

    def vly(self, l, y):
        if y >= 0:
            return self._interpolate(y, self._vly_data[l,:,:], self._vly_section_edges)
        else:
            return self._interpolate(-y, self._vly_data[l,:,:], self._vly_section_edges) * _even_odd_sign(l)

    def compute_Tnl(self):
        pass

    def _interpolate(self, x, data, section_edges):
        """

        :param x:
        :param data:
        :param section_edges:
        :return:
        """
        section_idx = min(numpy.searchsorted(section_edges, x)-1, len(section_edges)-1)
        return self._interpolate_impl(x - section_edges[section_idx], data[section_idx,:])

    def _interpolate_derivative(self, x, order, data, section_edges):
        """

        :param x:
        :param order:
        :param data:
        :param section_edges:
        :return:
        """
        section_idx = numpy.searchsorted(section_edges, x)
        coeffs = self._differentiate_coeff(data[section_idx,:], order)
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
        coeffs_deriv = numpy.zeros_like(coeffs)
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

    args = parser.parse_args()

    if os.path.exists(args.inputfile):
        rb = basis(args.inputfile)
    else:
        print("Input file does not exist.")
        exit(-1)
    
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
        plt.plot(xvec, numpy.array([rb.ulx(l,x) for x in xvec]), marker='', linestyle='-', color='r')

    plt.figure(3)
    for l in [0, 1, 4, 6]:
        plt.plot(xvec, numpy.array([rb.vly(l,x) for x in xvec])/rb.vly(l,1), marker='', linestyle='-', color=colors[idx], label='l='+str(l))
        
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
