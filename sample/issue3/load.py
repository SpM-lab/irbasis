import numpy
import h5py

class basis(object):
    def __init__(self, file_name, path):
        with open(file_name, 'r') as f:
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
        return _interpolate(x, self._ulx_data[l,:,:], self._ulx_section_edges)

    def vly(self, l, y):
        return _interpolate(y, self._vly_data[l,:,:], self._vly_section_edges)

    def compute_Tnl(self):
        pass

    def _interpolate(self, x, data, section_edges):
        """

        :param x:
        :param data:
        :param section_edges:
        :return:
        """
        section_idx = numpy.searchsorted(section_edges, x)
        return _interpolate_impl(x - section_edges[section_idx], data[section_idx,:])

    def _interpolate_derivative(self, x, order, data, section_edges):
        """

        :param x:
        :param order:
        :param data:
        :param section_edges:
        :return:
        """
        section_idx = numpy.searchsorted(section_edges, x)
        coeffs = _differentiate_coeff(data[section_idx,:], order)
        return _interpolate_impl(x - section_edges[section_idx], coeffs)

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
