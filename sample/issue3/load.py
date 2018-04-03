import numpy
import h5py

class basis(object):
    def __init__(self, file_name, path):
        with open(file_name, 'r') as f:
            self.Lambda = f['info/Lambda'].value
            self.dim = f['info/dim'].value
            self.statistics = f['info/statistics'].value

            self.ulx_data = f['ulx/data'].value
            assert self.ulx_data.shape[0] == self.dim
            assert self.ulx_data.shape[1] == f['ulx/ns'].value
            assert self.ulx_data.shape[2] == f['ulx/np'].value

            self.vly_data = f['vly/data'].value
            assert self.vly_data.shape[0] == self.dim
            assert self.vly_data.shape[1] == f['vly/ns'].value
            assert self.vly_data.shape[2] == f['vly/np'].value


    def dim(self):
        pass

    def sl(self):
        pass

    def ulx(self, l, x):
        pass

    def vly(self, l, y):
        pass

    def compute_Tnl(self):
        pass
