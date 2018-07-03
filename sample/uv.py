from __future__ import print_function
#from builtins import range

import numpy as np
import matplotlib.pyplot as plt

import numpy
import irbasis
plt.rc('text', usetex=True)



if __name__ == '__main__':
    # Fermion, Lambda = 1000.0
    #beta = 100.0
    Lambda = 1000.0
    basis = irbasis.load('F',  Lambda)
    linst = ['solid','dashed','dashdot','dotted']

    mi = numpy.linspace(-1,1,10000)
    
    for l in [0,1,2]:
        plt.figure(1)
        plt.plot(mi,numpy.array([basis.ulx(l,x) for x in numpy.linspace(-1,1,10000)]),linestyle = linst[l],label = "$l = {}$".format(l))
        
        plt.figure(2)
        plt.plot(mi,numpy.array([basis.vly(l,y) for y in numpy.linspace(-1,1,10000)]),linestyle = linst[l],label = "$l = {}$".format(l))
        
    plt.figure(1)   
    plt.xlim(-1.1,1.1)
    plt.xlabel(r'$x$',fontsize = 18)
    plt.tick_params(labelsize=21)
    plt.ylabel(r'$u^{0}(x)$'.format(r"{\rm{F}}"),fontsize = 21)
    plt.legend(frameon=False,fontsize = 21)
    plt.tight_layout()
    plt.savefig('u'+'.pdf')  
    
    
    plt.figure(2)   
    plt.xlim(-1.1,1.1)
    plt.xlabel(r'$y$',fontsize = 18)
    plt.tick_params(labelsize=21)
    #plt.plot(mi,Tn.imag[0,:])
    plt.ylabel(r'$v^{0}(y)$'.format(r"{\rm{F}}"),fontsize = 21)
    plt.legend(frameon=False,fontsize = 21)
    plt.tight_layout()
    plt.savefig('v'+'.pdf')  
    plt.show()

    
    
    
