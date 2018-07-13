from __future__ import print_function

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)

import numpy
import irbasis


if __name__ == '__main__':
    Lambda = 1000.0
    statistics = 'F' # Fermion. use 'B' for bosons
    basis = irbasis.load(statistics,  Lambda)
    linst = ['solid','dashed','dashdot','dotted']

    mi = numpy.linspace(-1,1,10000)

    #plt.figure(1, figsize=(8,6))
    #plt.figure(2, figsize=(8,6))
    
    idx = 0
    for l in [0,1,2]:
        plt.figure(1)
        plt.plot(mi,numpy.array([basis.ulx(l,x) for x in mi]),linestyle = linst[idx],label = "$l = {}$".format(l))
        
        plt.figure(2)
        plt.plot(mi,numpy.array([basis.vly(l,y) for y in mi]),linestyle = linst[idx],label = "$l = {}$".format(l))

        idx += 1
        
    plt.figure(1)   
    plt.xlim(-1.1,1.1)
    plt.xlabel(r'$x$',fontsize = 18)
    plt.tick_params(labelsize=21)
    plt.ylabel(r'$u_l^\mathrm{{{0}}}(x)$'.format(statistics),fontsize = 21)
    plt.legend(frameon=False,fontsize = 21)
    plt.grid()
    plt.tight_layout()
    plt.savefig('u.png')  
    
    
    plt.figure(2)   
    plt.xlim(-1.1,1.1)
    plt.xlabel(r'$y$',fontsize = 18)
    plt.tick_params(labelsize=21)
    plt.ylabel(r'$v_l^\mathrm{{{0}}}(y)$'.format(statistics),fontsize = 21)
    plt.legend(frameon=False,fontsize = 21)
    plt.grid()
    plt.tight_layout()
    plt.savefig('v.png')  
    #plt.show()
