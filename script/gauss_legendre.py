import sys
from mpmath import *
from mpmath.calculus.quadrature import GaussLegendre

dps = 300

mp.dps = dps
prec = int(dps * 3.33333)
mp.pretty = False

print("""
inline
std::vector<std::pair<double,double> >
gauss_legendre_nodes(int num_nodes) {
""")

#Note: mpmath gives wrong results for degree==1! 
for degree in [4,5]:
    g = GaussLegendre(mp)
    gl = g.get_nodes(-1, 1, degree=degree, prec=prec)



    N = 3*2**(degree-1)

    print("""
    if (num_nodes == %d) {
        std::vector<std::pair<double,double> > nodes(%d);
"""%(N,N))

    for i in range(len(gl)):
        print("        nodes[{:<5}] = std::make_pair<double>({:.25}, {:.25});".format(i, float(gl[i][0]), float(gl[i][1])));

    print("""
        return nodes;
    }
""")

print("""
    throw std::runtime_error("Invalid num_nodes passed to gauss_legendre_nodes");
}
""")
