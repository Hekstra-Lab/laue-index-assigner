import reciprocalspaceship as rs

output_file = 'r-free-flags.mtz'
cell = [34.297, 45.552, 99.035, 90., 90., 90.]
sg = 19
dmin = 1.
rfraction = 0.1

h,k,l = rs.utils.generate_reciprocal_asu(cell, sg, dmin).T

ds = rs.DataSet({
        'H' : h,
        'K' : k,
        'L' : l,
    }, 
    cell=cell, 
    spacegroup=sg
).infer_mtz_dtypes().set_index(['H', 'K', 'L'])

ds = rs.utils.add_rfree(ds, rfraction)
ds.write_mtz(output_file)

from pylab import *

ds.compute_dHKL(inplace=True)
n = 50
dmin = 1.
dmax = 5.
x = np.linspace(dmin**-2., dmax**-2., n)**-0.5
y = []
for cutoff in x:
    frac = ds[ds.dHKL >= cutoff].mean()['R-free-flags']
    y.append(frac)

plt.plot(x, y, '-k')
plt.xlabel("Resolution")
plt.ylabel("Rfree Fraction")
plt.show()
