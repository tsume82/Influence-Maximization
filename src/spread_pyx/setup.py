# python setup.py build_ext --inplace

from distutils.core import setup
from Cython.Build import cythonize

setup(name='monte_carlo',
      ext_modules=cythonize("monte_carlo.pyx"))

setup(name='monte_carlo_max_hop',
      ext_modules=cythonize("monte_carlo_max_hop.pyx"))

setup(name='two_hop',
      ext_modules=cythonize("two_hop.pyx"))
