from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

setup(
	name = 'The New algorithm for Structure Sparsity',
	version = '1.0',
	author = "Wu Tianyi",
    cmdclass = {'build_ext': build_ext},
    ext_modules = [
        Extension("prox_dp", 
                  ['prox_dp.pyx'], 
		          extra_compile_args=["-mtune=core2", "-march=core2", 
                                      "-O3", "-Wno-unused"],
                  include_dirs=[numpy.get_include()])]
)
