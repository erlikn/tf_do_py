from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules = cythonize('dataset_prepare_cy.pyx'))