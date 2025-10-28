import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

extensions = [
    Extension(
        "clinometer.clinometer",
        sources=["clinometer/clinometer.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-O3"],
    ),
]

setup(
    name="clinometer",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "infer_types": True,
        },
        annotate=True,
    ),
    zip_safe=False,
)
