"""
submission/setup.py — builds the optional C++ indexing extension.

The Dockerfile, .github/workflows/conformance.yml and course staff's
grading image all detect this file and run

    cd submission && python setup.py build_ext --inplace

at *image build* time, never inside build_index(). That matters: the
assignment charges everything build_index() does against the index-build
efficiency metric, and a one-time compile is not indexing work
(docs/SUBMISSION_INTERFACE.md, "Compiled extensions").

The extension is declared `optional=True` on purpose. If Cython or a
compiler is unavailable, or the build fails for any reason, setuptools
prints a warning and still exits 0 — and submission/indexer.py imports
`_tokenizer` inside a try/except, falling back to a pure-Python path that
produces a byte-identical index. So the worst case of a failed compile is
losing the speedup, never a broken submission or a failed image build.
"""
import os
import sys

from setuptools import Extension, setup

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(HERE, "_tokenizer.pyx")

if sys.platform == "win32":
    extra_compile_args = ["/O2"]
else:
    extra_compile_args = ["-O3", "-std=c++11"]

extensions = [
    Extension(
        "_tokenizer",
        [SOURCE],
        language="c++",
        extra_compile_args=extra_compile_args,
        optional=True,
    )
]

try:
    from Cython.Build import cythonize
except ImportError:
    # No Cython: fall back to a pre-generated _tokenizer.cpp if one was
    # shipped, and otherwise build nothing at all. Either way indexer.py
    # still works through its pure-Python path.
    cpp = os.path.join(HERE, "_tokenizer.cpp")
    if os.path.exists(cpp):
        extensions[0].sources = [cpp]
    else:
        extensions = []
else:
    extensions = cythonize(extensions, language_level=3, quiet=True)

setup(
    name="a1_fast_tokenizer",
    ext_modules=extensions,
    zip_safe=False,
)
