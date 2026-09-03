"""
submission/setup.py — builds the optional C++ indexing extension.

The Dockerfile, .github/workflows/conformance.yml and course staff's
grading image all detect this file and run

    cd submission && python setup.py build_ext --inplace

at *image build* time, never inside build_index(). That matters: the
assignment charges everything build_index() does against the index-build
efficiency metric, and a one-time compile is not indexing work
(docs/SUBMISSION_INTERFACE.md, "Compiled extensions").

THIS SCRIPT MUST NEVER EXIT NON-ZERO.

The workflow step that runs it has no `continue-on-error`, so a failure
here does not merely skip the speedup — it fails the whole conformance
job and the Docker image build. `Extension(optional=True)` alone is not
enough for that: it only covers the compiler failing on the extension
itself, and does nothing if setuptools or Cython cannot even be imported,
because the script dies before setup() is ever called. That is exactly
what happened on the first push of this extension — Python 3.12 dropped
setuptools from ensurepip, so the CI interpreter had pip but no
setuptools, and setup.py died with an ImportError traceback.

So every stage below is wrapped, and every failure path ends in a clear
message and exit code 0. submission/indexer.py imports the extension
inside a try/except and falls back to a pure-Python path that produces a
byte-identical index, so "no extension" is a slower build, never a
different one or a broken submission.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PYX = os.path.join(HERE, "_tokenizer.pyx")
CPP = os.path.join(HERE, "_tokenizer.cpp")


def skip(reason):
    """Report why the extension is not being built, and exit successfully."""
    print(f"[a1] optional C++ extension not built: {reason}")
    print("[a1] submission/indexer.py will use its pure-Python fallback, "
          "which produces an identical index.")
    raise SystemExit(0)


try:
    from setuptools import Extension, setup
except Exception as exc:                       # noqa: BLE001 - must not raise
    skip(f"setuptools unavailable ({exc!r})")

if sys.platform == "win32":
    extra_compile_args = ["/O2"]
else:
    extra_compile_args = ["-O3", "-std=c++11"]


def make_extension(sources):
    return Extension(
        "_tokenizer",
        sources,
        language="c++",
        extra_compile_args=extra_compile_args,
        # Covers the remaining case: sources are fine and the toolchain is
        # present, but the compile itself fails. setuptools then warns and
        # continues instead of raising.
        optional=True,
    )


ext_modules = []
try:
    from Cython.Build import cythonize
except Exception:
    # No Cython. Use a pre-generated _tokenizer.cpp if one was shipped;
    # otherwise there is nothing to build.
    if os.path.exists(CPP):
        ext_modules = [make_extension([CPP])]
    else:
        skip("Cython is not installed and no pre-generated _tokenizer.cpp "
             "was shipped")
else:
    if not os.path.exists(PYX):
        skip(f"{PYX} is missing")
    try:
        ext_modules = cythonize([make_extension([PYX])],
                                language_level=3, quiet=True)
    except Exception as exc:                   # noqa: BLE001 - must not raise
        skip(f"cythonize failed ({exc!r})")

try:
    setup(
        name="a1_fast_tokenizer",
        ext_modules=ext_modules,
        zip_safe=False,
    )
except SystemExit as exc:
    # setup() exits non-zero on a build failure that `optional` did not
    # absorb (a linker error, say). Downgrade it: a missing extension is
    # not a reason to fail the image build.
    if exc.code:
        skip(f"build_ext failed with exit code {exc.code}")
    raise
except Exception as exc:                       # noqa: BLE001 - must not raise
    skip(f"build_ext raised ({exc!r})")
