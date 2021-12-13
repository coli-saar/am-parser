C++ Implementation of Fast Smatch
=================================

#This directory comes from https://github.com/Oneplus/tamr

We use the *oracle smatch score*
to evaluate each generated alignment,
and we found the original smatch script
greatly slowed down our program.
So we use `Cython` to re-implement the smatch
script.

## Compilation

run `python setup.py build` in the `amr_aligner/smatch`
folder. It will generate a dynamic library `_smatch.so` 
under the `build/lib.${arch}-2.7/` folder.
Move the dynamic library into `amr_aligner/smatch`
and it will do the work.

## Smatch Version

2.0.2