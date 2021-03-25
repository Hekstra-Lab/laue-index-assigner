# laue-index-assigner

This is just a simple (library)[diffgeolib.py] of useful functions and classes for interpreting laue data. 
It contains a singular (example)[assign.py] of assigning indices to a Laue image starting from poor initial geometry.

The example script should be run through the `cctbx.python` supplied by your (DIALS)[https://dials.github.io] installation.
The example has some extra dependencies which are available through pip and can probably be installed by the following command.

```bash
cctbx.python -m pip install reciprocalspaceship cvxpy
```

