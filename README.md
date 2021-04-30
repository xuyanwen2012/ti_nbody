# ti_nbody
NBody Simulation written in Taichi 

## Install Taichi

To make sure taichi is working, the easiest thing todo is 

```
python3 -m pip install taichi
```

Then for example you can run my Nbody program using

```
python n_body.py 10             # For simulating 2^10 particles
```

## Files

* `n_body.py` Contains both *O(N^2)* and *O(NlogN)* n-body simulation, currently working correctly. 
* `hof.py` A buggy Higher Order Function Kerner which we are trying to debug. 


