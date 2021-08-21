# ti_nbody
NBody Simulation written in Taichi 

## Install Taichi

To make sure taichi is working, the easiest thing todo is 

```
python3 -m pip install taichi
```

We want to install this local *ti-nbody* package

```
pip install .
```

Then for example you can run my Nbody program using

```
python ./examples/hello_nbody.py
```

## Files

* `n_body.py` Contains both *O(N^2)* and *O(NlogN)* n-body simulation, currently working correctly. 
* `hof.py` A buggy Higher Order Function Kerner which we are trying to debug. 


