# Example Gauss fit
An example of how to perform a gaussian fit using sci-py's curve_fit with a Levenberg-Marquardt algorithm.

## Installation
To install this package, please download it and run `pip install .` in the folder where it was downloaded.


## Usage
To use this package after installing you can import and use it as such:

```
from example_gauss_fit import example_gauss_fit
import numpy as np 

x=np.array([-1,0.5,0,0.5,1])
y=np.array([0.1,.25,.5,.25,0.1])

paremeters = gauss_fit.gauss_fit(x,y)

xeval=np.linspace(-10,10,1000)
yeval=gauss_fit.gauss_eval(xeval,parameters)
```

The `parameters` object is now a dictionary with the best-fit parameters.
The `xeval` and `yeval` contain the fitted gaussian from the data.

## Dependencies

This package depends on:
```
numpy
sci-py
```
