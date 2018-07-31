[![Build Status](https://travis-ci.org/SpM-lab/irbasis.svg?branch=master)](https://travis-ci.org/SpM-lab)

irbasis
======
Open-source database and software for intermediate-representation basis functions of imaginary-time Green's function and Python and C++ libraries

Detailed instructions are available [online](https://github.com/SpM-lab/irbasis/wiki).
Please also check [our citation policy](https://github.com/SpM-lab/irbasis/wiki/Citation-policy).

Below we will briefly describe this software.

# Table of Contents
- [Structure](#structure)
- [Installation](#installation)
- [Usage](#usage)

## Structure
We briefly describe files constituting this software below.

* c++/irbasis.hpp<br>C++ library
* python/irbasis.py<br>Python library
* database/irbasis.h5<br>Database file
* sample/<br> Directory including samples in C++ and Python
* test/<br>Unit tests in C++ and Python

## Installation
### Python

You need to install only a few standard scientific libraries (such as numpy, h5py) shown in [our PyPI project page](https://pypi.org/project/irbasis/).
If you install irbasis through pip, pip will take care of these dependencies properly.

We strongly recommend to install the irbasis library using the standard Python package system.
This package contains the data file (irbasis.h5) as well.
```
python -mpip install -U pip
python -mpip install -U irbasis
```

Alternatively, we can put [irbasis.py](https://github.com/SpM-lab/irbasis/blob/master/python/irbasis.py) and [irbasis.h5](https://github.com/SpM-lab/irbasis/blob/master/database/irbasis.h5) into your working directory.
You can load irbasis and use the full functionality.

If you want run [sample Python scripts](Samples),
please also install additional Python packages (scipy, matplotlib) using the following command.

```
pip install scipy matplotlib
```


### C++

You need a C++03-compatible compiler.
The use of the C++ irbasis library requires only the HDF5 C library (not C++).

The C++ library consists of a single header file.
All what you need to do is to include [irbasis.hpp](https://github.com/SpM-lab/irbasis/blob/master/c++/irbasis.hpp) in your C++ project.
The data file [irbasis.h5](https://github.com/SpM-lab/irbasis/blob/master/database/irbasis.h5) will be read at runtime.
Please do not forget to link your executable to the HDF5 C library.


## Usage
In the following, we demonstrate how to use irbasis database.
The irbasis database is available in Python and C++.
irbasis can calculate the IR basis functions, its Fourier transform, the derivatives and corresponding singular values.

**In the following, we assume that you have installed the irbasis Python library via pip.**
If not, please modify the sample script files appropriately to specify the location of a database file (see a comment in api.py).

**Some of sample Python scripts depend on scipy and matplotlib.**

For other examples, please refer to our online document.

### Python
You can download [api.py](https://github.com/SpM-lab/irbasis/blob/master/sample/api.py)
and save it to your working directory.
Then, please run the following command

```python
python api.py
```

### C++
You can download [api.cpp](https://github.com/SpM-lab/irbasis/blob/master/sample/api.py)
and save it to your working directory.
After copying irbasis.hpp into the same directory,
you can build the sample program as follows (see [compile.sh](https://github.com/SpM-lab/irbasis/blob/master/sample/compile.sh)).

```c++
g++ api.cpp -I /usr/local/include -L /usr/local/lib -lhdf5 -DNDEBUG -O3
```

Here, we assume that the header file and the library file of the HDF5 C library are installed into "/usr/local/include" and  "/usr/local/lib", respectively.
When running the executable, irbasis.h5 must exist in your working directory.
