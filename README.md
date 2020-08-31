# OneACPose - Relative Pose from Deep Learned Depth and a Single Affine Correspondence

Code for our ECCV 2020 paper:
[Relative Pose from Deep Learned Depth and a Single Affine Correspondence](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123570613.pdf)
Ivan Eichhardt, Daniel Barath; The European Conference on Computer Vision (ECCV), 2020

Access the [paper](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123570613.pdf) and [supplementary material](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123570613-supp.pdf) at ecva.net.

Cite it as
```
@InProceedings{Eichhardt_Barath_2020_ECCV,
	author = {Eichhardt, Ivan and Barath, Daniel},
	title = {Relative Pose from Deep Learned Depth and a Single Affine Correspondence},
	booktitle = {The European Conference on Computer Vision (ECCV)},
	month = {August},
	year = {2020},
	doi = {10.1007/978-3-030-58610-2_37}
}
```

Build using CMAKE
-----------------

See [BUILD](BUILD.md) for dependencies and instructions to build our C++ library and sample projects.

Python bindings
---------------

To install Python bindings (package called "pyoneacpose") use the following commands:

```shell
$ git clone --recursive https://github.com/eivan/one-ac-pose.git
$ pip install ./one-ac-pose
```

To use the package, launch python and import "pyoneacpose"
```shell
$ python
```
```python
>>> import pyoneacpose
```

Usage
-----

For python:

- Install Python bindings (see above)
- Open Jupyter Notebook from the root of the cloned repository
- Launch either of the example scripts

Example Jupyter Notebooks:
--------------------------

- [Piccadilly scene](example_piccadilly.ipynb) from the 1DSfM dataset
- [Fountain scene](example_fountain.ipynb) from the Strecha dataset
