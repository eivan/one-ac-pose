# OneACPose - Relative Pose from Deep Learned Depth and a Single Affine Correspondence

Build instructions
------------------

Required tools:

- CMake (version >= 3.11)
- Git
- C/C++ compiler (C++17)

Dependencies:

- OpenCV (tested with v4.3.0)

Optional dependencies:

- OpenMP
- DoxyGen

Note:

- CMAKE variables you can configure:
<a name="cmakevariables"></a>

  - ONEACPOSE_BUILD_DOC (ON/OFF(default))
      - Build documentation using DoxyGen
  - ONEACPOSE_USE_OPENMP (ON/OFF(default))
      - Use OpenMP for parallelization
  - ONEACPOSE_BUILD_SAMPLES (ON(defailt)/OFF)
      - Build sample project for synthetic and real-world tests
	  
Checking out the project and build it
-------------------------------------

- [Getting the project](#checkout)
- [Compiling](#compiling)

Getting the project
-------------------
<a name="checkout"></a>

Getting the sources (and the submodules):
```shell
$ git clone --recursive https://github.com/eivan/one-ac-pose.git
```

Compiling
---------
<a name="compiling"></a>

1. Make a directory for the build files to be generated.
```shell
$ mkdir build_dir
$ cd build_dir
```

2. Configure CMAKE (see the choice of [CMAKE options](#cmakevariables)).
```shell
$ cmake-gui ../src
```

3. Compile.