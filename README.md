# Probabilistic Volumetric 3D Reconstruction

This repository implements a probabilistic volumetric 3D reconstruction algorithm. The repo builds on the VXL library. 

If you use this software please cite the following publication:
```
@inproceedings{3dv2015,
  title = {Towards Probabilistic Volumetric Reconstruction using Ray Potentials},
  author = {Ulusoy, Ali Osman and Geiger, Andreas and Black, Michael J.},
  booktitle = {3D Vision (3DV), 2015 3rd International Conference on},
  pages = {10-18},
  address = {Lyon},
  month = oct,
  year = {2015}
}
```

[![Towards Probabilistic Volumetric Reconstruction using Ray Potentials](http:////imgur.com/cmMvzIF
)](https://www.youtube.com/watch?v=NGj9sGaeOVY)

## Requirements:
- [cmake](http://cmake.org) 
- OpenCL/OpenGL
- Nvidia GPU with compute capability at least 3.0, see https://en.wikipedia.org/wiki/CUDA#Supported_GPU)

## Compilation:
```bash
mkdir vxl_build_directory
cd vxl_build_directory
cmake ../vxl/ -DCMAKE_BUILD_TYPE=Release
make -j -k
```

## Running the reconstruction code

## Visualizing the 3D models

## Exporting point clouds
