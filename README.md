# Probabilistic Volumetric 3D Reconstruction

This repository implements a probabilistic and volumetric 3D reconstruction algorithm. The algorithm takes as input images (with known camera pose and intrinsics) and generates a dense probabilistic 3D model that exposes the uncertainty in the reconstruction. Please see the video below for a short explanation and results. 

[![Towards Probabilistic Volumetric Reconstruction using Ray Potentials](https://raw.githubusercontent.com/aliosmanulusoy/vxl/master/youtube_img.png
)](https://www.youtube.com/watch?v=NGj9sGaeOVY)

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

## Requirements:
- [cmake](http://cmake.org) 
- OpenCL
- OpenGL
- Nvidia GPU with compute capability at least 2.0, see https://en.wikipedia.org/wiki/CUDA#Supported_GPU)

## Compilation:
```bash
cd /path/to/my/build/folder
cmake /path/to/vxl/source/folder -DCMAKE_BUILD_TYPE=Release
make -j -k
```
If everything compiled correctly you should see the executable /path/to/my/build/folder/bin/boxm2_ocl_render_view as well as the library /path/to/my/build/folder/lib/boxm2_batch.so 

Add python scripts to the PYTHONPATH as follows,
```bash
export PYTHONPATH=$PYTHONPATH:/path/to/my/build/folder/lib/:/path/to/vxl/source/folder/contrib/brl/bseg/boxm2/pyscripts/
```

## Running the reconstruction code

### The input

* Images: The current implementation works with intensity images. If you supply an RGB image, it will be automatically converted to intensity. 

* Cameras: Our algorithm expects camera intrinsics (`K` 3x3 matrix) and extrinsics (`[R|t]` 3x4 matrix) for each image. The projection matrix is `P = K [R | t]`. Cameras are specified in separate text files for each image. Each camera text file is formatted as follows: 
```bash
f_x    s    c_x
0     f_y   c_y
0      0     1

R_11   R_12  R_13
R_21   R_22  R_23
R_31   R_32  R_33

t_1    t_2   t_3
```
where `f_x` and `f_y` are the focal lenghts, `c_x` and `c_y` is the principal point and `s` is the skew parameter. Together, these parameters make up the intrinsics matrix `K`. The second matrix is the rotation matrix `R`. The final row is the translation vector `t`. 

**Important note**: The reconstruction scripts assume when the list of images and camera files are sorted alphabetically they form a correspondence:
```python
img_files = glob(img_folder + "/*.png")
cam_files = glob(cam_folder + "/*.txt")
img_files.sort()
cam_files.sort()

img = img_files[index]
cam = cam_files[index]
```
One way to ensure this correspondence is to name the cameras to match the images, i.e., `img_00001.png` and `img_00001.txt`. 
 
### Creating the scene file

### Reconstruction

## Visualizing the 3D models

## Exporting point clouds
