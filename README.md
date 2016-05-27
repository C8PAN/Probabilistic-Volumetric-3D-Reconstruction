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
- Glut
- Glew
- Nvidia GPU with compute capability at least 2.0, see https://en.wikipedia.org/wiki/CUDA#Supported_GPU)
- Python >= 2.7

## Compilation:
```bash
cd /path/to/my/build/folder
cmake /path/to/vxl/source/folder -DCMAKE_BUILD_TYPE=Release
make -j -k
```
If everything compiled correctly you should see the executable `/path/to/my/build/folder/bin/boxm2_ocl_render_view` as well as the library `/path/to/my/build/folder/lib/boxm2_batch.so` 

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
 
### Creating the scene
You can specify the dimensions of the volume of interest, minimum allowed voxel size in the octree (in world coordinates), and the prior on occupancy probability (see paper reference above) in an XML file `scene_info.xml` as follows,
```xml
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<bwm_info_for_boxm2>
<bbox maxx="1" maxy="1" maxz="1" minx="0" miny="0" minz="0" >
</bbox>
<min_octree_cell_length val="0.001">
</min_octree_cell_length>
<prior_probability val="0.01">
</prior_probability>
</bwm_info_for_boxm2>
```
Please run the following python script: 
```python
import boxm2_create_scene_scripts.py
boxm2_create_scene_scripts.create_scene('/path/to/scene_info.xml','/path/to/scene/')
```
This script should create the folder `/path/to/scene/` and an xml file called `scene.xml` in it. 

### Reconstruction
We provide the following python script to reconstruct a scene from a set of images and cameras:
```bash
/path/to/vxl/source/folder/contrib/brl/bseg/boxm2/pyscripts/reconstruct.py
```
Please follow the instructions inside the script. 

## Visualizing the 3D models
You can visualize the volumetric models using the renderer `boxm2_ocl_render_view`.
```bash
/path/to/my/build/folder/bin/boxm2_ocl_render_view -scene /path/to/scene/scene.xml
```
The renderer computes expected pixel intensities by ray-tracing the probabilistic 3d volume. You can press `d` to render depth maps and `e` to render the entropy in depth distributions. Please see the paper for details. 

![alt tag](https://raw.githubusercontent.com/aliosmanulusoy/vxl/master/teaser_img.png)

## Exporting point clouds
The probabilistic volumetric 3D model can also be visualized as a point cloud. We provide the following script that extracts a point cloud from the 3D model and exports it in XYZ format which can be visualized using [CloudCompare](http://www.danielgm.net/cc/):
```bash
/path/to/vxl/source/folder/contrib/brl/bseg/boxm2/pyscripts/export_point_cloud.py
```
This script outputs points that correspond to voxel centers. Point with very small probability are filtered for better visualization. The script also outputs the marginal occupancy belief for each point. CloudCompare can be used to visualize these probabilities as shown below:
![alt tag](https://raw.githubusercontent.com/aliosmanulusoy/vxl/master/cloud_compare.png)
