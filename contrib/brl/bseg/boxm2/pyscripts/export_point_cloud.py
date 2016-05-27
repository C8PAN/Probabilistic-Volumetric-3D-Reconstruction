import random,re;
from boxm2_scene_adaptor import *
from boxm2_adaptor import *
from bbas_adaptor import remove_from_db
from glob import glob

#import matplotlib.pyplot as plt;
boxm2_batch.register_processes();
boxm2_batch.register_datatypes();

class dbvalue:
  def __init__(self, index, type):
    self.id = index    # unsigned integer
    self.type = type   # string


#We assume the following scene structure
#model_dir/           : parent folder
#model_dir/boxm2_site : contains the 3d model
model_dir = "/data/3dv15_experiments/3d_scenes/capitol_2006/"
boxm2_dir = model_dir + "/boxm2_site"

if not os.path.isdir(boxm2_dir + '/'):
    print "Invalid Site Dir"
    sys.exit(-1);


scene = boxm2_scene_adaptor(boxm2_dir + "/scene.xml", "gpu0");
extract_cell_centers(scene.scene, scene.cpu_cache,prob_thresh = 0.05)
export_points_and_normals(scene.scene, scene.cpu_cache,boxm2_dir+"/pt.xyz")

raw_input("ENTER TO EXIT...")
boxm2_batch.clear()
