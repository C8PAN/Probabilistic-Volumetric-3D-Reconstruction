#Example script to reconstruct a scene.
import random,re;
from boxm2_scene_adaptor import *
from boxm2_adaptor import *
from bbas_adaptor import remove_from_db
from glob import glob

def deleteAllFiles(boxm2_dir):
    for file in glob(boxm2_dir + "/*.bin"):
        os.remove(file)

def deleteAppModelFiles(boxm2_dir):
    pattern1 = "boxm2_mog?";
    pattern2 = "boxm2_viewdep?";
    pattern3 = "boxm2_num_obs?"
    pattern4 = "aux?"
    for f in os.listdir(boxm2_dir + "/"):
        if re.search(pattern1, f) or re.search(pattern2, f) or re.search(pattern3, f) or re.search(pattern4, f):
            print f;
            os.remove(os.path.join(boxm2_dir, f))

def deleteMsgFiles(boxm2_dir):
    pattern1 = "boxm2_app_msg?"
    pattern3 = "boxm2_log_msg?"
    pattern2 = "boxm2_sum?"
    for f in os.listdir(boxm2_dir + "/"):
        if  re.search(pattern1, f) or re.search(pattern2, f) or re.search(pattern3, f) :
            print f;
            os.remove(os.path.join(boxm2_dir, f))

def deleteTmpFiles(boxm2_dir):
    pattern4 = "aux?"
    for f in os.listdir(boxm2_dir + "/"):
        if re.search(pattern4, f):
            print f;
            os.remove(os.path.join(boxm2_dir, f))

boxm2_batch.register_processes();
boxm2_batch.register_datatypes();

class dbvalue:
  def __init__(self, index, type):
    self.id = index    # unsigned integer
    self.type = type   # string

#We assume the following scene structure
#model_dir/           : parent folder
#model_dir/boxm2_site : contains the 3d model
#model_dir/imgs       : contains the images
#model_dir/cams_krt   : contains the cameras
model_dir = "/data/3dv15_experiments/3d_scenes/downtown/"
boxm2_dir = model_dir + "/boxm2_site"

if not os.path.isdir(boxm2_dir + '/'):
    print "Invalid Site Dir"
    sys.exit(-1);

imgs = glob(model_dir + "/imgs/*.png")
cams = glob(model_dir + "/cams_krt/*.txt")
imgs.sort()
cams.sort()
if len(imgs) != len(cams) :
    print "CAMS NOT ONE TO ONE WITH IMAGES"
    sys.exit();

#the folder where expected images are saved
expected_img_dir = boxm2_dir +  "/expectedImgs"
if not os.path.isdir(expected_img_dir + '/'):
    os.mkdir(expected_img_dir + '/');


list = range(0,len(imgs),1);

#delete *.bin files in the folder
deleteAllFiles(boxm2_dir)

scene = boxm2_scene_adaptor(boxm2_dir + "/scene.xml", "gpu0");

#choosen a cam to render the expected images
render_cam = load_perspective_camera(cams[0]);

#do a few iterations over the images while refining the octree 
NUM_ITER = 4
for iter in range(NUM_ITER):
    random.shuffle(list);

    #do linear randomized pass
    for idx, image_number in enumerate(list):

        #refine the octree every 10th iteration
        if idx % 10 == 9:
            print "REFINING OCTREE"
            scene.refine_bp(0.3)           

        #re-initialize appearance beliefs once in a while
        #improves convergence
        if iter < 1 and (idx == 75 or idx == 135):
            scene.clear_app_models()

        print 'ITERATION %d' % idx
        print "Updating with img: %s and cam: %s" % (imgs[image_number],cams[image_number])

        [img, ni, nj] = load_image(imgs[image_number]);
        cam = load_perspective_camera(cams[image_number]);

        scene.update(cam, img, image_number, update_occupancy=True, update_app=True, mask=None, var=0.08)

        exp_img = scene.render(render_cam,ni,nj);
        exp_img_byte = convert_image(exp_img);
        save_image(exp_img_byte, expected_img_dir + "/expected_image_%05d_%05d.png"%(iter,idx));

        remove_from_db(img)
        remove_from_db(cam)
        remove_from_db(exp_img)
        remove_from_db(exp_img_byte)


    scene.write_cache()
    scene.clear_cache()

    deleteMsgFiles(boxm2_dir)
    deleteTmpFiles(boxm2_dir)
    deleteAppModelFiles(boxm2_dir)

#keeping the octree structure fixed, iterate over the images a few times
for iter in range(NUM_ITER,NUM_ITER+4):

    random.shuffle(list);

    #do linear randomized pass
    for idx, image_number in enumerate(list):

        #re-initialize appearance beliefs once in a while
        #improves convergence
        if iter == NUM_ITER and (idx == 75 or idx == 135):
            scene.clear_app_models()

        print 'ITERATION %d' % idx
        print "Updating with img: %s and cam: %s" % (imgs[image_number],cams[image_number])

        [img, ni, nj] = load_image(imgs[image_number]);
        cam = load_perspective_camera(cams[image_number]);

        scene.update(cam, img, image_number, update_occupancy=True, update_app=True, mask=None, var=0.08)

        exp_img = scene.render(render_cam,ni,nj);
        exp_img_byte = convert_image(exp_img);
        save_image(exp_img_byte, expected_img_dir + "/expected_image_%05d_%05d.png"%(iter,idx));

        remove_from_db(img)
        remove_from_db(cam)
        remove_from_db(exp_img)
        remove_from_db(exp_img_byte)

    scene.write_cache()

    deleteTmpFiles(boxm2_dir)

raw_input("ENTER TO EXIT...")
boxm2_batch.clear()
