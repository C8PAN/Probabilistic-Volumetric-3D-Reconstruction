// This is brl/bseg/boxm2/ocl/pro/processes/boxm2_ocl_update_occupancy_belief_process.cxx
#include <bprb/bprb_func_process.h>
//:
// \file
// \brief  A process for updating the scene using belief pro. eq.
//
// \author Ali Osman Ulusoy
// \date Mar 25, 2015

#include <vcl_fstream.h>
#include <vcl_algorithm.h>
#include <boxm2/boxm2_scene.h>
#include <boxm2/boxm2_block.h>
#include <boxm2/boxm2_data_base.h>
#include <boxm2/ocl/boxm2_ocl_util.h>
#include <boxm2/boxm2_util.h>
#include <vil/vil_image_view.h>

#include <vil/vil_new.h>
#include <vpl/vpl.h> // vpl_unlink()

#include <boxm2/ocl/algo/boxm2_ocl_camera_converter.h>
#include <boxm2/ocl/algo/boxm2_ocl_update_belief.h>

//brdb stuff
#include <brdb/brdb_value.h>

//directory utility
#include <vcl_where_root_dir.h>
#include <bocl/bocl_device.h>
#include <bocl/bocl_kernel.h>
#include <vul/vul_timer.h>

namespace boxm2_ocl_update_belief_process_globals
{
  const unsigned int n_inputs_  = 11;
  const unsigned int n_outputs_ = 0;
}

bool boxm2_ocl_update_belief_process_cons(bprb_func_process& pro)
{
  using namespace boxm2_ocl_update_belief_process_globals;

  //process takes 9 inputs (of which the four last ones are optional):
  vcl_vector<vcl_string> input_types_(n_inputs_);
  input_types_[0] = "bocl_device_sptr";
  input_types_[1] = "boxm2_scene_sptr";
  input_types_[2] = "boxm2_opencl_cache_sptr";
  input_types_[3] = "vpgl_camera_double_sptr";      //input camera
  input_types_[4] = "vil_image_view_base_sptr";     //input image
  input_types_[5] = "unsigned";                     //image index, starts from 0
  input_types_[6] = "vil_image_view_base_sptr";     //mask image view
  input_types_[7] = "float";                        //variance value? if 0.0 or less, then use variable variance
  input_types_[8] = "bool";                         //update app models
  input_types_[9] = "bool";                         //update occ models

  // process has no outputs
  vcl_vector<vcl_string>  output_types_(n_outputs_);
  bool good = pro.set_input_types(input_types_) && pro.set_output_types(output_types_);

  brdb_value_sptr empty_mask = new brdb_value_t<vil_image_view_base_sptr>(new vil_image_view<unsigned char>(1,1));
  brdb_value_sptr def_var    = new brdb_value_t<float>(-1.0f);
  brdb_value_sptr def_update_app    = new brdb_value_t<bool>(true);
  brdb_value_sptr def_update_occ    = new brdb_value_t<bool>(true);

  pro.set_input(6, empty_mask);
  pro.set_input(7, def_var);
  pro.set_input(8, def_update_app);
  pro.set_input(9, def_update_occ);

  return good;
}

bool boxm2_ocl_update_belief_process(bprb_func_process& pro)
{
  using namespace boxm2_ocl_update_belief_process_globals;

  //sanity check inputs
  if ( pro.n_inputs() < n_inputs_ ) {
    vcl_cout << pro.name() << ": The input number should be " << n_inputs_<< vcl_endl;
    return false;
  }

  //get the inputs
  unsigned int i = 0;
  bocl_device_sptr         device       = pro.get_input<bocl_device_sptr>(i++);
  boxm2_scene_sptr         scene        = pro.get_input<boxm2_scene_sptr>(i++);
  boxm2_opencl_cache_sptr  opencl_cache = pro.get_input<boxm2_opencl_cache_sptr>(i++);
  vpgl_camera_double_sptr  cam          = pro.get_input<vpgl_camera_double_sptr>(i++);
  vil_image_view_base_sptr img          = pro.get_input<vil_image_view_base_sptr>(i++);
  unsigned                 image_idx    = pro.get_input<unsigned>(i++);
  vil_image_view_base_sptr mask_sptr    = pro.get_input<vil_image_view_base_sptr>(i++);
  float                    mog_var      = pro.get_input<float>(i++);
  bool                     update_app  = pro.get_input<bool>(i++);
  bool                     update_occ  = pro.get_input<bool>(i++);

  vul_timer t;

  t.mark();
  boxm2_ocl_update_belief::update(scene, device, opencl_cache, cam, img, image_idx, mask_sptr, mog_var, update_app, update_occ);

  vcl_cout<<"Update belief process took "<<t.all() << " msec."<<vcl_endl;
  return true;
}
