// This is brl/bseg/boxm2/ocl/pro/processes/boxm2_ocl_refine_bp_process.cxx
//:
// \file
// \brief  A process to refine the scene octree
//
// \author Ali Osman Ulusoy

#include <bprb/bprb_func_process.h>
#include <boxm2/ocl/algo/boxm2_ocl_refine.h>
#include <boxm2/ocl/boxm2_opencl_cache.h>
#include <boxm2/boxm2_scene.h>
#include <bocl/bocl_device.h>
namespace boxm2_ocl_refine_bp_process_globals
{
    const unsigned n_inputs_ = 6;
    const unsigned n_outputs_ = 1;
}

bool boxm2_ocl_refine_bp_process_cons(bprb_func_process& pro)
{
    using namespace boxm2_ocl_refine_bp_process_globals;

    //process takes 1 input
    vcl_vector<vcl_string> input_types_(n_inputs_);
    input_types_[0] = "bocl_device_sptr";
    input_types_[1] = "boxm2_scene_sptr";
    input_types_[2] = "boxm2_opencl_cache_sptr";
    input_types_[3] = "float";
    input_types_[4] = "unsigned"; //num_imgs
    input_types_[5] = "bool"; //skip msg refinement

    // process has 1 output:
    vcl_vector<vcl_string>  output_types_(n_outputs_);
    output_types_[0] = "int";  //numcells
    return pro.set_input_types(input_types_) && pro.set_output_types(output_types_);
}

bool boxm2_ocl_refine_bp_process(bprb_func_process& pro)
{
    using namespace boxm2_ocl_refine_bp_process_globals;
    if ( pro.n_inputs() < n_inputs_ ) {
        vcl_cout << pro.name() << ": The input number should be " << n_inputs_<< vcl_endl;
        return false;
    }

    //get the inputs
    unsigned i = 0;
    bocl_device_sptr        device       = pro.get_input<bocl_device_sptr>(i++);
    boxm2_scene_sptr        scene        = pro.get_input<boxm2_scene_sptr>(i++);
    boxm2_opencl_cache_sptr opencl_cache = pro.get_input<boxm2_opencl_cache_sptr>(i++);
    float                   thresh       = pro.get_input<float>(i++);
    unsigned                num_imgs     = pro.get_input<unsigned>(i++);
    bool                skip_msg_refinement = pro.get_input<bool>(i++);

    unsigned num_cells = boxm2_ocl_refine_bp::refine_scene(device, scene, opencl_cache, thresh,num_imgs,skip_msg_refinement);
    vcl_cout<<"boxm2_ocl_refine_bp_process num split: "<<num_cells<<vcl_endl;

    //set output
    pro.set_output_val<int>(0, num_cells);

    return true;
}

