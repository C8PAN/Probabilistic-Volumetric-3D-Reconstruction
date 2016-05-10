// This is brl/bseg/boxm2/ocl/pro/processes/boxm2_ocl_clear_app_models_process.cxx
#include <bprb/bprb_func_process.h>
//:
// \file
// \brief  A process to initialize appearance belief
//
// \author Ali Osman Ulusoy

#include <vcl_fstream.h>
#include <vcl_algorithm.h>
#include <boxm2/ocl/boxm2_opencl_cache.h>
#include <boxm2/boxm2_scene.h>
#include <boxm2/boxm2_block.h>
#include <boxm2/boxm2_data_base.h>
#include <boxm2/ocl/boxm2_ocl_util.h>
#include <boxm2/boxm2_util.h>
#include <vil/vil_image_view.h>

#include <vil/vil_new.h>
#include <vpl/vpl.h> // vpl_unlink()

//brdb stuff
#include <brdb/brdb_value.h>

//directory utility
#include <vcl_where_root_dir.h>
#include <bocl/bocl_device.h>
#include <bocl/bocl_kernel.h>
#include <vul/vul_timer.h>

namespace boxm2_ocl_clear_app_models_process_globals
{
  const unsigned int n_inputs_  = 3;
  const unsigned int n_outputs_ = 0;
}

bool boxm2_ocl_clear_app_models_process_cons(bprb_func_process& pro)
{
  using namespace boxm2_ocl_clear_app_models_process_globals;

  //process takes 9 inputs (of which the four last ones are optional):
  vcl_vector<vcl_string> input_types_(n_inputs_);
  input_types_[0] = "bocl_device_sptr";
  input_types_[1] = "boxm2_scene_sptr";
  input_types_[2] = "boxm2_opencl_cache_sptr";


  // process has no outputs
  vcl_vector<vcl_string>  output_types_(n_outputs_);
  bool good = pro.set_input_types(input_types_) && pro.set_output_types(output_types_);

  return good;
}

bool boxm2_ocl_clear_app_models_process(bprb_func_process& pro)
{
	using namespace boxm2_ocl_clear_app_models_process_globals;

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

	vcl_size_t binCache = opencl_cache.ptr()->bytes_in_cache();
	vcl_cout<<"Update MBs in cache: "<<binCache/(1024.0*1024.0)<<vcl_endl;

	//make correct data types are here
	vcl_string data_type, num_obs_type;
	vcl_vector<vcl_string> apps = scene->appearances();
	bool foundDataType = false, foundNumObsType = false;
	for (unsigned int i=0; i<apps.size(); ++i) {
		if ( apps[i] == boxm2_data_traits<BOXM2_MOG3_GREY>::prefix() )
		{
			data_type = apps[i];
			foundDataType = true;
		}
		else if ( apps[i] == boxm2_data_traits<BOXM2_NUM_OBS>::prefix() )
		{
			num_obs_type = apps[i];
			foundNumObsType = true;
		}
	}
	if (!foundDataType) {
		vcl_cout<<"boxm2_ocl_clear_app_models_process ERROR: scene doesn't have BOXM2_MOG3_GREY or BOXM2_MOG3_GREY_16 data type"<<vcl_endl;
		return false;
	}
	if (!foundNumObsType) {
		vcl_cout<<"boxm2_ocl_clear_app_models_process ERROR: scene doesn't have BOXM2_NUM_OBS type"<<vcl_endl;
		return false;
	}

   // create a command queue.
   int status=0;
   cl_command_queue queue = clCreateCommandQueue( device->context(),
												  *(device->device_id()),
												  CL_QUEUE_PROFILING_ENABLE,
												  &status);

	vcl_vector<boxm2_block_id> vis_order = scene->get_block_ids();

	for (vcl_vector<boxm2_block_id>::iterator id = vis_order.begin(); id != vis_order.end(); ++id)
	{
	  vcl_cout << "Processing: " << *id << vcl_endl;
	  bocl_mem* blk       = opencl_cache->get_block(scene,*id);
	  bocl_mem* blk_info  = opencl_cache->loaded_block_info();
	  bocl_mem* alpha     = opencl_cache->get_data<BOXM2_ALPHA>(scene,*id,0,false);
	  boxm2_scene_info* info_buffer = (boxm2_scene_info*) blk_info->cpu_buffer();
	  int alphaTypeSize = (int)boxm2_data_info::datasize(boxm2_data_traits<BOXM2_ALPHA>::prefix());
	  info_buffer->data_buffer_length = (int) (alpha->num_bytes()/alphaTypeSize);
	  blk_info->write_to_buffer((queue));

	  int appTypeSize = (int)boxm2_data_info::datasize(data_type);
	  int nobsTypeSize = (int)boxm2_data_info::datasize(num_obs_type);
	  // data type string may contain an identifier so determine the buffer size
	  bocl_mem* mog       = opencl_cache->get_data(scene,*id,data_type,alpha->num_bytes()/alphaTypeSize*appTypeSize,false);    //info_buffer->data_buffer_length*boxm2_data_info::datasize(data_type));
	  bocl_mem* num_obs   = opencl_cache->get_data(scene,*id,num_obs_type,alpha->num_bytes()/alphaTypeSize*nobsTypeSize,false);//,info_buffer->data_buffer_length*boxm2_data_info::datasize(num_obs_type));

	  mog->zero_gpu_buffer(queue);
	  num_obs->zero_gpu_buffer(queue);
	  clFinish(queue);

	  mog->read_to_buffer(queue);
	  num_obs->read_to_buffer(queue);

	}
	clReleaseCommandQueue(queue);


}

