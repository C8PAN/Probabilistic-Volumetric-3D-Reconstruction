// This is brl/bseg/boxm2/ocl/pro/processes/boxm2_ocl_render_median_depth_process_cons.cxx
#include <bprb/bprb_func_process.h>
//:
// \file
// \brief  A process for rendering depth map of a scene.
//
// \author Vishal Jain
// \date Mar 10, 2011

#include <vcl_fstream.h>
#include <boxm2/ocl/boxm2_opencl_cache.h>
#include <boxm2/boxm2_scene.h>
#include <boxm2/boxm2_block.h>
#include <boxm2/boxm2_data_base.h>
#include <boxm2/ocl/boxm2_ocl_util.h>
#include <vil/vil_image_view.h>
//brdb stuff
#include <brdb/brdb_value.h>
#include <vcl_algorithm.h>
#include <boxm2/boxm2_util.h>

//directory utility
#include <vcl_where_root_dir.h>
#include <bocl/bocl_device.h>
#include <bocl/bocl_kernel.h>
#include <vul/vul_timer.h>
#include <vpgl/vpgl_lvcs_sptr.h>
#include <vpgl/file_formats/vpgl_geo_camera.h>
#include <boxm2/ocl/algo/boxm2_ocl_camera_converter.h>

namespace boxm2_ocl_render_median_depth_process_globals
{
  const unsigned n_inputs_ = 5;
  const unsigned n_outputs_ = 2;
  vcl_size_t local_threads[2]={8,8};
  void compile_kernel(bocl_device_sptr device,vcl_vector<bocl_kernel*> & vec_kernels, vcl_string app_options)
  {
    //gather all render sources... seems like a lot for rendering...
    vcl_vector<vcl_string> src_paths;
    vcl_string source_dir = boxm2_ocl_util::ocl_src_root();
    src_paths.push_back(source_dir + "scene_info.cl");
    src_paths.push_back(source_dir + "pixel_conversion.cl");
    src_paths.push_back(source_dir + "bit/bit_tree_library_functions.cl");
    src_paths.push_back(source_dir + "backproject.cl");
    src_paths.push_back(source_dir + "statistics_library_functions.cl");
    src_paths.push_back(source_dir + "ray_bundle_library_opt.cl");
    src_paths.push_back(source_dir + "bit/render_median_depth.cl");
    src_paths.push_back(source_dir + "bit/cast_ray_bit.cl");

    //set kernel options
    vcl_string options = " -D COMPUTE_DEPTH_NORM " + app_options;
    options += " -D STEP_CELL=step_cell_render_depth(data_ptr,aux_args,tblock,linfo->block_len)";

    //have kernel construct itself using the context and device
    bocl_kernel * ray_trace_kernel=new bocl_kernel();

    ray_trace_kernel->create_kernel( &device->context(),
                                     device->device_id(),
                                     src_paths,
                                     "render_depth",   //kernel name
                                     options,              //options
                                     "compute depth norm image"); //kernel identifier (for error checking)
    vec_kernels.push_back(ray_trace_kernel);

    //create normalize image kernel
    vcl_vector<vcl_string> norm_src_paths;
    norm_src_paths.push_back(source_dir + "scene_info.cl");
    norm_src_paths.push_back(source_dir + "bit/render_median_depth.cl");
    bocl_kernel * normalize_render_kernel=new bocl_kernel();

    vcl_string normalize_options = " -D REINIT_VIS ";
    normalize_render_kernel->create_kernel( &device->context(),
                                            device->device_id(),
                                            norm_src_paths,
                                            "proc_norm_image",   //kernel name
                                            normalize_options,              //options
                                            "normalize proc_norm_image depth kernel"); //kernel identifier (for error checking)
    vec_kernels.push_back(normalize_render_kernel);

    options = " -D COMPUTE_MEDIAN_DEPTH " + app_options;
    options += " -D STEP_CELL=step_cell_render_depth(data_ptr,aux_args,tblock,linfo->block_len)";

    //have kernel construct itself using the context and device
    bocl_kernel * compute_median_kernel=new bocl_kernel();

    compute_median_kernel->create_kernel( &device->context(),
                                     device->device_id(),
                                     src_paths,
                                     "render_depth",   //kernel name
                                     options,              //options
                                     "compute median depth image"); //kernel identifier (for error checking)
    vec_kernels.push_back(compute_median_kernel);


    normalize_options = " -D REINIT_VIS2 ";
    bocl_kernel * normalize_render_kernel2=new bocl_kernel();
    normalize_render_kernel2->create_kernel( &device->context(),
											device->device_id(),
											norm_src_paths,
											"proc_norm_image",   //kernel name
											normalize_options,              //options
											"normalize proc_norm_image depth kernel"); //kernel identifier (for error checking)
	vec_kernels.push_back(normalize_render_kernel2);


  }
  static vcl_map<vcl_string,vcl_vector<bocl_kernel*> > kernels;
}

bool boxm2_ocl_render_median_depth_process_cons(bprb_func_process& pro)
{
  using namespace boxm2_ocl_render_median_depth_process_globals;

  //process takes 1 input
  vcl_vector<vcl_string> input_types_(n_inputs_);
  input_types_[0] = "bocl_device_sptr";
  input_types_[1] = "boxm2_scene_sptr";
  input_types_[2] = "boxm2_opencl_cache_sptr";
  input_types_[3] = "vpgl_camera_double_sptr";
  input_types_[4] = "vil_image_view_base_sptr";     //input image


  // process has 1 output:
  // output[0]: scene sptr
  vcl_vector<vcl_string>  output_types_(n_outputs_);
  output_types_[0] = "vil_image_view_base_sptr";
  output_types_[1] = "vil_image_view_base_sptr";

  return pro.set_input_types(input_types_) && pro.set_output_types(output_types_);
}

bool boxm2_ocl_render_median_depth_process(bprb_func_process& pro)
{
  using namespace boxm2_ocl_render_median_depth_process_globals;

  if ( pro.n_inputs() < n_inputs_ ) {
    vcl_cout << pro.name() << ": The input number should be " << n_inputs_<< vcl_endl;
    return false;
  }
  float transfer_time=0.0f;
  float gpu_time=0.0f;
  //get the inputs
  unsigned i = 0;
  bocl_device_sptr device= pro.get_input<bocl_device_sptr>(i++);
  boxm2_scene_sptr scene =pro.get_input<boxm2_scene_sptr>(i++);

  boxm2_opencl_cache_sptr opencl_cache= pro.get_input<boxm2_opencl_cache_sptr>(i++);
  vpgl_camera_double_sptr cam= pro.get_input<vpgl_camera_double_sptr>(i++);
  vil_image_view_base_sptr img          = pro.get_input<vil_image_view_base_sptr>(i++);


  vcl_string identifier=device->device_identifier();

  // create a command queue.
  int status=0;
  cl_command_queue queue = clCreateCommandQueue(device->context(),
                                                *(device->device_id()),
                                                CL_QUEUE_PROFILING_ENABLE,
                                                &status);
  if (status!=0)
    return false;


  vcl_string data_type, app_options;
  int appTypeSize =  boxm2_data_traits<BOXM2_MOG3_GREY>::datasize();
  data_type = boxm2_data_traits<BOXM2_MOG3_GREY>::prefix();
  app_options=" -D MOG_TYPE_8 ";

  // compile the kernel
  if (kernels.find(identifier)==kernels.end())
  {
    //vcl_cout<<"===========Compiling kernels==========="<<vcl_endl;
    vcl_vector<bocl_kernel*> ks;
    compile_kernel(device,ks,app_options);
    kernels[identifier]=ks;
  }
  //grab input image, establish cl_ni, cl_nj (so global size is divisible by local size)
  vil_image_view_base_sptr float_img = boxm2_util::prepare_input_image(img, true);
  vil_image_view<float>* img_view = static_cast<vil_image_view<float>* >(float_img.ptr());
  const unsigned cl_ni=(unsigned)RoundUp(img_view->ni(),(int)local_threads[0]);
  const unsigned cl_nj=(unsigned)RoundUp(img_view->nj(),(int)local_threads[1]);

  unsigned ni = img_view->ni();
  unsigned nj = img_view->nj();

  float* median_depth_buff = new float[cl_ni*cl_nj];
  for (unsigned i=0;i<cl_ni*cl_nj;i++) median_depth_buff[i]=0.0f;
  float* vis_buff = new float[cl_ni*cl_nj];
  for (unsigned i=0;i<cl_ni*cl_nj;i++) vis_buff[i]=1.0f;
  float* input_buff=new float[cl_ni*cl_nj];
  for (unsigned i=0;i<cl_ni*cl_nj;i++) input_buff[i]=1.0f;
  float* t_infinity_buff = new float[cl_ni*cl_nj];
  for (unsigned i=0;i<cl_ni*cl_nj;i++) t_infinity_buff[i]=0.0f;
  float* depth_prob_norm_buff = new float[cl_ni*cl_nj];
  for (unsigned i=0;i<cl_ni*cl_nj;i++) depth_prob_norm_buff[i]=0.0f;
  float* depth_cdf_buff = new float[cl_ni*cl_nj];
  for (unsigned i=0;i<cl_ni*cl_nj;i++) depth_cdf_buff[i]=0.0f;


  //copy input vals into image
  int count=0;
  for (unsigned int j=0;j<cl_nj;++j) {
    for (unsigned int i=0;i<cl_ni;++i) {
      input_buff[count] = 0.0f;
      if ( i<img_view->ni() && j< img_view->nj() )
        input_buff[count] = (*img_view)(i,j);
      ++count;
    }
  }

  //bocl_mem_sptr in_image=new bocl_mem(device->context(),input_buff,cl_ni*cl_nj*sizeof(float),"input image buffer");
  bocl_mem_sptr in_image = opencl_cache->alloc_mem(cl_ni*cl_nj*sizeof(float), input_buff, "input image buffer");
  in_image->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR);

  bocl_mem_sptr median_depth_image=opencl_cache->alloc_mem(cl_ni*cl_nj*sizeof(float),median_depth_buff,"median_depth_image");
  median_depth_image->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR);

  bocl_mem_sptr vis_image=opencl_cache->alloc_mem(cl_ni*cl_nj*sizeof(float),vis_buff,"vis image buffer");
  vis_image->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR);

  bocl_mem_sptr t_infinity=opencl_cache->alloc_mem(cl_ni*cl_nj*sizeof(float),t_infinity_buff,"t infinity buffer");
  t_infinity->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR);

  bocl_mem_sptr depth_prob_norm_image=opencl_cache->alloc_mem(cl_ni*cl_nj*sizeof(float),depth_prob_norm_buff,"depth_prob_norm_buff ");
  depth_prob_norm_image->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR);

  bocl_mem_sptr depth_cdf_image=opencl_cache->alloc_mem(cl_ni*cl_nj*sizeof(float),depth_cdf_buff,"depth_cdf_image ");
  depth_cdf_image->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR);


  //set generic cam
  cl_float* ray_origins = new cl_float[4*cl_ni*cl_nj];
  cl_float* ray_directions = new cl_float[4*cl_ni*cl_nj];
  bocl_mem_sptr ray_o_buff = opencl_cache->alloc_mem(cl_ni*cl_nj * sizeof(cl_float4), ray_origins, "ray_origins buffer");
  bocl_mem_sptr ray_d_buff = opencl_cache->alloc_mem(cl_ni*cl_nj * sizeof(cl_float4), ray_directions, "ray_directions buffer");

  boxm2_ocl_camera_converter::compute_ray_image( device, queue, cam, cl_ni, cl_nj, ray_o_buff, ray_d_buff);



  // Image Dimensions
  int img_dim_buff[4];
  img_dim_buff[0] = 0;
  img_dim_buff[1] = 0;
  img_dim_buff[2] = img_view->ni();
  img_dim_buff[3] = img_view->nj();
  bocl_mem_sptr exp_img_dim=new bocl_mem(device->context(), img_dim_buff, sizeof(int)*4, "image dims");
  exp_img_dim->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR);

  // Output Array
  float output_arr[100];
  for (int i=0; i<100; ++i) output_arr[i] = 0.0f;
  bocl_mem_sptr  cl_output=new bocl_mem(device->context(), output_arr, sizeof(float)*100, "output buffer");
  cl_output->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR);

  // bit lookup buffer
  cl_uchar lookup_arr[256];
  boxm2_ocl_util::set_bit_lookup(lookup_arr);
  bocl_mem_sptr lookup=new bocl_mem(device->context(), lookup_arr, sizeof(cl_uchar)*256, "bit lookup buffer");
  lookup->create_buffer(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);

  //2. set workgroup size
  vcl_size_t lThreads[] = {8, 8};
  vcl_size_t gThreads[] = {cl_ni,cl_nj};

  float subblk_dim = 0.0;
  // set arguments
  vcl_vector<boxm2_block_id> vis_order;
  if(cam->type_name() == "vpgl_geo_camera" )
      vis_order= scene->get_block_ids(); // order does not matter for a top down orthographic camera  and axis aligned blocks
  else if(cam->type_name() == "vpgl_perspective_camera")
      vis_order= scene->get_vis_blocks_opt((vpgl_perspective_camera<double>*)cam.ptr(),ni,nj);
  else
      vis_order= scene->get_vis_blocks(cam);


  vcl_vector<boxm2_block_id>::iterator id;
  for (id = vis_order.begin(); id != vis_order.end(); ++id)
  {
    //choose correct render kernel
    boxm2_block_metadata mdata = scene->get_block_metadata(*id);
    bocl_kernel* kern =  kernels[identifier][0];

    //write the image values to the buffer
    vul_timer transfer;
    bocl_mem* blk           = opencl_cache->get_block(scene,*id);
    bocl_mem* alpha         = opencl_cache->get_data<BOXM2_ALPHA>(scene,*id);
    bocl_mem * blk_info     = opencl_cache->loaded_block_info();
    int alphaTypeSize = (int)boxm2_data_info::datasize(boxm2_data_traits<BOXM2_ALPHA>::prefix());
    bocl_mem* mog       = opencl_cache->get_data(scene,*id,data_type,alpha->num_bytes()/alphaTypeSize*appTypeSize,false);    //info_buffer->data_buffer_length*boxm2_data_info::datasize(data_type));

    ////3. SET args
    kern->set_arg( blk_info );
    kern->set_arg( blk );
    kern->set_arg( alpha );
    kern->set_arg( mog );
    kern->set_arg( ray_o_buff.ptr() );
    kern->set_arg( ray_d_buff.ptr() );
    kern->set_arg( in_image.ptr() );
    kern->set_arg( t_infinity.ptr() );
    kern->set_arg( vis_image.ptr() );
    kern->set_arg( depth_prob_norm_image.ptr() );

    kern->set_arg( exp_img_dim.ptr());
    kern->set_arg( lookup.ptr() );
    //local tree , cumsum buffer, imindex buffer
    kern->set_local_arg( local_threads[0]*local_threads[1]*sizeof(cl_uchar16) );
    kern->set_local_arg( local_threads[0]*local_threads[1]*10*sizeof(cl_uchar) );
    kern->set_local_arg( local_threads[0]*local_threads[1]*sizeof(cl_int) );
    //execute kernel
    kern->execute(queue, 2, lThreads, gThreads);
    clFinish(queue);
    gpu_time += kern->exec_time();
    // clear render kernel args so it can reset em on next execution
    kern->clear_args();
  }


  bocl_kernel* kern =  kernels[identifier][1];
  kern->set_arg( in_image.ptr() );
  kern->set_arg( depth_prob_norm_image.ptr() );
  kern->set_arg( vis_image.ptr() );
  kern->set_arg( exp_img_dim.ptr() );
  //execute kernel
  kern->execute( queue, 2,  lThreads, gThreads);
  status = clFinish(queue);
  if (!check_val(status, MEM_FAILURE, "UPDATE EXECUTE FAILED: " + error_to_string(status)))
	  return false;
  kern->clear_args();


  for (id = vis_order.begin(); id != vis_order.end(); ++id)
  {
    //choose correct render kernel
    boxm2_block_metadata mdata = scene->get_block_metadata(*id);
    bocl_kernel* kern =  kernels[identifier][2];

    //write the image values to the buffer
    vul_timer transfer;
    bocl_mem* blk           = opencl_cache->get_block(scene,*id);
    bocl_mem* alpha         = opencl_cache->get_data<BOXM2_ALPHA>(scene,*id);
    bocl_mem * blk_info     = opencl_cache->loaded_block_info();
    int alphaTypeSize = (int)boxm2_data_info::datasize(boxm2_data_traits<BOXM2_ALPHA>::prefix());
    bocl_mem* mog       = opencl_cache->get_data(scene,*id,data_type,alpha->num_bytes()/alphaTypeSize*appTypeSize,false);    //info_buffer->data_buffer_length*boxm2_data_info::datasize(data_type));

    ////3. SET args
    kern->set_arg( blk_info );
    kern->set_arg( blk );
    kern->set_arg( alpha );
    kern->set_arg( mog );
    kern->set_arg( ray_o_buff.ptr() );
    kern->set_arg( ray_d_buff.ptr() );
    kern->set_arg( in_image.ptr() );
    kern->set_arg( vis_image.ptr() );
    kern->set_arg( depth_prob_norm_image.ptr() );
    kern->set_arg( median_depth_image.ptr() );
    kern->set_arg( depth_cdf_image.ptr() );

    kern->set_arg( exp_img_dim.ptr());
    kern->set_arg( lookup.ptr() );
    //local tree , cumsum buffer, imindex buffer
    kern->set_local_arg( local_threads[0]*local_threads[1]*sizeof(cl_uchar16) );
    kern->set_local_arg( local_threads[0]*local_threads[1]*10*sizeof(cl_uchar) );
    kern->set_local_arg( local_threads[0]*local_threads[1]*sizeof(cl_int) );
    //execute kernel
    kern->execute(queue, 2, lThreads, gThreads);
    clFinish(queue);
    gpu_time += kern->exec_time();
    // clear render kernel args so it can reset em on next execution
    kern->clear_args();
  }

  bocl_kernel* kern2 =  kernels[identifier][3];
  kern2->set_arg( depth_cdf_image.ptr() );
  kern2->set_arg( depth_prob_norm_image.ptr());
  kern2->set_arg( median_depth_image.ptr() );
  kern2->set_arg( vis_image.ptr() );
  kern2->set_arg( t_infinity.ptr() );
  kern2->set_arg( exp_img_dim.ptr() );
  //execute kernel
  kern2->execute( queue, 2,  lThreads, gThreads);
  status = clFinish(queue);
  if (!check_val(status, MEM_FAILURE, "UPDATE EXECUTE FAILED: " + error_to_string(status)))
	  return false;
  kern2->clear_args();


  median_depth_image->read_to_buffer(queue);
  vis_image->read_to_buffer(queue);
  status = clFinish(queue);


  vil_image_view<float>* exp_img_out=new vil_image_view<float>(ni,nj);
  vil_image_view<float>* vis_out=new vil_image_view<float>(ni,nj);

  for (unsigned c=0;c<nj;c++)
  {
    for (unsigned r=0;r<ni;r++)
    {
      (*exp_img_out)(r,c)= median_depth_buff[c*cl_ni+r];
      (*vis_out)(r,c)=vis_buff[c*cl_ni+r];
    }
  }

  delete[] depth_cdf_buff;
  delete[] depth_prob_norm_buff;
  delete[] median_depth_buff;
  delete[] vis_buff;
  delete[] input_buff;
  delete[] t_infinity_buff;
  delete[] ray_origins;
  delete[] ray_directions;

  opencl_cache->unref_mem(in_image.ptr());
  opencl_cache->unref_mem(depth_prob_norm_image.ptr());
  opencl_cache->unref_mem(median_depth_image.ptr());
  opencl_cache->unref_mem(vis_image.ptr());
  opencl_cache->unref_mem(t_infinity.ptr());
  opencl_cache->unref_mem(depth_cdf_image.ptr());
  opencl_cache->unref_mem(ray_o_buff.ptr());
  opencl_cache->unref_mem(ray_d_buff.ptr());

  clReleaseCommandQueue(queue);
  i=0;
  pro.set_output_val<vil_image_view_base_sptr>(i++, exp_img_out);
  pro.set_output_val<vil_image_view_base_sptr>(i++, vis_out);
  return true;
}





namespace boxm2_ocl_render_median_depth_without_app_process_globals
{
  const unsigned n_inputs_ = 6;
  const unsigned n_outputs_ = 2;
  vcl_size_t local_threads[2]={8,8};
  void compile_kernel(bocl_device_sptr device,vcl_vector<bocl_kernel*> & vec_kernels)
  {
    //gather all render sources... seems like a lot for rendering...
    vcl_vector<vcl_string> src_paths;
    vcl_string source_dir = boxm2_ocl_util::ocl_src_root();
    src_paths.push_back(source_dir + "scene_info.cl");
    src_paths.push_back(source_dir + "pixel_conversion.cl");
    src_paths.push_back(source_dir + "bit/bit_tree_library_functions.cl");
    src_paths.push_back(source_dir + "backproject.cl");
    src_paths.push_back(source_dir + "statistics_library_functions.cl");
    src_paths.push_back(source_dir + "ray_bundle_library_opt.cl");
    src_paths.push_back(source_dir + "bit/render_median_depth.cl");
    src_paths.push_back(source_dir + "bit/cast_ray_bit.cl");

    //set kernel options
    vcl_string options = " -D RENDER_MEDIAN_DEPTH_WITHOUT_APP ";
    options +=  "-D DETERMINISTIC";
    options += " -D STEP_CELL=step_cell_render_depth2(tblock,linfo->block_len,aux_args.alpha,data_ptr,d*linfo->block_len,aux_args.vis,aux_args.expdepth,aux_args.expdepthsqr,aux_args.probsum,aux_args.t,aux_args.cdf)";

    //have kernel construct itself using the context and device
    bocl_kernel * ray_trace_kernel=new bocl_kernel();

    ray_trace_kernel->create_kernel( &device->context(),
                                     device->device_id(),
                                     src_paths,
                                     "render_depth",   //kernel name
                                     options,              //options
                                     "boxm2 opencl render depth image"); //kernel identifier (for error checking)
    vec_kernels.push_back(ray_trace_kernel);

  }
  static vcl_map<vcl_string,vcl_vector<bocl_kernel*> > kernels;
}

bool boxm2_ocl_render_median_depth_without_app_process_cons(bprb_func_process& pro)
{
  using namespace boxm2_ocl_render_median_depth_without_app_process_globals;

  //process takes 1 input
  vcl_vector<vcl_string> input_types_(n_inputs_);
  input_types_[0] = "bocl_device_sptr";
  input_types_[1] = "boxm2_scene_sptr";
  input_types_[2] = "boxm2_opencl_cache_sptr";
  input_types_[3] = "vpgl_camera_double_sptr";
  input_types_[4] = "unsigned";
  input_types_[5] = "unsigned";


  // process has 1 output:
  // output[0]: scene sptr
  vcl_vector<vcl_string>  output_types_(n_outputs_);
  output_types_[0] = "vil_image_view_base_sptr";
 output_types_[1] = "vil_image_view_base_sptr";
//  output_types_[2] = "vil_image_view_base_sptr";

  return pro.set_input_types(input_types_) && pro.set_output_types(output_types_);
}

bool boxm2_ocl_render_median_depth_without_app_process(bprb_func_process& pro)
{
  using namespace boxm2_ocl_render_median_depth_without_app_process_globals;

  if ( pro.n_inputs() < n_inputs_ ) {
    vcl_cout << pro.name() << ": The input number should be " << n_inputs_<< vcl_endl;
    return false;
  }
  float transfer_time=0.0f;
  float gpu_time=0.0f;
  //get the inputs
  unsigned i = 0;
  bocl_device_sptr device= pro.get_input<bocl_device_sptr>(i++);
  boxm2_scene_sptr scene =pro.get_input<boxm2_scene_sptr>(i++);

  boxm2_opencl_cache_sptr opencl_cache= pro.get_input<boxm2_opencl_cache_sptr>(i++);
  vpgl_camera_double_sptr cam= pro.get_input<vpgl_camera_double_sptr>(i++);
  unsigned ni=pro.get_input<unsigned>(i++);
  unsigned nj=pro.get_input<unsigned>(i++);

  vcl_string identifier=device->device_identifier();

  // create a command queue.
  int status=0;
  cl_command_queue queue = clCreateCommandQueue(device->context(),
                                                *(device->device_id()),
                                                CL_QUEUE_PROFILING_ENABLE,
                                                &status);
  if (status!=0)
    return false;

  // compile the kernel
  if (kernels.find(identifier)==kernels.end())
  {
    //vcl_cout<<"===========Compiling kernels==========="<<vcl_endl;
    vcl_vector<bocl_kernel*> ks;
    compile_kernel(device,ks);
    kernels[identifier]=ks;
  }

#if 0
  // create all buffers
  cl_float cam_buffer[48];
  boxm2_ocl_util::set_persp_camera(cam, cam_buffer);
  bocl_mem_sptr persp_cam=new bocl_mem(device->context(), cam_buffer, 3*sizeof(cl_float16), "persp cam buffer");
  persp_cam->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR);
#endif

  unsigned cl_ni=RoundUp(ni,local_threads[0]);
  unsigned cl_nj=RoundUp(nj,local_threads[1]);
  float* buff = new float[cl_ni*cl_nj];
  for (unsigned i=0;i<cl_ni*cl_nj;i++) buff[i]=0.0f;
  float* var_buff = new float[cl_ni*cl_nj];
  for (unsigned i=0;i<cl_ni*cl_nj;i++) var_buff[i]=0.0f;
  float* vis_buff = new float[cl_ni*cl_nj];
  for (unsigned i=0;i<cl_ni*cl_nj;i++) vis_buff[i]=1.0f;
  float* prob_buff = new float[cl_ni*cl_nj];
  for (unsigned i=0;i<cl_ni*cl_nj;i++) prob_buff[i]=0.0f;
  float* t_infinity_buff = new float[cl_ni*cl_nj];
  for (unsigned i=0;i<cl_ni*cl_nj;i++) t_infinity_buff[i]=0.0f;
  float* cdf_buff = new float[cl_ni*cl_nj];
  for (unsigned i=0;i<cl_ni*cl_nj;i++) cdf_buff[i]=0.0f;

  bocl_mem_sptr cdf_image=opencl_cache->alloc_mem(cl_ni*cl_nj*sizeof(float),cdf_buff,"asd image buffer");
  cdf_image->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR);

  bocl_mem_sptr exp_image=opencl_cache->alloc_mem(cl_ni*cl_nj*sizeof(float),buff,"exp image buffer");
  exp_image->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR);

  bocl_mem_sptr var_image=opencl_cache->alloc_mem(cl_ni*cl_nj*sizeof(float),var_buff,"var image buffer");
  var_image->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR);

  bocl_mem_sptr vis_image=opencl_cache->alloc_mem(cl_ni*cl_nj*sizeof(float),vis_buff,"vis image buffer");
  vis_image->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR);

  bocl_mem_sptr prob_image=opencl_cache->alloc_mem(cl_ni*cl_nj*sizeof(float),prob_buff,"vis x omega image buffer");
  prob_image->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR);

  bocl_mem_sptr t_infinity=opencl_cache->alloc_mem(cl_ni*cl_nj*sizeof(float),t_infinity_buff,"t infinity buffer");
  t_infinity->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR);

  //set generic cam
  cl_float* ray_origins = new cl_float[4*cl_ni*cl_nj];
  cl_float* ray_directions = new cl_float[4*cl_ni*cl_nj];
  bocl_mem_sptr ray_o_buff = opencl_cache->alloc_mem(cl_ni*cl_nj * sizeof(cl_float4), ray_origins, "ray_origins buffer");
  bocl_mem_sptr ray_d_buff = opencl_cache->alloc_mem(cl_ni*cl_nj * sizeof(cl_float4), ray_directions, "ray_directions buffer");
  if(cam->type_name() == "vpgl_geo_camera" )
  {
     vcl_cerr << "ERROR geo cam not implemented." << vcl_endl;
  }
  else
  {
      boxm2_ocl_camera_converter::compute_ray_image( device, queue, cam, cl_ni, cl_nj, ray_o_buff, ray_d_buff);
  }

  // Image Dimensions
  int img_dim_buff[4];
  img_dim_buff[0] = 0;
  img_dim_buff[1] = 0;
  img_dim_buff[2] = ni;
  img_dim_buff[3] = nj;
  bocl_mem_sptr exp_img_dim=new bocl_mem(device->context(), img_dim_buff, sizeof(int)*4, "image dims");
  exp_img_dim->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR);

  // Output Array
  float output_arr[100];
  for (int i=0; i<100; ++i) output_arr[i] = 0.0f;
  bocl_mem_sptr  cl_output=new bocl_mem(device->context(), output_arr, sizeof(float)*100, "output buffer");
  cl_output->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR);

  // bit lookup buffer
  cl_uchar lookup_arr[256];
  boxm2_ocl_util::set_bit_lookup(lookup_arr);
  bocl_mem_sptr lookup=new bocl_mem(device->context(), lookup_arr, sizeof(cl_uchar)*256, "bit lookup buffer");
  lookup->create_buffer(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);

  //2. set workgroup size
  vcl_size_t lThreads[] = {8, 8};
  vcl_size_t gThreads[] = {cl_ni,cl_nj};
  float subblk_dim = 0.0;
  // set arguments
  vcl_vector<boxm2_block_id> vis_order;
  if(cam->type_name() == "vpgl_geo_camera" )
      vis_order= scene->get_block_ids(); // order does not matter for a top down orthographic camera  and axis aligned blocks
  else if(cam->type_name() == "vpgl_perspective_camera")
      vis_order= scene->get_vis_blocks_opt((vpgl_perspective_camera<double>*)cam.ptr(),ni,nj);
  else
      vis_order= scene->get_vis_blocks(cam);

  vcl_vector<boxm2_block_id>::iterator id;
  for (id = vis_order.begin(); id != vis_order.end(); ++id)
  {
    //choose correct render kernel
    boxm2_block_metadata mdata = scene->get_block_metadata(*id);
    bocl_kernel* kern =  kernels[identifier][0];

    //write the image values to the buffer
    vul_timer transfer;
    bocl_mem* blk           = opencl_cache->get_block(scene,*id);
    bocl_mem* alpha         = opencl_cache->get_data<BOXM2_ALPHA>(scene,*id);
    bocl_mem * blk_info     = opencl_cache->loaded_block_info();
    transfer_time          += (float) transfer.all();
    subblk_dim              = mdata.sub_block_dim_.x();
    ////3. SET args
    kern->set_arg( blk_info );
    kern->set_arg( blk );
    kern->set_arg( alpha );
    //kern->set_arg( persp_cam.ptr() );
    kern->set_arg( ray_o_buff.ptr() );
    kern->set_arg( ray_d_buff.ptr() );
    kern->set_arg( exp_image.ptr() );
    kern->set_arg( var_image.ptr() );
    kern->set_arg( cdf_image.ptr() );
    kern->set_arg( exp_img_dim.ptr());
    kern->set_arg( cl_output.ptr() );
    kern->set_arg( lookup.ptr() );
    kern->set_arg( vis_image.ptr() );
    kern->set_arg( prob_image.ptr() );
    kern->set_arg( t_infinity.ptr() );

    //local tree , cumsum buffer, imindex buffer
    kern->set_local_arg( local_threads[0]*local_threads[1]*sizeof(cl_uchar16) );
    kern->set_local_arg( local_threads[0]*local_threads[1]*10*sizeof(cl_uchar) );
    kern->set_local_arg( local_threads[0]*local_threads[1]*sizeof(cl_int) );

    //execute kernel
    kern->execute(queue, 2, lThreads, gThreads);
    clFinish(queue);
    gpu_time += kern->exec_time();

    cl_output->read_to_buffer(queue);

    // clear render kernel args so it can reset em on next execution
    kern->clear_args();
  }

//  bocl_mem_sptr  subblk_dim_mem=new bocl_mem(device->context(), &(subblk_dim), sizeof(float), "sub block dim buffer");
//  subblk_dim_mem->create_buffer(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR );
//  // normalize
//  {
//    bocl_kernel* normalize_kern= kernels[identifier][1];
//    normalize_kern->set_arg( exp_image.ptr() );
//    normalize_kern->set_arg( var_image.ptr() );
//    normalize_kern->set_arg( vis_image.ptr() );
//    normalize_kern->set_arg( exp_img_dim.ptr());
//    normalize_kern->set_arg( t_infinity.ptr());
//    normalize_kern->set_arg( subblk_dim_mem.ptr());
//    normalize_kern->execute( queue, 2, local_threads, gThreads);
//    clFinish(queue);
//    gpu_time += normalize_kern->exec_time();
//
//    //clear render kernel args so it can reset em on next execution
//    normalize_kern->clear_args();
    exp_image->read_to_buffer(queue);
//    var_image->read_to_buffer(queue);
    vis_image->read_to_buffer(queue);
//  }


  clReleaseCommandQueue(queue);

  vil_image_view<float>* exp_img_out=new vil_image_view<float>(ni,nj);
//  vil_image_view<float>* exp_var_out=new vil_image_view<float>(ni,nj);
 vil_image_view<float>* vis_out=new vil_image_view<float>(ni,nj);

  for (unsigned c=0;c<nj;c++)
  {
    for (unsigned r=0;r<ni;r++)
    {
      (*exp_img_out)(r,c)=buff[c*cl_ni+r];
//      (*exp_var_out)(r,c)=var_buff[c*cl_ni+r];
     (*vis_out)(r,c)=vis_buff[c*cl_ni+r];
    }
  }

  delete[] buff;
  delete[] var_buff;
  delete[] vis_buff;
  delete[] prob_buff;
  delete[] t_infinity_buff;
  delete[] ray_origins;
  delete[] ray_directions;
  delete[] cdf_buff;

  opencl_cache->unref_mem(cdf_image.ptr());
  opencl_cache->unref_mem(exp_image.ptr());
  opencl_cache->unref_mem(var_image.ptr());
  opencl_cache->unref_mem(vis_image.ptr());
  opencl_cache->unref_mem(prob_image.ptr());
  opencl_cache->unref_mem(t_infinity.ptr());
  opencl_cache->unref_mem(ray_o_buff.ptr());
  opencl_cache->unref_mem(ray_d_buff.ptr());
  clReleaseCommandQueue(queue);

  // store scene smaprt pointer
  i=0;
  pro.set_output_val<vil_image_view_base_sptr>(i++, exp_img_out);
//  pro.set_output_val<vil_image_view_base_sptr>(i++, exp_var_out);
 pro.set_output_val<vil_image_view_base_sptr>(i++, vis_out);
  return true;
}
