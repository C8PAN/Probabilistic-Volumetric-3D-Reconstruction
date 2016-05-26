#ifdef COMPUTE_DEPTH_NORM
//need to define a struct of type AuxArgs with auxiliary arguments
// to supplement cast ray args
typedef struct
{
  __global float* alpha;
  __global MOG_TYPE * mog;
  float pixel_intensity;
  float* t;
  float* vis;
  float* depth_prob_norm;
} AuxArgs;

//forward declare cast ray (so you can use it)
void cast_ray(int,int,float,float,float,float,float,float,
              __constant RenderSceneInfo*, __global int4*,
              __local uchar16*, __constant uchar *,__local uchar *,
              float*, AuxArgs,float tnear, float tfar);
__kernel
void
render_depth( __constant  RenderSceneInfo    * linfo,
              __global    int4               * tree_array,
              __global    float              * alpha_array,
              __global    MOG_TYPE           * mixture_array,     // mixture for each block
              __global    float4             * ray_origins,
              __global    float4             * ray_directions,
              __global    float              * in_image,        // camera orign and SVD of inverse of camera matrix
              __global    float              * t_image,        // camera orign and SVD of inverse of camera matrix
              __global    float              * vis_image,        // camera orign and SVD of inverse of camera matrix
              __global    float              * depth_prob_norm_image,        // camera orign and SVD of inverse of camera matrix
              __global    uint4              * exp_image_dims,   // sum of squares.
              __constant  uchar              * bit_lookup,
              __local     uchar16            * local_tree,
              __local     uchar              * cumsum,        // cumulative sum helper for data pointer
              __local     int                * imIndex)
{
  //----------------------------------------------------------------------------
  //get local id (0-63 for an 8x8) of this patch + image coordinates and camera
  // check for validity before proceeding
  //----------------------------------------------------------------------------
  uchar llid = (uchar)(get_local_id(0) + get_local_size(0)*get_local_id(1));
  int i=0,j=0;
  i=get_global_id(0);
  j=get_global_id(1);

  // check to see if the thread corresponds to an actual pixel as in some
  // cases #of threads will be more than the pixels.
  if (i>=(*exp_image_dims).z || j>=(*exp_image_dims).w)
    return;

   //Store image index (may save a register).  Also initialize VIS and expected_int
  imIndex[llid] = j*get_global_size(0)+i;

  //----------------------------------------------------------------------------
  // Calculate ray origin, and direction
  // (make sure ray direction is never axis aligned)
  //----------------------------------------------------------------------------
  float4 ray_o = ray_origins[ imIndex[llid] ];
  float4 ray_d = ray_directions[ imIndex[llid] ];
  float ray_ox, ray_oy, ray_oz, ray_dx, ray_dy, ray_dz;
  calc_scene_ray_generic_cam(linfo, ray_o, ray_d, &ray_ox, &ray_oy, &ray_oz, &ray_dx, &ray_dy, &ray_dz);

  //----------------------------------------------------------------------------
  // we know i,j map to a point on the image, have calculated ray
  // BEGIN RAY TRACE
  //----------------------------------------------------------------------------

  float vis_rec     = vis_image[imIndex[llid]];
  float t           = t_image[imIndex[llid]];
  float pixel_intensity = in_image[imIndex[llid]];
  float depth_prob_norm = depth_prob_norm_image[imIndex[llid]];


  AuxArgs aux_args;
  aux_args.alpha  = alpha_array;
  aux_args.mog        = mixture_array;
  aux_args.t = &t;
  aux_args.vis = &vis_rec;
  aux_args.depth_prob_norm = &depth_prob_norm;



  float vis = 1.0;
  cast_ray( i, j,
            ray_ox, ray_oy, ray_oz,
            ray_dx, ray_dy, ray_dz,
            linfo, tree_array,                                    //scene info
            local_tree, bit_lookup, cumsum, &vis, aux_args,0,MAXFLOAT);      //utility info

  //store the expected intensity
  vis_image[imIndex[llid]]  = vis_rec;
  t_image[imIndex[llid]]  = (* aux_args.t) ;
  depth_prob_norm_image[imIndex[llid]]  = (* aux_args.depth_prob_norm) ;
}

void step_cell_render_depth(int data_ptr,
                             AuxArgs aux_args,
                             float depth,
                             float block_len)
{
  float prob = aux_args.alpha[data_ptr];
  float diff_omega= 1-prob;

  CONVERT_FUNC_FLOAT8(mixture,aux_args.mog[data_ptr])/NORM;
  float  weight3  = (1.0f-mixture.s2-mixture.s5);
  float PI = gauss_3_mixture_prob_density( aux_args.pixel_intensity,
                                         mixture.s0,
                                         mixture.s1,
                                         mixture.s2,
                                         mixture.s3,
                                         mixture.s4,
                                         mixture.s5,
                                         mixture.s6,
                                         mixture.s7,
                                         weight3 //(1.0f-mixture.s2-mixture.s5)
                                        );/* PI */

  float omega = (*aux_args.vis) * prob;

  (*aux_args.depth_prob_norm) += omega * PI;
  (*aux_args.vis)    *= diff_omega;
  (*aux_args.t) = depth*block_len;
}

#endif


#ifdef REINIT_VIS
__kernel void proc_norm_image (   __global float * in_image,
                                  __global float* depth_prob_norm_image,
                                  __global float* vis_image,
                                  __global uint4 * imgdims)
{
    // linear global id of the normalization imagee,
    int i=0;
    int j=0;
    i=get_global_id(0);
    j=get_global_id(1);
    float vis = vis_image[j*get_global_size(0) + i];
    float intensity = in_image[j*get_global_size(0) + i];

    //float density = (exp(-(intensity - appbuffer[0])*(intensity - appbuffer[0])/(2*appbuffer[1]*appbuffer[1]) ))/(sqrt(2*3.141)*appbuffer[1]);

    if (i>=(*imgdims).z && j>=(*imgdims).w)
        return;

    depth_prob_norm_image[j*get_global_size(0) + i] += 1-vis;
    vis_image[j*get_global_size(0) + i] = 1.0f; // initial vis = 1.0f
}
#endif // REINIT_VIS


#ifdef COMPUTE_MEDIAN_DEPTH
//need to define a struct of type AuxArgs with auxiliary arguments
// to supplement cast ray args
typedef struct
{
  __global float* alpha;
  __global MOG_TYPE * mog;
  float pixel_intensity;
  float* vis;
  float depth_prob_norm;
  float* depth_cdf;
  float* median_depth;
} AuxArgs;

//forward declare cast ray (so you can use it)
void cast_ray(int,int,float,float,float,float,float,float,
              __constant RenderSceneInfo*, __global int4*,
              __local uchar16*, __constant uchar *,__local uchar *,
              float*, AuxArgs,float tnear, float tfar);
__kernel
void
render_depth( __constant  RenderSceneInfo    * linfo,
              __global    int4               * tree_array,
              __global    float              * alpha_array,
              __global    MOG_TYPE           * mixture_array,     // mixture for each block
              __global    float4             * ray_origins,
              __global    float4             * ray_directions,
              __global    float              * in_image,        // camera orign and SVD of inverse of camera matrix
              __global    float              * vis_image,        // camera orign and SVD of inverse of camera matrix
              __global    float              * depth_prob_norm_image,        // camera orign and SVD of inverse of camera matrix
              __global    float              * median_depth_image,        // camera orign and SVD of inverse of camera matrix
              __global    float              * depth_cdf_image,        // camera orign and SVD of inverse of camera matrix
              __global    uint4              * exp_image_dims,   // sum of squares.
              __constant  uchar              * bit_lookup,
              __local     uchar16            * local_tree,
              __local     uchar              * cumsum,        // cumulative sum helper for data pointer
              __local     int                * imIndex)
{
  //----------------------------------------------------------------------------
  //get local id (0-63 for an 8x8) of this patch + image coordinates and camera
  // check for validity before proceeding
  //----------------------------------------------------------------------------
  uchar llid = (uchar)(get_local_id(0) + get_local_size(0)*get_local_id(1));
  int i=0,j=0;
  i=get_global_id(0);
  j=get_global_id(1);

  // check to see if the thread corresponds to an actual pixel as in some
  // cases #of threads will be more than the pixels.
  if (i>=(*exp_image_dims).z || j>=(*exp_image_dims).w)
    return;

   //Store image index (may save a register).  Also initialize VIS and expected_int
  imIndex[llid] = j*get_global_size(0)+i;

  //----------------------------------------------------------------------------
  // Calculate ray origin, and direction
  // (make sure ray direction is never axis aligned)
  //----------------------------------------------------------------------------
  float4 ray_o = ray_origins[ imIndex[llid] ];
  float4 ray_d = ray_directions[ imIndex[llid] ];
  float ray_ox, ray_oy, ray_oz, ray_dx, ray_dy, ray_dz;
  calc_scene_ray_generic_cam(linfo, ray_o, ray_d, &ray_ox, &ray_oy, &ray_oz, &ray_dx, &ray_dy, &ray_dz);

  //----------------------------------------------------------------------------
  // we know i,j map to a point on the image, have calculated ray
  // BEGIN RAY TRACE
  //----------------------------------------------------------------------------

  float vis_rec     = vis_image[imIndex[llid]];
  float pixel_intensity = in_image[imIndex[llid]];
  float depth_prob_norm = depth_prob_norm_image[imIndex[llid]];
  float depth_cdf = depth_cdf_image[imIndex[llid]];
  float median_depth = median_depth_image[imIndex[llid]];


  AuxArgs aux_args;
  aux_args.alpha  = alpha_array;
  aux_args.mog        = mixture_array;
  aux_args.vis = &vis_rec;
  aux_args.depth_prob_norm = depth_prob_norm;
  aux_args.depth_cdf = &depth_cdf;
  aux_args.median_depth = &median_depth;

  float vis = 1.0;
  cast_ray( i, j,
            ray_ox, ray_oy, ray_oz,
            ray_dx, ray_dy, ray_dz,
            linfo, tree_array,                                    //scene info
            local_tree, bit_lookup, cumsum, &vis, aux_args,0,MAXFLOAT);      //utility info

  //store the expected intensity
  vis_image[imIndex[llid]]  = vis_rec;
  depth_cdf_image[imIndex[llid]]  = (* aux_args.depth_cdf) ;
  median_depth_image[imIndex[llid]]  = (* aux_args.median_depth) ;
}

void step_cell_render_depth(int data_ptr,
                             AuxArgs aux_args,
                             float depth,
                             float block_len)
{
  float prob = aux_args.alpha[data_ptr];
  float diff_omega= 1-prob;

  CONVERT_FUNC_FLOAT8(mixture,aux_args.mog[data_ptr])/NORM;
  float  weight3  = (1.0f-mixture.s2-mixture.s5);
  float PI = gauss_3_mixture_prob_density( aux_args.pixel_intensity,
                                         mixture.s0,
                                         mixture.s1,
                                         mixture.s2,
                                         mixture.s3,
                                         mixture.s4,
                                         mixture.s5,
                                         mixture.s6,
                                         mixture.s7,
                                         weight3 //(1.0f-mixture.s2-mixture.s5)
                                        );/* PI */

  float omega = (*aux_args.vis) * prob;

  if((*aux_args.depth_cdf)/(aux_args.depth_prob_norm) <= 0.5 && ((*aux_args.depth_cdf) + omega * PI)/(aux_args.depth_prob_norm) > 0.5f)
       (*aux_args.median_depth) = depth * block_len;

  (*aux_args.depth_cdf) += omega * PI;
  (*aux_args.vis)    *= diff_omega;
}

#endif



#ifdef REINIT_VIS2
__kernel void proc_norm_image (   __global float * depth_cdf_image,
                                  __global float* depth_cdf_norm_image,
                                  __global float* median_depth_image,
                                  __global float* vis_image,
                                  __global float* t_infty_image,
                                  __global uint4 * imgdims)
{
    // linear global id of the normalization imagee,
    int i=0;
    int j=0;
    i=get_global_id(0);
    j=get_global_id(1);
    float vis = vis_image[j*get_global_size(0) + i];
    float depth_cdf = depth_cdf_image[j*get_global_size(0) + i];
    float depth_cdf_norm = depth_cdf_norm_image[j*get_global_size(0) + i];

    if (i>=(*imgdims).z && j>=(*imgdims).w)
        return;

    float omega = 1-vis;
    float PI = 1.0f;
    if(depth_cdf/depth_cdf_norm <= 0.5 && (depth_cdf + omega * PI)/(depth_cdf_norm) > 0.5f)
       median_depth_image[j*get_global_size(0) + i] = t_infty_image[j*get_global_size(0) + i];

}
#endif // REINIT_VIS2




#ifdef RENDER_MEDIAN_DEPTH_WITHOUT_APP
//need to define a struct of type AuxArgs with auxiliary arguments
// to supplement cast ray args
typedef struct
{
  __global float* alpha;
  float* expdepth;
  float* expdepthsqr;
  float* probsum;
  float* t;
  float* vis;
  float* cdf;
} AuxArgs;

//forward declare cast ray (so you can use it)
void cast_ray(int,int,float,float,float,float,float,float,
              __constant RenderSceneInfo*, __global int4*,
              __local uchar16*, __constant uchar *,__local uchar *,
              float*, AuxArgs,float tnear, float tfar);
__kernel
void
render_depth( __constant  RenderSceneInfo    * linfo,
              __global    int4               * tree_array,
              __global    float              * alpha_array,
              __global    float4             * ray_origins,
              __global    float4             * ray_directions,
              __global    float              * exp_image,        // camera orign and SVD of inverse of camera matrix
              __global    float              * exp_sqr_image,    // input image and store vis_inf and pre_inf
              __global    float              * cdf_image,    // input image and store vis_inf and pre_inf
              __global    uint4              * exp_image_dims,   // sum of squares.
              __global    float              * output,
              __constant  uchar              * bit_lookup,
              __global    float              * vis_image,
              __global    float              * prob_image,
              __global    float              * t_image,
              __local     uchar16            * local_tree,
              __local     uchar              * cumsum,        // cumulative sum helper for data pointer
              __local     int                * imIndex)
{
  //----------------------------------------------------------------------------
  //get local id (0-63 for an 8x8) of this patch + image coordinates and camera
  // check for validity before proceeding
  //----------------------------------------------------------------------------
  uchar llid = (uchar)(get_local_id(0) + get_local_size(0)*get_local_id(1));
  int i=0,j=0;
  i=get_global_id(0);
  j=get_global_id(1);

  // check to see if the thread corresponds to an actual pixel as in some
  // cases #of threads will be more than the pixels.
  if (i>=(*exp_image_dims).z || j>=(*exp_image_dims).w)
    return;

   //Store image index (may save a register).  Also initialize VIS and expected_int
  imIndex[llid] = j*get_global_size(0)+i;

  //----------------------------------------------------------------------------
  // Calculate ray origin, and direction
  // (make sure ray direction is never axis aligned)
  //----------------------------------------------------------------------------
  float4 ray_o = ray_origins[ imIndex[llid] ];
  float4 ray_d = ray_directions[ imIndex[llid] ];
  float ray_ox, ray_oy, ray_oz, ray_dx, ray_dy, ray_dz;
  calc_scene_ray_generic_cam(linfo, ray_o, ray_d, &ray_ox, &ray_oy, &ray_oz, &ray_dx, &ray_dy, &ray_dz);

  //----------------------------------------------------------------------------
  // we know i,j map to a point on the image, have calculated ray
  // BEGIN RAY TRACE
  //----------------------------------------------------------------------------

  float expdepthsqr = 0.0f;
  float probsum     = prob_image[imIndex[llid]];
  float expdepth     = exp_image[imIndex[llid]];
  float vis_rec     = vis_image[imIndex[llid]];
  float t           = t_image[imIndex[llid]];
  float cdf         = cdf_image[imIndex[llid]];
  AuxArgs aux_args;
  aux_args.alpha  = alpha_array;
  aux_args.expdepth = &expdepth;
  aux_args.expdepthsqr = &expdepthsqr;
  aux_args.probsum = &probsum;
  aux_args.t = &t;
  aux_args.vis = &vis_rec;
  aux_args.cdf = &cdf;

  float vis = 1.0;
  cast_ray( i, j,
            ray_ox, ray_oy, ray_oz,
            ray_dx, ray_dy, ray_dz,
            linfo, tree_array,                                    //scene info
            local_tree, bit_lookup, cumsum, &vis, aux_args,0,MAXFLOAT);      //utility info

  //store the expected intensity
  exp_image[imIndex[llid]] = (* aux_args.expdepth);
  exp_sqr_image[imIndex[llid]] += (* aux_args.expdepthsqr)*linfo->block_len*linfo->block_len;
  prob_image[imIndex[llid]] = (* aux_args.probsum);
  //store visibility at the end of this block
  vis_image[imIndex[llid]]  = vis_rec;
  t_image[imIndex[llid]]  = (* aux_args.t) ;
  cdf_image[imIndex[llid]]  = (* aux_args.cdf) ;
}

void step_cell_render_depth2(float depth,
                             float block_len,
                             __global float  * alpha_data,
                             int      data_ptr,
                             float    d,
                             float  * vis,
                             float  * expected_depth,
                             float  * expected_depth_square,
                             float  * probsum,
                             float  * t,
                             float  * cdf)
{
  float alpha = alpha_data[data_ptr];
  // float diff_omega=exp(-alpha*d);
  float diff_omega=1-alpha;
  float omega=(*vis) * (1.0f - diff_omega);
  (*probsum)+=omega;
  (*vis)    *= diff_omega;
  if((*cdf) < 0.5f && (*cdf)+omega > 0.5f)
    (*expected_depth) = depth * block_len ;
  // (*expected_depth) += depth * block_len * omega;

  (*expected_depth_square)+=depth*depth*omega;
  (*t)=depth*block_len;
  //(*t) = depth;
  (*cdf)+=omega;
}
#endif //RENDER_MEDIAN_DEPTH_WITHOUT_APP
