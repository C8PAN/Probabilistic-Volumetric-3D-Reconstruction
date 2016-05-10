//THIS IS UPDATE BIT SCENE OPT
//Created Sept 30, 2010,
//Implements the parallel work group segmentation algorithm.
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics: enable
#if NVIDIA
#pragma OPENCL EXTENSION cl_khr_gl_sharing : enable
#endif

#define EPSILON 1.0e-7f

#ifdef SEGLEN
typedef struct
{
    __global int* seg_len;
    __global int* mean_obs;
    float   obs;
    float * ray_len;
    __constant RenderSceneInfo * linfo;

} AuxArgs;

//forward declare cast ray (so you can use it)
void cast_ray(int,int,float,float,float,float,float,float,__constant RenderSceneInfo*,
              __global int4*,local uchar16*,constant uchar *,local uchar *,float*,AuxArgs, float tnear, float tfar);
__kernel
    void
    seg_len_main(__constant  RenderSceneInfo    * linfo,
    __global    int4               * tree_array,       // tree structure for each block
    __global    float              * alpha_array,      // alpha for each block
    __global    int                * aux_array0,       // aux data array (four aux arrays strung together)
    __global    int                * aux_array1,       // aux data array (four aux arrays strung together)
    __constant  uchar              * bit_lookup,       // used to get data_index
    __global    float4             * ray_origins,
    __global    float4             * ray_directions,
    __global    float              * nearfarplanes,
    __global    uint4              * imgdims,          // dimensions of the input image
    __global    float              * in_image,         // the input image
    __global    float              * output,
    __local     uchar16            * local_tree,       // cache current tree into local memory
    __local     uchar              * cumsum )          // cumulative sum for calculating data pointer
{
    //get local id (0-63 for an 8x8) of this patch
    uchar llid = (uchar)(get_local_id(0) + get_local_size(0)*get_local_id(1));

    //----------------------------------------------------------------------------
    // get image coordinates and camera,
    // check for validity before proceeding
    //----------------------------------------------------------------------------
    int i=0,j=0;
    i=get_global_id(0);
    j=get_global_id(1);
    int imIndex = j*get_global_size(0) + i;

    //grab input image value (also holds vis)
    float obs = in_image[imIndex];
    float vis = 1.0f;  //no visibility in this pass
    barrier(CLK_LOCAL_MEM_FENCE);

    // cases #of threads will be more than the pixels.
    if (i>=(*imgdims).z || j>=(*imgdims).w || i<(*imgdims).x || j<(*imgdims).y || obs < 0.0f)
        return;

    //----------------------------------------------------------------------------
    // we know i,j map to a point on the image,
    // BEGIN RAY TRACE
    //----------------------------------------------------------------------------
    float4 ray_o = ray_origins[ imIndex ];
    float4 ray_d = ray_directions[ imIndex ];
    float ray_ox, ray_oy, ray_oz, ray_dx, ray_dy, ray_dz;
    calc_scene_ray_generic_cam(linfo, ray_o, ray_d, &ray_ox, &ray_oy, &ray_oz, &ray_dx, &ray_dy, &ray_dz);

    //----------------------------------------------------------------------------
    // we know i,j map to a point on the image, have calculated ray
    // BEGIN RAY TRACE
    //----------------------------------------------------------------------------
    AuxArgs aux_args;
    aux_args.linfo    = linfo;
    aux_args.seg_len  = aux_array0;
    aux_args.mean_obs = aux_array1;
    aux_args.obs = obs;
    float nearplane = nearfarplanes[0]/linfo->block_len;
    float farplane = nearfarplanes[1]/linfo->block_len;


    cast_ray( i, j,
        ray_ox, ray_oy, ray_oz,
        ray_dx, ray_dy, ray_dz,
        linfo, tree_array,                                  //scene info
        local_tree, bit_lookup, cumsum, &vis, aux_args,nearplane,farplane);    //utility info
}
#endif // SEGLEN

#ifdef PREINF
typedef struct
{
    __global float* alpha;
    __global MSG_TYPE* msg;
    __global MOG_TYPE * mog;
    __global int* seg_len;
    __global int* mean_obs;
    float* vis_inf;
    float* pre_inf;
    unsigned img_idx;

    __global float* pos_log_msg_sum;
    __global float* output;
    __constant RenderSceneInfo * linfo;
    // bool use_unary,init_msgs; 
} AuxArgs;

//forward declare cast ray (so you can use it)
void cast_ray(int,int,float,float,float,float,float,float,__constant RenderSceneInfo*,
              __global int4*,local uchar16*,constant uchar *,local uchar *,float*,AuxArgs, float tnear, float tfar);

__kernel
    void
    pre_inf_main(__constant  RenderSceneInfo    * linfo,
    __global    int4               * tree_array,       // tree structure for each block
    __global    float              * alpha_array,      // alpha for each block
    __global    MSG_TYPE           * msg_array,        // messages for each block
    __global    MOG_TYPE           * mixture_array,    // mixture for each block
    __global    ushort4            * num_obs_array,    // num obs for each block
    __global    int                * aux_array0,       // four aux arrays strung together
    __global    int                * aux_array1,       // four aux arrays strung together
    __global    float              * pos_log_msg_sum,       // four aux arrays strung together
    __constant  uchar              * bit_lookup,       // used to get data_index
    __global    float4             * ray_origins,
    __global    float4             * ray_directions,
    __global    float              * nearfarplanes,
    __global    uint4              * imgdims,          // dimensions of the input image
    __global    float              * vis_image,        // visibility image
    __global    float              * pre_image,        // preinf image
    __global    unsigned           * image_idx,        // image index
    __global    float              * output,
    __local     uchar16            * local_tree,       // cache current tree into local memory
    __local     uchar              * cumsum )          // cumulative sum for calculating data pointer
{
    //get local id (0-63 for an 8x8) of this patch
    uchar llid = (uchar)(get_local_id(0) + get_local_size(0)*get_local_id(1));

    //----------------------------------------------------------------------------
    // get image coordinates and camera,
    // check for validity before proceeding
    //----------------------------------------------------------------------------
    int i=0,j=0;
    i=get_global_id(0);
    j=get_global_id(1);

    // check to see if the thread corresponds to an actual pixel as in some
    // cases #of threads will be more than the pixels.
    if (i>=(*imgdims).z || j>=(*imgdims).w || i<(*imgdims).x || j<(*imgdims).y)
        return;
    //float4 inImage = in_image[j*get_global_size(0) + i];
    float vis_inf = vis_image[j*get_global_size(0) + i];
    float pre_inf = pre_image[j*get_global_size(0) + i];

    if (vis_inf <0.0)
        return;
    //vis for cast_ray, never gets decremented so no cutoff occurs
    float vis = 1.0f;
    barrier(CLK_LOCAL_MEM_FENCE);

    //----------------------------------------------------------------------------
    // we know i,j map to a point on the image,
    // BEGIN RAY TRACE
    //----------------------------------------------------------------------------
    float4 ray_o = ray_origins[ j*get_global_size(0) + i ];
    float4 ray_d = ray_directions[ j*get_global_size(0) + i ];
    float ray_ox, ray_oy, ray_oz, ray_dx, ray_dy, ray_dz;
    calc_scene_ray_generic_cam(linfo, ray_o, ray_d, &ray_ox, &ray_oy, &ray_oz, &ray_dx, &ray_dy, &ray_dz);

    //----------------------------------------------------------------------------
    // we know i,j map to a point on the image, have calculated ray
    // BEGIN RAY TRACE
    //----------------------------------------------------------------------------
    AuxArgs aux_args;
    aux_args.linfo   = linfo;
    aux_args.alpha   = alpha_array;
    aux_args.msg     = msg_array;
    aux_args.mog     = mixture_array;
    aux_args.seg_len   = aux_array0;
    aux_args.mean_obs  = aux_array1;
    aux_args.vis_inf = &vis_inf;
    aux_args.pre_inf = &pre_inf;
    aux_args.img_idx = (*image_idx);
    aux_args.output = output;
    // aux_args.p_init_ = p_init;
    aux_args.pos_log_msg_sum  = pos_log_msg_sum;
    // aux_args.use_unary  = (*use_unary);
    // aux_args.init_msgs  = (*init_msgs);
 
    float nearplane = nearfarplanes[0]/linfo->block_len;
    float farplane = nearfarplanes[1]/linfo->block_len;

    cast_ray( i, j,
        ray_ox, ray_oy, ray_oz,
        ray_dx, ray_dy, ray_dz,
        linfo, tree_array,                                  //scene info
        local_tree, bit_lookup, cumsum, &vis, aux_args,nearplane,farplane);    //utility info

    //store the vis_inf/pre_inf in the image
    vis_image[j*get_global_size(0)+i] = vis_inf;
    pre_image[j*get_global_size(0)+i] = pre_inf;
}
#endif // PREINF

#ifdef BAYES
typedef struct
{
    __global float*   alpha;
    __global MOG_TYPE * mog;
    __global int* seg_len;
    __global int* mean_obs;
    __global int* vis_array;
    __global int* depth_array;
    __global float* pos_log_msg_sum;
    __global int* new_msg_array;
    __global MSG_TYPE* msg;
    // __global float* p_init_;
    __global float* output;
    unsigned img_idx;
    float   norm;
    float*  ray_vis;
    float*  ray_pre;
    // bool use_unary, init_msgs; 

    __constant RenderSceneInfo * linfo;
} AuxArgs;

//forward declare cast ray (so you can use it)
void cast_ray(int,int,float,float,float,float,float,float,__constant RenderSceneInfo*,
              __global int4*,local uchar16*,constant uchar *,local uchar *,float*,AuxArgs, float tnear, float tfar);

__kernel
    void
    bayes_main(__constant  RenderSceneInfo    * linfo,
    __global    int4               * tree_array,        // tree structure for each block
    __global    float              * alpha_array,       // alpha for each block
    __global    MSG_TYPE           * msg_array,         // messages for each block
    __global    MOG_TYPE           * mixture_array,     // mixture for each block
    __global    ushort4            * num_obs_array,     // num obs for each block
    __global    int                * aux_array0,        // four aux arrays strung together
    __global    int                * aux_array1,        // four aux arrays strung together
    __global    int                * aux_array2,        // four aux arrays strung together
    __global    int                * aux_array3,        // four aux arrays strung together
    __global    float              * pos_log_msg_sum,   // four aux arrays strung together
    __global    int                * new_msg,           // four aux arrays strung together
    __constant  uchar              * bit_lookup,        // used to get data_index
    __global    float4             * ray_origins,
    __global    float4             * ray_directions,
    __global    float              * nearfarplanes,
    __global    uint4              * imgdims,           // dimensions of the input image
    __global    float              * vis_image,         // visibility image (for keeping vis across blocks)
    __global    float              * pre_image,         // preinf image (for keeping pre across blocks)
    __global    float              * norm_image,        // norm image (for bayes update normalization factor)
    __global    unsigned           * image_idx,         // image index
    __global    float              * output,
    __local     uchar16            * local_tree,        // cache current tree into local memory
    __local     uchar              * cumsum)            // cumulative sum for calculating data pointer
{

    //get local id (0-63 for an 8x8) of this patch
    uchar llid = (uchar)(get_local_id(0) + get_local_size(0)*get_local_id(1));

    //----------------------------------------------------------------------------
    // get image coordinates and camera,
    // check for validity before proceeding
    //----------------------------------------------------------------------------
    int i=0,j=0;
    i=get_global_id(0);
    j=get_global_id(1);

    // check to see if the thread corresponds to an actual pixel as in some
    // cases #of threads will be more than the pixels.
    if (i>=(*imgdims).z || j>=(*imgdims).w || i<(*imgdims).x || j<(*imgdims).y)
        return;
    float vis0 = 1.0f;
    float norm = norm_image[j*get_global_size(0) + i];
    float vis = vis_image[j*get_global_size(0) + i];
    float pre = pre_image[j*get_global_size(0) + i];
    if (vis <0.0)
        return;
    barrier(CLK_LOCAL_MEM_FENCE);
    //----------------------------------------------------------------------------
    // we know i,j map to a point on the image,
    // BEGIN RAY TRACE
    //----------------------------------------------------------------------------
    float4 ray_o = ray_origins[ j*get_global_size(0) + i ];
    float4 ray_d = ray_directions[ j*get_global_size(0) + i ];
    float ray_ox, ray_oy, ray_oz, ray_dx, ray_dy, ray_dz;
    //calc_scene_ray(linfo, camera, i, j, &ray_ox, &ray_oy, &ray_oz, &ray_dx, &ray_dy, &ray_dz);
    calc_scene_ray_generic_cam(linfo, ray_o, ray_d, &ray_ox, &ray_oy, &ray_oz, &ray_dx, &ray_dy, &ray_dz);

    //----------------------------------------------------------------------------
    // we know i,j map to a point on the image, have calculated ray
    // BEGIN RAY TRACE
    //----------------------------------------------------------------------------
    AuxArgs aux_args;
    aux_args.linfo      = linfo;
    aux_args.alpha      = alpha_array;
    aux_args.mog        = mixture_array;
    aux_args.seg_len    = aux_array0;
    aux_args.mean_obs   = aux_array1;
    aux_args.vis_array  = aux_array2;
    aux_args.depth_array  = aux_array3;
    aux_args.new_msg_array = new_msg;
    aux_args.norm = norm;
    aux_args.ray_vis = &vis;
    aux_args.ray_pre = &pre;
    aux_args.msg     = msg_array;
    aux_args.output     = output;
    aux_args.img_idx = (*image_idx);
    aux_args.pos_log_msg_sum  = pos_log_msg_sum;

    float nearplane = nearfarplanes[0]/linfo->block_len;
    float farplane = nearfarplanes[1]/linfo->block_len;

    cast_ray( i, j,
        ray_ox, ray_oy, ray_oz,
        ray_dx, ray_dy, ray_dz,
        linfo, tree_array,                                  //scene info
        local_tree, bit_lookup, cumsum, &vis0, aux_args,nearplane,farplane);    //utility info

    //write out vis and pre
    vis_image[j*get_global_size(0)+i] = vis;
    pre_image[j*get_global_size(0)+i] = pre;
}
#endif // BAYES

#ifdef PROC_NORM
// normalize the pre_inf image...
//
__kernel
    void
    proc_norm_image (  __global float* norm_image,
    __global float * in_image,
    __global float* vis_image,
    __global float* pre_image,
    __global uint4 * imgdims)
{
    // linear global id of the normalization imagee,
    int i=0;
    int j=0;
    i=get_global_id(0);
    j=get_global_id(1);
    float img_obs = in_image[j*get_global_size(0) + i];
    float vis = vis_image[j*get_global_size(0) + i];
    float intensity = in_image[j*get_global_size(0) + i];

    if (i>=(*imgdims).z && j>=(*imgdims).w && vis<0.0f)
        return;

    float pre = pre_image[j*get_global_size(0) + i];
    float norm = (pre+vis);
    norm_image[j*get_global_size(0) + i] = norm;

    // the following  quantities have to be re-initialized before
    // the bayes_ratio kernel is executed
    vis_image[j*get_global_size(0) + i] = 1.0f; // initial vis = 1.0f
    pre_image[j*get_global_size(0) + i] = 0.0f; // initial pre = 0.0
}
#endif // PROC_NORM

#ifdef UPDATE_BIT_SCENE_MAIN
// Update each cell using its aux data
//

void update_cell(float16 * data, float4 aux_data,float t_match, float init_sigma, float min_sigma)
{
    float mu0 = (*data).s1, sigma0 = (*data).s2, w0 = (*data).s3;
    float mu1 = (*data).s5, sigma1 = (*data).s6, w1 = (*data).s7;
    float mu2 = (*data).s9, sigma2 = (*data).sa;
    float w2=0.0f;


    if (w0>0.0f && w1>0.0f)
        w2=1-(*data).s3-(*data).s7;

    short Nobs0 = (short)(*data).s4, Nobs1 = (short)(*data).s8, Nobs2 = (short)(*data).sb;
    float Nobs_mix = (*data).sc;

    update_gauss_3_mixture(aux_data.y,              //mean observation
                           aux_data.w,              //cell_visability
                           t_match,
                           init_sigma,min_sigma,
                           &mu0,&sigma0,&w0,&Nobs0,
                           &mu1,&sigma1,&w1,&Nobs1,
                           &mu2,&sigma2,&w2,&Nobs2,
                           &Nobs_mix);

    //float beta = aux_data.z; //aux_data.z/aux_data.x;
    //clamp(beta,0.5f,2.0f);
    //(*data).s0 *= beta;
    (*data).s1=mu0; (*data).s2=sigma0, (*data).s3=w0;(*data).s4=(float)Nobs0;
    (*data).s5=mu1; (*data).s6=sigma1, (*data).s7=w1;(*data).s8=(float)Nobs1;
    (*data).s9=mu2; (*data).sa=sigma2, (*data).sb=(float)Nobs2;
    (*data).sc=(float)Nobs_mix;
}


__kernel
void
update_bit_scene_main(__global RenderSceneInfo  * info,
                      __global float            * alpha_array,
                      __global MOG_TYPE         * mixture_array,
                      __global ushort4          * nobs_array,
                      __global int              * aux_array0,
                      __global int              * aux_array1,
                      __global int              * aux_array2,       // vis
                      __global int              * aux_array3,       // depth
                      __global    float         * pos_log_msg_sum,  // four aux arrays strung together
                      __global int              * new_msg,          // four aux arrays strung together
                      __global MSG_TYPE         * msg_array,        // messages for each block
                      __global unsigned         * image_idx,        // image index
                      // __global bool             * use_unary,        // 
                      __global float            * mog_var,          // if 0 or less, variable var, otherwise use as fixed var
                      __global bool             * update_app,          // if 0 or less, variable var, otherwise use as fixed var
                      __global bool             * update_occ,          // if 0 or less, variable var, otherwise use as fixed var
                      __global float            * output)
{
    int gid=get_global_id(0);
    int datasize = info->data_len ;//* info->num_buffer;
    if (gid<datasize)
    {
        //if alpha is less than zero don't update
        float  alpha    = alpha_array[gid];

        //get cell cumulative length and make sure it isn't 0
        int len_int = aux_array0[gid];
        float cum_len  = convert_float(len_int);//SEGLEN_FACTOR;

        //update cell if alpha and cum_len are greater than 0
        // if ( (cum_len / SEGLEN_FACTOR) > 1e-10f && alpha >= 0.0f)
        if ( (cum_len / SEGLEN_FACTOR) > 1e-10f )
        {
            int obs_int = aux_array1[gid];
            int vis_int = aux_array2[gid];
            int depth_int= aux_array3[gid];
            int new_msg_int= new_msg[gid];
            float mean_obs = convert_float(obs_int) / convert_float(len_int);
            float cell_vis  = convert_float(vis_int) / convert_float(len_int);
            // int depth  = convert_int( convert_float(depth_int) / convert_float(len_int));
            float cell_new_msg = convert_float(new_msg_int) / (convert_float(len_int));

            if(*update_occ) {
                cell_new_msg = clamp(cell_new_msg, EPSILON,1.0f-EPSILON);
                float new_log_msg  = log(cell_new_msg) - log(1-cell_new_msg);

                //save log sums
                float old_msg = msg_array[gid];
                float updated_pos_log_msg_sum = pos_log_msg_sum[gid] + new_log_msg;

                if(!isnan(old_msg)) 
                    updated_pos_log_msg_sum -= old_msg;
                
                pos_log_msg_sum[gid] = updated_pos_log_msg_sum;

                //save msg
                msg_array[gid] = new_log_msg;


                // compute new alpha
                float max_msg = max(updated_pos_log_msg_sum, 0.0f);
                float neg = exp(0.0f-max_msg);
                float pos = exp(updated_pos_log_msg_sum-max_msg);

                //write alpha if update alpha is 0
                alpha_array[gid] = pos / (pos + neg);
            }

            if(*update_app) {
                ////////////
                float4 aux_data = (float4) (cum_len, mean_obs, 0, cell_vis);
                float4 nobs     = convert_float4(nobs_array[gid]);
                CONVERT_FUNC_FLOAT8(mixture,mixture_array[gid])/NORM;
                float16 data = (float16) (alpha,
                                         (mixture.s0), (mixture.s1), (mixture.s2), (nobs.s0),
                                         (mixture.s3), (mixture.s4), (mixture.s5), (nobs.s1),
                                         (mixture.s6), (mixture.s7), (nobs.s2), (nobs.s3/100.0),
                                         0.0, 0.0, 0.0);
                //use aux data to update cells
                update_cell(&data, aux_data, 2.5f, 0.10f, 0.05f);


                //set appearance model (figure out if var is fixed or not)
                float8 post_mix       = (float8) (data.s1, data.s2, data.s3,
                                                  data.s5, data.s6, data.s7,
                                                  data.s9, data.sa)*(float) NORM;
                float4 post_nobs      = (float4) (data.s4, data.s8, data.sb, data.sc*100.0);
                //check if mog_var is fixed, if so, overwrite variance in post_mix
                if (*mog_var > 0.0f) {
                    post_mix.s1 = (*mog_var) * (float) NORM;
                    post_mix.s4 = (*mog_var) * (float) NORM;
                    post_mix.s7 = (*mog_var) * (float) NORM;
                }

                //reset the cells in memory
                CONVERT_FUNC_SAT_RTE(mixture_array[gid],post_mix);
                nobs_array[gid] = convert_ushort4_sat_rte(post_nobs);
            }
        }
//         else
//             alpha_array[gid] = -1.0f;

        //clear out aux data
        aux_array0[gid] = 0;
        aux_array1[gid] = 0;
        aux_array2[gid] = 0;
        aux_array3[gid] = 0;
        new_msg[gid] = 0;
    }
}

#endif // UPDATE_BIT_SCENE_MAIN


