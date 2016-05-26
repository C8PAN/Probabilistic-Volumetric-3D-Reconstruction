#ifdef SEGLEN
//Update step cell functor::seg_len
void step_cell_seglen(AuxArgs aux_args, int data_ptr, uchar llid, float d)
{
    int seg_int = convert_int_rte(d * SEGLEN_FACTOR);
    atom_add(&aux_args.seg_len[data_ptr], seg_int);
    int cum_obs = convert_int_rte(d * aux_args.obs * SEGLEN_FACTOR );
    atom_add(&aux_args.mean_obs[data_ptr], cum_obs);
}
#endif // SEGLEN

#ifdef PREINF
void pre_infinity_opt(  float    cum_len,
                        float    PI,
                        float  * vis_inf,
                        float  * pre_inf,
                        float    alpha,
                        float    cur_log_msg,
                        float8   mixture,
                        float    weight3,
                        float    pos_log_msg_sum,
                        __global float * output)
{
    /* if total length of rays is too small, do nothing */
    if (cum_len > 1.0e-10f)
    {
      if(!isnan(cur_log_msg) ) { //subtract current message from sum of messages only if it has been added before, i.e., if the message is valid.
        pos_log_msg_sum -= cur_log_msg;
      }
   

      float max_msg = max(pos_log_msg_sum, 0.0f);
      float neg = exp(0.0f-max_msg);
      float pos = exp(pos_log_msg_sum-max_msg);
      float neg_normalized =  neg /  (neg + pos);

      neg_normalized = clamp(neg_normalized, EPSILON,1.0f-EPSILON);


      float diff_omega = neg_normalized;


      ////////////////
      float vis_prob_end = (*vis_inf) * diff_omega;

      // update pre
      (*pre_inf) += ((*vis_inf) - vis_prob_end) *  PI;
      // (*pre_inf) = max((*pre_inf) , 1-diff_omega);

      /* updated visibility probability */
      (*vis_inf) = vis_prob_end;
  }
}

//preinf step cell functor
void step_cell_preinf(AuxArgs aux_args, int data_ptr, uchar llid, float d)
{
    //keep track of cells being hit
    ////cell data, i.e., alpha and app model is needed for some passes
    float  alpha    = aux_args.alpha[data_ptr];

    //get current message
    MSG_TYPE cur_msg    = aux_args.msg[data_ptr];

    float pos_log_msg_sum    = aux_args.pos_log_msg_sum[data_ptr];

    // float prior = aux_args.p_init_[data_ptr];

    CONVERT_FUNC_FLOAT8(mixture,aux_args.mog[data_ptr])/NORM;
    float  weight3  = (1.0f-mixture.s2-mixture.s5);

    int cum_int = aux_args.seg_len[data_ptr];
    float cum_len = convert_float(cum_int) / SEGLEN_FACTOR;

    float PI    = aux_args.PI_integrals[data_ptr];


    //calculate pre_infinity denomanator (shape of image)
    pre_infinity_opt( cum_len,
                      PI,
                      aux_args.vis_inf,
                      aux_args.pre_inf,
                      alpha,
                      cur_msg,
                      mixture,
                      weight3,
                      pos_log_msg_sum,
                      aux_args.output);
}
#endif // PREINF

#ifdef BAYES
/* bayes ratio independent functor (for independent rays) */
void bayes_ratio_ind( float alpha,
                      float  d,
                      float  cum_len,
                      float  PI,
                      float  cur_log_msg,
                      float  norm,
                      float* ray_pre,
                      float* ray_vis,
                      float* vis_cont,
                      float* new_msg,
                      float    pos_log_msg_sum)
{
    /* Compute PI for all threads */
    if (cum_len >1.0e-10f) {    /* if  too small, do nothing */

      if(!isnan(cur_log_msg) )  //subtract current message from sum of messages only if it has been added before, i.e., if the message is valid.
        pos_log_msg_sum -= cur_log_msg;
      
      
      float max_msg = max(pos_log_msg_sum, 0.0f);
      float neg = exp(0.0f-max_msg);
      float pos = exp(pos_log_msg_sum-max_msg);
      float neg_normalized =  neg /  (neg + pos);

      neg_normalized = clamp(neg_normalized, EPSILON,1.0f-EPSILON);

      float diff_omega = neg_normalized;

      float pos_msg;
      float pos_msg_unnormalized = (*ray_pre) + PI*(*ray_vis);
      
      float neg_msg_unnormalized = (*ray_pre) + (norm - (*ray_pre) - PI*(*ray_vis)*(1-diff_omega)) / diff_omega;
      pos_msg = pos_msg_unnormalized / (pos_msg_unnormalized + neg_msg_unnormalized); //normalize it here to avoid num. prob. later

      //update msg
      (*new_msg) = pos_msg * d;

      (*vis_cont) = ((*ray_vis))  * d;

    
        //update ray_pre and ray_vis
      (*ray_pre) += (*ray_vis)*(1-diff_omega)*PI;
      (*ray_vis) *= diff_omega;

  }
}

//bayes step cell functor
void step_cell_bayes(AuxArgs aux_args, int data_ptr, uchar llid, float d)
{
    //slow beta calculation ----------------------------------------------------
    float  alpha    = aux_args.alpha[data_ptr];
    CONVERT_FUNC_FLOAT8(mixture,aux_args.mog[data_ptr])/NORM;
    float weight3   = (1.0f-mixture.s2-mixture.s5);

    //get current message
    MSG_TYPE cur_msg    = aux_args.msg[data_ptr];

    float pos_log_msg_sum    = aux_args.pos_log_msg_sum[data_ptr];

    int cum_int = aux_args.seg_len[data_ptr];
    float cum_len = convert_float(cum_int) / SEGLEN_FACTOR;

    float PI    = aux_args.PI_integrals[data_ptr];

    float  vis_cont, new_msg;
    bayes_ratio_ind( alpha,
                     d,
                     cum_len,
                     PI,
                     cur_msg,
                     aux_args.norm,
                     aux_args.ray_pre,
                     aux_args.ray_vis,
                     &vis_cont,
                     &new_msg,
                     pos_log_msg_sum);

    int vis_int  = convert_int_rte((vis_cont) * SEGLEN_FACTOR);
    atom_add(&aux_args.vis_array[data_ptr], vis_int);

    int new_msg_int  = convert_int_rte(new_msg * SEGLEN_FACTOR);
    atom_add(&aux_args.new_msg_array[data_ptr], new_msg_int);
}
#endif // BAYES