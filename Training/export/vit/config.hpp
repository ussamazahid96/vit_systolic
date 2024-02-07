
#include "ap_int.h"
#include "ap_fixed.h"

#define INPUTBITWIDTH 8
#define EMB_DIM 512
#define NUM_TOKENS 64
#define DIM_HEAD 64
#define HEADS 2

#define BUSWIDTH 128
#define PACKEDWIDTH 64
#define PEs 8
#define SIMDs 32
#define WMEM (DIM_HEAD * EMB_DIM) / (PEs * SIMDs)

#define AttnPEs 8
#define AttnSIMDs 8
#define AttnWMEM (NUM_TOKENS * DIM_HEAD) / (AttnPEs * AttnSIMDs)

#define LastFCPEs 8
#define LastFCSIMDs 8
#define LastFCWMEM (NUM_TOKENS * DIM_HEAD) / (LastFCPEs * LastFCSIMDs)

#define HFCPEs 8
#define HFCSIMDs 32
#define HFCWMEM (EMB_DIM*HEADS*DIM_HEAD) / (HFCPEs * HFCSIMDs)


typedef ap_fixed<8,5,AP_TRN,AP_SAT>   	 input_t;
typedef ap_ufixed<8,2,AP_RND,AP_SAT> 	 norm_w_t;
typedef ap_fixed<8,1,AP_RND,AP_SAT>  	 norm_b_t;
typedef ap_fixed<8,3,AP_RND,AP_SAT>  	 norm_out_t;
typedef ap_int<2> 						 weight_t;
typedef ap_fixed<8,3,AP_TRN,AP_SAT> 	 qkv_t;
typedef ap_fixed<8,4,AP_TRN,AP_SAT> 	 attn_score_t;
typedef ap_ufixed<18, 10, AP_RND, AP_SAT> attn_score_t_;
typedef ap_fixed<8,1,AP_TRN,AP_SAT> 	 attn_prob_t;
typedef ap_fixed<8,3,AP_TRN,AP_SAT> 	 emb_out_t;
typedef ap_fixed<8,1,AP_TRN,AP_SAT> 	 hfc_out_t;
typedef ap_fixed<8,4,AP_TRN,AP_SAT>      skip_out_t;