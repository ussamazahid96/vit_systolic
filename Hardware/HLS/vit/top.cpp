#include "bnn-library.h"
#include "activations.hpp"

#include "config.hpp"

#include "layernorm.hpp"
#include "mem_init.hpp"
#include "softmax.hpp"
#include "transformer.hpp"


void AttnHead(hls::stream<ap_uint<BUSWIDTH>> &input_and_weights_stream,
		      hls::stream<ap_uint<PACKEDWIDTH>> &query_weights_stream,
		      hls::stream<ap_uint<PACKEDWIDTH>> &key_weights_stream,
		      hls::stream<ap_uint<PACKEDWIDTH>> &value_weights_stream,
			  hls::stream<ap_uint<DIM_HEAD*emb_out_t::width>> &output_stream,
			  bool MemInit)
{
#pragma HLS INTERFACE mode=s_axilite bundle=control name=MemInit port=MemInit
#pragma HLS INTERFACE mode=s_axilite bundle=control port=return

#pragma HLS INTERFACE mode=axis register_mode=both depth=1 port=input_and_weights_stream register
#pragma HLS INTERFACE mode=axis register_mode=both depth=1 port=query_weights_stream register
#pragma HLS INTERFACE mode=axis register_mode=both depth=1 port=key_weights_stream register
#pragma HLS INTERFACE mode=axis register_mode=both depth=1 port=value_weights_stream register
#pragma HLS INTERFACE mode=axis register_mode=both depth=1 port=output_stream register

	//static ap_uint<BUSWIDTH> normW[EMB_DIM * norm_w_t::width / BUSWIDTH];
	//static ap_uint<BUSWIDTH> normB[EMB_DIM * norm_b_t::width / BUSWIDTH];

	static FixedPointWeights<SIMDs, weight_t, PEs, WMEM> QWeights;
	static FixedPointWeights<SIMDs, weight_t, PEs, WMEM> KWeights;
	static FixedPointWeights<SIMDs, weight_t, PEs, WMEM> VWeights;

	static ScalingActivation<1<<(qkv_t::width-qkv_t::iwidth), ap_fixed<24, 12>, qkv_t> qkv_scale;
	static ScalingActivation<1<<(attn_score_t::width-attn_score_t::iwidth), ap_fixed<24, 12>, attn_score_t> tempurature;
	static ScalingActivation<1<<(emb_out_t::width-emb_out_t::iwidth), ap_fixed<24, 12>, emb_out_t> emb_out_scale;

// Memory Partitioning
#pragma HLS ARRAY_PARTITION dim=1 type=complete variable=QWeights.m_weights
#pragma HLS ARRAY_PARTITION dim=1 type=complete variable=KWeights.m_weights
#pragma HLS ARRAY_PARTITION dim=1 type=complete variable=VWeights.m_weights

	if(MemInit){
		DoMemInit(input_and_weights_stream,
				  query_weights_stream,
				  key_weights_stream,
				  value_weights_stream,
				  QWeights, KWeights, VWeights, qkv_scale, tempurature, emb_out_scale);
	}
	else{
		DoCompute(input_and_weights_stream, output_stream,
				  QWeights, KWeights, VWeights,
				  qkv_scale, tempurature, emb_out_scale);
	}

}


void TransRDMA(ap_uint<BUSWIDTH> *input_and_weights,
			   ap_uint<BUSWIDTH> *qhfc_weights,
			   ap_uint<BUSWIDTH> *key_weights,
		       ap_uint<BUSWIDTH> *value_weights,
			   bool MemInit,
			  hls::stream<ap_uint<BUSWIDTH>> &input_skip_stream,
			  hls::stream<ap_uint<BUSWIDTH>> input_and_weights_stream[HEADS],
			  hls::stream<ap_uint<PACKEDWIDTH>> query_weights_stream[HEADS],
			  hls::stream<ap_uint<PACKEDWIDTH>> key_weights_stream[HEADS],
			  hls::stream<ap_uint<PACKEDWIDTH>> value_weights_stream[HEADS],
			  hls::stream<ap_uint<BUSWIDTH>> &hfc_weights_stream)
{
#pragma HLS INTERFACE mode=s_axilite bundle=control name=input_and_weights port=input_and_weights
#pragma HLS INTERFACE mode=s_axilite bundle=control name=qhfc_weights port=qhfc_weights
#pragma HLS INTERFACE mode=s_axilite bundle=control name=key_weights port=key_weights
#pragma HLS INTERFACE mode=s_axilite bundle=control name=value_weights port=value_weights
#pragma HLS INTERFACE mode=s_axilite bundle=control name=MemInit port=MemInit
#pragma HLS INTERFACE mode=s_axilite bundle=control port=return

#pragma HLS INTERFACE m_axi offset=slave port=input_and_weights  bundle=input_and_weights_mem  depth=1
#pragma HLS INTERFACE m_axi offset=slave port=qhfc_weights bundle=qhfc_weights_mem depth=1
#pragma HLS INTERFACE m_axi offset=slave port=key_weights   bundle=key_weights_mem   depth=1
#pragma HLS INTERFACE m_axi offset=slave port=value_weights   bundle=value_weights_mem depth=1

#pragma HLS INTERFACE mode=axis register_mode=both depth=2048 port=input_skip_stream register
#pragma HLS INTERFACE mode=axis register_mode=both depth=1 port=input_and_weights_stream register
#pragma HLS INTERFACE mode=axis register_mode=both depth=1 port=query_weights_stream register
#pragma HLS INTERFACE mode=axis register_mode=both depth=1 port=key_weights_stream register
#pragma HLS INTERFACE mode=axis register_mode=both depth=1 port=value_weights_stream register
#pragma HLS INTERFACE mode=axis register_mode=both depth=1 port=hfc_weights_stream register

	const unsigned int inBits = NUM_TOKENS*EMB_DIM*norm_out_t::width;
	if(MemInit){
		WeightsSplit(input_and_weights, qhfc_weights, key_weights, value_weights,
					 input_and_weights_stream, query_weights_stream,
					 key_weights_stream, value_weights_stream,
					 hfc_weights_stream);
	}
	else{
		InputSplit<BUSWIDTH, HEADS, inBits>(input_and_weights, input_and_weights_stream, input_skip_stream);

	}

}

void TransWDMAs(hls::stream<ap_uint<DIM_HEAD*emb_out_t::width>> &output_stream, ap_uint<BUSWIDTH> *output)
{

#pragma HLS INTERFACE mode=s_axilite bundle=control name=output port=output
#pragma HLS INTERFACE mode=s_axilite bundle=control port=return

#pragma HLS INTERFACE mode=axis register_mode=both depth=64 port=output_stream register
#pragma HLS INTERFACE m_axi offset=slave port=output  bundle=output_mem  depth=1

	const unsigned int outputBits = DIM_HEAD*NUM_TOKENS*emb_out_t::width;
	hls::stream<ap_uint<BUSWIDTH>> inter("inter");

	StreamingDataWidthConverter_Batch<DIM_HEAD*emb_out_t::width, BUSWIDTH, NUM_TOKENS>(output_stream, inter, 1);
	Stream2Mem<BUSWIDTH, outputBits/8>(inter, output);

}

void HeadFC(hls::stream<ap_uint<BUSWIDTH>> &hfc_weights_stream, 
			hls::stream<ap_uint<DIM_HEAD*emb_out_t::width>> output_stream[HEADS],
			hls::stream<ap_uint<BUSWIDTH>> &input_skip_stream,
			ap_uint<BUSWIDTH> *output, 
			bool MemInit)
{
#pragma HLS INTERFACE mode=s_axilite bundle=control name=output port=output
#pragma HLS INTERFACE mode=s_axilite bundle=control name=MemInit port=MemInit
#pragma HLS INTERFACE mode=s_axilite bundle=control port=return

#pragma HLS INTERFACE mode=axis register_mode=both depth=1 port=input_skip_stream register
#pragma HLS INTERFACE mode=axis register_mode=both depth=1 port=hfc_weights_stream register
#pragma HLS INTERFACE mode=axis register_mode=both depth=1 port=output_stream register
#pragma HLS INTERFACE m_axi offset=slave port=output  bundle=output_mem  depth=1

	static FixedPointWeights<HFCSIMDs, weight_t, HFCPEs, HFCWMEM> HFCWeights;
	static ScalingActivation<1<<(hfc_out_t::width-hfc_out_t::iwidth), ap_fixed<24, 12>, hfc_out_t> hfc_scale;

	if(MemInit){
		DoHFCMemInit(hfc_weights_stream, HFCWeights, hfc_scale);
	}else{
		DoHFCCompute(output_stream, input_skip_stream, output, HFCWeights, hfc_scale);
	}

}

void TransAccl(ap_uint<BUSWIDTH> *input_and_weights,
		       ap_uint<BUSWIDTH> *qhfc_weights,
			   ap_uint<BUSWIDTH> *key_weights,
			   ap_uint<BUSWIDTH> *value_weights,
			   ap_uint<BUSWIDTH> *output,
			   bool MemInit,
			   int idx)
{
	hls::stream<ap_uint<BUSWIDTH>> isk_stream;
	hls::stream<ap_uint<BUSWIDTH>> iw_stream[HEADS];
	hls::stream<ap_uint<PACKEDWIDTH>> qw_stream[HEADS];
	hls::stream<ap_uint<PACKEDWIDTH>> kw_stream[HEADS];
	hls::stream<ap_uint<PACKEDWIDTH>> vw_stream[HEADS];
	hls::stream<ap_uint<DIM_HEAD*emb_out_t::width>> o_stream[HEADS];
	hls::stream<ap_uint<BUSWIDTH>> hfcw_stream;

	TransRDMA(input_and_weights, qhfc_weights, key_weights, value_weights, MemInit,
			  isk_stream, iw_stream, qw_stream, kw_stream, vw_stream, hfcw_stream);
	AttnHead(iw_stream[idx], qw_stream[idx], kw_stream[idx], vw_stream[idx], o_stream[idx], MemInit);
	if(!MemInit){
		TransWDMAs(o_stream[idx], output);
	}else{
		HeadFC(hfcw_stream, o_stream, isk_stream, output, MemInit);
	}
}
