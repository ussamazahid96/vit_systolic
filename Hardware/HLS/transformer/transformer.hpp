#ifndef __TRANSFORMER__HPP__
#define __TRANSFORMER__HPP__

#include "hls_stream.h"


template<typename TW, typename TS1>
void DoHFCMemInit(hls::stream<ap_uint<BUSWIDTH>> &hfc_weights_stream,
				  TW &HFCWeights, TS1 &HFC_scale)
{
#pragma HLS DATAFLOW
	hls::stream<ap_uint<PACKEDWIDTH>> inter("inter");
#pragma HLS STREAM depth=1 type=fifo variable=inter	
	
	const unsigned int words = HFCPEs*HFCWMEM*PACKEDWIDTH / BUSWIDTH;
	StreamingDataWidthConverter_Batch<BUSWIDTH, PACKEDWIDTH, words>(hfc_weights_stream, inter, 1);
	StreamingAttnMemInit<HFCPEs, HFCWMEM>(inter, HFCWeights);
	HFC_scale.scale = 0.0625;
}


template<unsigned int pe, unsigned int wmem, typename TW, typename TO>
void weights2Stream(TW const &W_in, hls::stream<TO> &out)
{
	for(int i = 0 ; i < pe; i++ )
	{
		for(int mem =0;mem<wmem;mem++)
		{
			ap_uint<PACKEDWIDTH> elem = W_in.m_weights[i][mem];
			out.write(elem);
		}
	}
}

template<typename TW, typename TS1>
void DoHFCCompute(hls::stream<ap_uint<DIM_HEAD*emb_out_t::width>> output_stream[HEADS],
				  hls::stream<ap_uint<BUSWIDTH>> &input_skip_stream,
				  ap_uint<BUSWIDTH> *output, 
				  TW const &HFCWeights, TS1 const &HFC_scale)
{

#pragma HLS DATAFLOW
	hls::stream<ap_uint<HEADS*DIM_HEAD*emb_out_t::width>> combined_stream("combined_stream");
#pragma HLS STREAM depth=64 type=fifo variable=combined_stream
	hls::stream<ap_uint<BUSWIDTH>> hfc_out_stream("hfc_out_stream");
#pragma HLS STREAM depth=1 type=fifo variable=hfc_out_stream
	hls::stream<ap_uint<BUSWIDTH>> skip_out_stream("skip_out_stream");
#pragma HLS STREAM depth=1 type=fifo variable=skip_out_stream

	const unsigned int outputBits = EMB_DIM*NUM_TOKENS*hfc_out_t::width;

	CombineStream<HEADS, DIM_HEAD, NUM_TOKENS>(output_stream, combined_stream);
	StreamingFCLayer_Batch<HEADS*DIM_HEAD, EMB_DIM, HFCSIMDs, HFCPEs, Slice<emb_out_t>, Slice<ap_int<hfc_out_t::width>>, Identity>
	(combined_stream, hfc_out_stream, HFCWeights, HFC_scale, NUM_TOKENS, ap_resource_lut());
	AddStreams<BUSWIDTH, outputBits, Slice<hfc_out_t>, Slice<input_t>, Slice<skip_out_t>>(hfc_out_stream, input_skip_stream, skip_out_stream);
	Stream2Mem<BUSWIDTH, outputBits/8>(skip_out_stream, output);
}



template<typename TW, typename TS1, typename TS2, typename TS3>
void DoMemInit(hls::stream<ap_uint<BUSWIDTH>> &input_and_weights_stream,
	           hls::stream<ap_uint<PACKEDWIDTH>> &query_weights_stream,
	           hls::stream<ap_uint<PACKEDWIDTH>> &key_weights_stream,
	           hls::stream<ap_uint<PACKEDWIDTH>> &value_weights_stream,
			   TW &QWeights, TW &KWeights, TW & VWeights,
			   TS1 &qkv_scale, TS2 &tempurature, TS3 &emb_out_scale)
{
#pragma HLS DATAFLOW

//	StreamingNormMemInit<EMB_DIM, norm_w_t::width>(input_and_weights_stream, normW, normB);
	StreamingAttnMemInit<PEs, WMEM>(query_weights_stream, QWeights);
	StreamingAttnMemInit<PEs, WMEM>(key_weights_stream, KWeights);
	StreamingAttnMemInit<PEs, WMEM>(value_weights_stream, VWeights);

	qkv_scale.scale = 0.03125;
	tempurature.scale = 0.0263671875;
	emb_out_scale.scale = 1;
}

template<typename TW, typename TS1, typename TS2, typename TS3>
void DoCompute(hls::stream<ap_uint<BUSWIDTH>> &input_and_weights_stream,
			   hls::stream<ap_uint<DIM_HEAD*emb_out_t::width>> &output_stream,
			   TW const &QWeights, TW const &KWeights, TW const &VWeights,
			   TS1 const &qkv_scale, TS2 const &tempurature, TS3 const &emb_out_scale)
{
#pragma HLS DATAFLOW

	hls::stream<ap_uint<BUSWIDTH>> input_q("input_q");
#pragma HLS stream depth=1 type=fifo variable=input_q
	hls::stream<ap_uint<BUSWIDTH>> input_k("input_k");
#pragma HLS stream depth=1 type=fifo variable=input_k
	hls::stream<ap_uint<BUSWIDTH>> input_v("input_v");
#pragma HLS stream depth=1 type=fifo variable=input_v

	// needs more depth because the FC consuming it, has to wait for the streaming weights
	hls::stream<ap_uint<AttnSIMDs*qkv_t::width>> query_stream("query_stream");
#pragma HLS stream depth=512 type=fifo variable=query_stream

	// Has to have this width for streaming it as weights
	hls::stream<ap_uint<AttnSIMDs*qkv_t::width>> key_stream("key_stream");
#pragma HLS stream depth=1 type=fifo variable=key_stream
	// has to have a width of whole tile
	hls::stream<ap_uint<AttnPEs*AttnSIMDs*qkv_t::width>> key_tiled_stream("key_tiled_stream");
#pragma HLS stream depth=1 type=fifo variable=key_tiled_stream

	// Has to have this width packing into PEs (for transposition) for streaming it as weights
	hls::stream<ap_uint<LastFCPEs*qkv_t::width>> value_stream("value_stream");
#pragma HLS stream depth=1 type=fifo variable=value_stream
	// has to have a width of whole tile
	hls::stream<ap_uint<LastFCPEs*LastFCSIMDs*qkv_t::width>> value_tiled_stream("value_tiled_stream");
#pragma HLS stream depth=64 type=fifo variable=value_tiled_stream

	hls::stream<ap_uint<AttnPEs*attn_score_t::width>> attn_scores_stream("attn_scores_stream");
#pragma HLS stream depth=1 type=fifo variable=attn_scores_stream
	// needs more depth because the FC consuming it, has to wait for the streaming weights
	hls::stream<ap_uint<AttnPEs*attn_prob_t::width>> attn_prob_stream("attn_prob_stream");
#pragma HLS stream depth=1 type=fifo variable=attn_prob_stream


	const unsigned int inputBits  = EMB_DIM*NUM_TOKENS*norm_out_t::width;

	TriplicateStreams<BUSWIDTH, inputBits/BUSWIDTH>(input_and_weights_stream, input_q, input_k, input_v);

	StreamingFCLayer_Batch<EMB_DIM, DIM_HEAD, SIMDs, PEs, Slice<norm_out_t>, Slice<ap_int<qkv_t::width>>, Identity>
	(input_q, query_stream, QWeights, qkv_scale, NUM_TOKENS, ap_resource_lut());
//
//	std::ofstream ofs("/home/uzahid/workspace/brevitas-template/debug/hls.txt");
//	int s = query_stream.size();
//	std::cout << "Size of the Stream = " << s << '\n';
//	for(int i=0;i<s;i++)
//	{
//		ap_uint<AttnSIMDs*qkv_t::width> el = query_stream.read();
//		qkv_t *fval = reinterpret_cast<qkv_t*>(&el);
//		for(int j=0;j<AttnSIMDs;j++)
//		{
//			ofs << fval[j] <<'\n';
//		}
//	}
//	ofs.close();
//	exit(0);


	StreamingFCLayer_Batch<EMB_DIM, DIM_HEAD, SIMDs, PEs, Slice<norm_out_t>, Slice<ap_int<qkv_t::width>>, Identity>
	(input_k, key_stream, KWeights, qkv_scale, NUM_TOKENS, ap_resource_lut());

	StreamingFCLayer_Batch<EMB_DIM, DIM_HEAD, SIMDs, PEs, Slice<norm_out_t>, Slice<ap_int<qkv_t::width>>, Identity>
	(input_v, value_stream, VWeights, qkv_scale, NUM_TOKENS, ap_resource_lut());

	StreamingAttnBufInit<DIM_HEAD, NUM_TOKENS, AttnPEs, AttnSIMDs, AttnWMEM, Slice<qkv_t>, Slice<ap_uint<AttnSIMDs*qkv_t::width>>>
	(key_stream, key_tiled_stream);

	StreamingWFCLayer_Batch<DIM_HEAD, NUM_TOKENS, AttnSIMDs, AttnPEs, Slice<qkv_t>, Slice<ap_int<attn_score_t::width>>, Identity, qkv_t>
	(query_stream, attn_scores_stream, key_tiled_stream, tempurature, NUM_TOKENS, ap_resource_dflt());

	SoftMaxLayer<DIM_HEAD, NUM_TOKENS, AttnPEs, Slice<attn_score_t>, Slice<attn_score_t_>, Slice<attn_prob_t>>
	(attn_scores_stream, attn_prob_stream);


	StreamingTransBufInit<DIM_HEAD, NUM_TOKENS, LastFCPEs, LastFCSIMDs, LastFCWMEM,
	Slice<ap_int<qkv_t::width>>, // here it has to be an int type rather than fixed point for proper packing in transposed fashion.
	Slice<ap_uint<LastFCSIMDs*qkv_t::width>>>
	(value_stream, value_tiled_stream);


	StreamingWFCLayer_Batch<DIM_HEAD, NUM_TOKENS, LastFCSIMDs, LastFCPEs, Slice<attn_prob_t>, Slice<ap_int<emb_out_t::width>>, Identity, qkv_t>
	(attn_prob_stream, output_stream, value_tiled_stream, emb_out_scale, NUM_TOKENS, ap_resource_dflt());

}

#endif
