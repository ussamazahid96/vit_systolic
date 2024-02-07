#include <iostream>
#include <stdint.h>

#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xnpy.hpp>
#include <xtensor/xpad.hpp>

#include "config.hpp"
#include "hls_stream.h"
#include "ap_int.h"

void TransAccl(ap_uint<BUSWIDTH> *input_and_weights,
		       ap_uint<BUSWIDTH> *qhfc_weights,
			   ap_uint<BUSWIDTH> *key_weights,
			   ap_uint<BUSWIDTH> *value_weights,
			   ap_uint<BUSWIDTH> *output,
			   bool MemInit,
			   int idx);

void HeadFC(hls::stream<ap_uint<BUSWIDTH>> &hfc_weights_stream, 
			hls::stream<ap_uint<DIM_HEAD*emb_out_t::width>> output_stream[HEADS], 
			hls::stream<ap_uint<BUSWIDTH>> &input_skip_stream,
			ap_uint<BUSWIDTH> *output, 
			bool MemInit);

int main(int argc, char* argv[])
{
	// load the data
	xt::xarray<float> input = xt::load_npy<float>( (std::string) argv[1] + "/input.npy");
	xt::xarray<float> input_norm = xt::load_npy<float>( (std::string) argv[1] + "/input_norm.npy");
	xt::xarray<uint64_t> qhfc_weights = xt::load_npy<uint64_t>( (std::string) argv[1] + "/qhfc_weights.npy");
	xt::xarray<uint64_t> key_weights = xt::load_npy<uint64_t>( (std::string) argv[1] + "/key_weights.npy");
	xt::xarray<uint64_t> value_weights = xt::load_npy<uint64_t>( (std::string) argv[1] + "/value_weights.npy");


	// cast to the quantized datatype
	xt::xarray<input_t> input_q = xt::cast<input_t>(input);
	xt::xarray<norm_out_t> input_norm_q = xt::cast<norm_out_t>(input_norm);

	// pack to uints
  	xt::xarray<ap_uint<input_t::width>>  input_int  = xt::cast<ap_uint<input_t::width>> (input_q  * (1<<(input_t::width-input_t::iwidth)));
  	xt::xarray<ap_uint<norm_out_t::width>>  input_norm_int  = xt::cast<ap_uint<norm_out_t::width>> (input_norm_q  * (1<<(norm_out_t::width-norm_out_t::iwidth)));
  	xt::xarray<ap_uint<input_t::width>> input_comb = xt::concatenate(xtuple(input_norm_int, input_int), 0);


	// buffer sizes
  	const unsigned int isize  = NUM_TOKENS*EMB_DIM*input_t::width / BUSWIDTH;
	const unsigned int osize  = NUM_TOKENS*DIM_HEAD*emb_out_t::width / BUSWIDTH;
	const unsigned int osize1 = NUM_TOKENS*EMB_DIM*hfc_out_t::width / BUSWIDTH;

	// pack to accelerator buffers
	ap_uint<BUSWIDTH> *input_buffer 	 	= reinterpret_cast<ap_uint<BUSWIDTH>*>(input_comb.data());
	ap_uint<BUSWIDTH> *qhfc_weights_buffer  = reinterpret_cast<ap_uint<BUSWIDTH>*>(qhfc_weights.data());
	ap_uint<BUSWIDTH> *key_weights_buffer   = reinterpret_cast<ap_uint<BUSWIDTH>*>(key_weights.data());
	ap_uint<BUSWIDTH> *value_weights_buffer = reinterpret_cast<ap_uint<BUSWIDTH>*>(value_weights.data());
	ap_uint<BUSWIDTH> output_buffer[HEADS*osize];
	ap_uint<BUSWIDTH> output_headfc[osize1];
	
	TransAccl(input_buffer, qhfc_weights_buffer, key_weights_buffer, value_weights_buffer, output_buffer, true, 0);
	// inference
	TransAccl(input_buffer, qhfc_weights_buffer, key_weights_buffer, value_weights_buffer, output_buffer, false, 0);

	// Memory Init
	TransAccl(input_buffer, qhfc_weights_buffer, key_weights_buffer, value_weights_buffer, &output_buffer[osize], true, 1);
	// inference
	TransAccl(input_buffer, qhfc_weights_buffer, key_weights_buffer, value_weights_buffer, &output_buffer[osize], false, 1);


	ap_uint<DIM_HEAD*emb_out_t::width> *output_comb = reinterpret_cast<ap_uint<DIM_HEAD*emb_out_t::width>*>(output_buffer);
	hls::stream<ap_uint<DIM_HEAD*emb_out_t::width>> headfc_in[HEADS];
	hls::stream<ap_uint<BUSWIDTH>> dummy_weights, input_skip;
	for(int i=0;i<NUM_TOKENS;i++)
	{
		headfc_in[0].write(output_comb[i]);
		headfc_in[1].write(output_comb[i+NUM_TOKENS]);
	}
	for(int i=0;i<isize;i++){
		input_skip.write(input_buffer[isize+i]);
	}

	HeadFC(dummy_weights, headfc_in, input_skip, output_headfc, false);


	// output
	skip_out_t *output = reinterpret_cast<skip_out_t*>(output_headfc);

	std::ofstream ofs((std::string) argv[1] + "/../../../debug/hls.txt");
	for(int i=0;i<NUM_TOKENS*EMB_DIM;i++){
		ofs << output[i] << '\n';
	}
	ofs.close();
	return 0;
}
