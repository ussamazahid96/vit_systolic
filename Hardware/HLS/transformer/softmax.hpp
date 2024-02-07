#ifndef SOFTMAX_HPP_
#define SOFTMAX_HPP_

#include "hls_stream.h"
#include "hls_exp_apfixed.h"
#include "interpret.hpp"
#include "exp_lookup.hpp"


template<unsigned int TOKENS, unsigned int ARRAY_SIZE,
		 typename TSrcI = Identity, typename TDstI = Identity,
         typename TI, typename TO>
void SoftMaxLayerSA(hls::stream<TI> &attn_scores, hls::stream<TO> &softmax)
{

	static_assert(TI::width == ARRAY_SIZE*TSrcI::width,"");
	static_assert(TO::width == 2*ARRAY_SIZE*TDstI::width,"");

	const unsigned total_read_cycles = 2*TOKENS-ARRAY_SIZE;
	const unsigned ITERATIONS = total_read_cycles + ARRAY_SIZE-1;
	const unsigned BUFFER_CAPACITY = 2*ARRAY_SIZE-2;
	const unsigned NON_ZERO_ELEMS = 2*ARRAY_SIZE-1;
	TI buffer[BUFFER_CAPACITY];
#pragma HLS ARRAY_PARTITION dim=0 type=complete variable=buffer
	for(int s=0;s<BUFFER_CAPACITY;s++){
#pragma HLS UNROLL
		buffer[s] = 0;
	}
	typename TSrcI::type sum_buffer[NON_ZERO_ELEMS];
#pragma HLS ARRAY_PARTITION dim=0 type=complete variable=sum_buffer
	for(int s=0;s<NON_ZERO_ELEMS;s++){
#pragma HLS UNROLL
		sum_buffer[s] = 0.0;
	}

	ap_uint<8> buffer_store_idx = 0;
	ap_uint<8> sum_buffer_idx = 0;
	ap_uint<8> buffer_read_idx = 0;
	ap_uint<8> col_idx_counters[ARRAY_SIZE-1];
#pragma HLS ARRAY_PARTITION dim=0 type=complete variable=col_idx_counters
	ap_int<8> col_elem_counters[ARRAY_SIZE-1];
#pragma HLS ARRAY_PARTITION dim=0 type=complete variable=col_elem_counters
	ap_uint<1> on_odd = 1, shift = 0, write_output=0;

	for(int s=0;s<ARRAY_SIZE-1;s++)
	{
#pragma HLS UNROLL
		col_elem_counters[s] = -s-1;
		ap_uint<8> idx = ARRAY_SIZE + 2*s;
		if(idx >= BUFFER_CAPACITY)
			idx -= BUFFER_CAPACITY;
		col_idx_counters[s] = idx;
	}

	for(unsigned i=0;i<ITERATIONS;i++)
	{
#pragma HLS PIPELINE
		if(i<total_read_cycles)
		{
			TI elemIn = attn_scores.read();
			buffer[buffer_store_idx] = elemIn;
			buffer_store_idx++;
			if(buffer_store_idx == BUFFER_CAPACITY)
				buffer_store_idx = 0;
			if(sum_buffer_idx == 0)
				write_output = 1;
		}

		else
		{
			write_output = 1;
			// assign the columns
			for(int s=0;s<ARRAY_SIZE-1;s++)
			{
#pragma HLS UNROLL
				ap_int<8> elem_idx1 = col_elem_counters[s];
				elem_idx1++;
				col_elem_counters[s] = elem_idx1;
				if(elem_idx1 == ARRAY_SIZE-1)
					shift = 1;
			}

			// apply the column shift
			if(shift)
			{
				for(int s=0;s<ARRAY_SIZE-1;s++)
				{
#pragma HLS UNROLL
					col_elem_counters[s] = (s<ARRAY_SIZE-2) ? col_elem_counters[s+1] : (ap_int<8>)-ARRAY_SIZE;
					col_idx_counters[s]  = (s<ARRAY_SIZE-2) ? col_idx_counters[s+1]  : (ap_uint<8>) 0;
				}
				shift = 0;
			}


			// increment counters
			buffer_read_idx += 2;
			if(buffer_read_idx >= BUFFER_CAPACITY)
				buffer_read_idx -= BUFFER_CAPACITY;
		}

		if(i>ARRAY_SIZE-1 && i<total_read_cycles)
		{
			if(on_odd)
			{
				write_output = 1;
				// get the column shift
				for(int s=0;s<ARRAY_SIZE-1;s++)
				{
#pragma HLS UNROLL
					ap_int<8> elem_idx2 = col_elem_counters[s];
					elem_idx2++;
					col_elem_counters[s] = elem_idx2;
					if(elem_idx2 == ARRAY_SIZE-1)
						shift = 1;
				}
				// apply the column shift
				if(shift)
				{
					for(int s=0;s<ARRAY_SIZE-1;s++)
					{
#pragma HLS UNROLL
						col_elem_counters[s] = (s<ARRAY_SIZE-2) ? col_elem_counters[s+1] : (ap_int<8>) 0;
						ap_uint<8> new_idx = (col_idx_counters[ARRAY_SIZE-2] + 2);
						if(new_idx >= BUFFER_CAPACITY)
							new_idx -= BUFFER_CAPACITY;
						col_idx_counters[s]  = (s<ARRAY_SIZE-2) ? col_idx_counters[s+1] : new_idx;
					}
					shift = 0;
				}
				// increment counters
				if(sum_buffer_idx > ARRAY_SIZE-1){
					buffer_read_idx += 2;
					if(buffer_read_idx >= BUFFER_CAPACITY)
						buffer_read_idx -= BUFFER_CAPACITY;
				}
				else
					buffer_read_idx++;
			}
			on_odd++;
		}

		// assigning the row and perform exp
		TI row = buffer[buffer_read_idx];
		auto const el = TSrcI()(row,0);
		for(int j=0;j<ARRAY_SIZE;j++){
#pragma HLS UNROLL
			typename TSrcI::type exp_elem = exp_reduce::exp(el(j,0));
			sum_buffer[j] = exp_elem;
		}

		// assigning the col and perform exp
		for(int s=0;s<ARRAY_SIZE-1;s++)
		{
#pragma HLS UNROLL
			ap_int<8> elem_idx3 = col_elem_counters[s];
			ap_uint<8> col_idx = col_idx_counters[s];
			TI col = buffer[col_idx];
			auto const col_fxd = TSrcI()(col, 0);
			sum_buffer[ARRAY_SIZE+s] = (elem_idx3 >=0) ? exp_reduce::exp(col_fxd(elem_idx3,0)) : (typename TSrcI::type) 0.0;
		}

		// calc sum
		ap_fixed<24,16, AP_RND, AP_SAT> sum = 0;
		for(int s=0;s<NON_ZERO_ELEMS;s++){
	#pragma HLS UNROLL
			sum += sum_buffer[s];
		}

		// normalize by sum
		for(int j=0;j<NON_ZERO_ELEMS;j++){
#pragma HLS UNROLL
			sum_buffer[j] /= sum;
		}

		if(write_output)
		{	
			// write on the output stream
			auto  outElem = TDstI().template operator()<TO>();
			for(int s=0;s<NON_ZERO_ELEMS;s++){
#pragma HLS UNROLL
				typename TDstI::type elemOut = sum_buffer[s];
				ap_uint<TDstI::width> elemOut_packed = *reinterpret_cast<ap_uint<TDstI::width>*>(&elemOut);
				outElem(s,0,1) = elemOut_packed;
			}
			outElem(2*ARRAY_SIZE-1,0,1) = 0;
			softmax.write(outElem);
			write_output=0;
			sum_buffer_idx++;
		}
	}
}

template<unsigned int TOKENS, unsigned int ARRAY_SIZE,
		 typename TSrcI = Identity, typename TDstI = Identity,
         typename TI, typename TO>
void SoftMaxLayerSA_Batch(hls::stream<TI> &attn_scores, hls::stream<TO> &softmax, unsigned int numReps)
{
	for(unsigned int i=0;i<numReps;i++){
		SoftMaxLayerSA<TOKENS, ARRAY_SIZE, TSrcI, TDstI>(attn_scores, softmax);
	}
}

template<unsigned int MatrixW, unsigned int MatrixH, unsigned int numPEs,
		 typename TSrcI = Identity, typename TAccI = Identity,typename TDstI = Identity,
		 typename TI, typename TO>
void SoftMaxLayer(hls::stream<TI> &input_stream, hls::stream<TO> &output_stream)
{
	static_assert(TI::width == numPEs*TSrcI::width, "");

	typename TAccI::type Buffer[MatrixW];

	const unsigned int FOLD = TSrcI::width*MatrixW / TI::width;
	const unsigned int ITERATIONS = 2*FOLD*MatrixH;

	unsigned int read_fold = 0, write_fold=0;
	ap_ufixed<24,16, AP_RND, AP_SAT> sum=0;

	for(unsigned int iter = 0; iter < ITERATIONS; iter++)
	{
#pragma HLS PIPELINE II=1
		if(read_fold < MatrixW)
		{		
			TI elem = input_stream.read();
			auto const attn_score = TSrcI()(elem, 0);
			typename TAccI::type sum_tree = 0;
			for(unsigned int i=0;i<numPEs;i++)
			{
#pragma HLS UNROLL
				auto const elem = attn_score(i,0);
				typename TAccI::type exp_elem=0;
				ExpLookup(elem, exp_elem);
				unsigned int addr = read_fold+i;
				Buffer[addr] = exp_elem;
				sum_tree += exp_elem;
			}
			sum += sum_tree;
			read_fold += numPEs;
		}
		else
		{
 			auto const outElem = TDstI().template operator()<TO>();
 			for(unsigned int i=0;i<numPEs;i++)
 			{
#pragma HLS UNROLL
 				unsigned int addr = write_fold+i;
 				typename TAccI::type elem = Buffer[addr];
 				typename TDstI::type elem2 = elem/sum;
 				outElem(i,0,1) = elem2 * (1 << (TDstI::type::width-TDstI::type::iwidth));
 			}
 			output_stream.write(outElem);
			write_fold += numPEs;
 			if(write_fold == MatrixW)
 			{
 				write_fold = 0;
 				read_fold = 0;
 				sum = 0;
 			}
		}
	}
}





#endif
