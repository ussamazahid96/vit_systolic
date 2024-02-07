#ifndef SYSTOLIC_ARRAY_H
#define SYSTOLIC_ARRAY_H

#include "hls_stream.h"
#include "interpret.hpp"


template<unsigned int MatrixW, unsigned int MatrixH, unsigned int ARRAY_SIZE, 
		 typename TSrcI = Identity, typename TDstI = Identity,
  		 typename TI, typename TO, typename R>
void SystolicArray_Batch(hls::stream<TI> &queries,
						 hls::stream<TI> &keys,
						 hls::stream<TO> &attn_scores,
						 int const numReps,
						 R const &r)
{
	 static_assert(TI::width == MatrixW*TSrcI::width, "");
	 TI localA[ARRAY_SIZE];
	 TI localB[ARRAY_SIZE];
	 typename TDstI::type localC[ARRAY_SIZE][ARRAY_SIZE];
#pragma HLS ARRAY_PARTITION variable=localA dim=1 complete
#pragma HLS ARRAY_PARTITION variable=localB dim=1 complete
#pragma HLS ARRAY_PARTITION variable=localC dim=0 complete

buffer_reset_loop:
	for(unsigned i=0;i<ARRAY_SIZE;i++){
#pragma HLS UNROLL
			localA[i] = 0;
			localB[i] = 0;
	}
	for(unsigned i=0;i<ARRAY_SIZE;i++){
#pragma HLS UNROLL
		for(unsigned j=0;j<ARRAY_SIZE;j++){
#pragma HLS UNROLL
			localC[i][j] = 0;
		}
	}


	unsigned int output_idx=0;

	// Execution Starts from here
	for(unsigned token=0; token<MatrixH; token++)
	{
		// shift the tokens in the buffer
		for(unsigned buffer_col=0;buffer_col<ARRAY_SIZE-1;buffer_col++)
		{
#pragma HLS UNROLL
			localA[buffer_col] = localA[buffer_col+1];
			localB[buffer_col] = localB[buffer_col+1];
		}
		// push the new input
		TI elemA = queries.read();
		TI elemB = keys.read();
		localA[ARRAY_SIZE-1] = elemA;
		localB[ARRAY_SIZE-1] = elemB;

		if(token >= ARRAY_SIZE-1)
		{
			// systolic MM
			systolic1:
			for (unsigned k = 0; k < MatrixW; k++)
			{
#pragma HLS LOOP_TRIPCOUNT min=MatrixW max=MatrixW
#pragma HLS PIPELINE II=1
				systolic2:
				for (unsigned i = 0; i < ARRAY_SIZE; i++)
				{
					systolic3:
					for (unsigned j = 0; j < ARRAY_SIZE; j++)
					{
						typename TDstI::type last  = (k == 0) ? (typename TDstI::type) 0 : localC[i][j];
						TI a_val = localA[i];
						TI b_val = localB[j];
						auto const a_val_cast = TSrcI()(a_val, 0);
						auto const b_val_cast = TSrcI()(b_val, 0);
						typename TDstI::type pp = mul(a_val_cast(k,0), b_val_cast(k,0), r);
						typename TDstI::type result = last + pp;
						localC[i][j] = result;
					}
				}
			} // end systolic loop

			// writing the result buffer to the stream
			if(output_idx == 0)
			{
				for(unsigned loc=0;loc<ARRAY_SIZE;loc++)
				{
#pragma HLS PIPELINE II=1
					auto  outElem = TDstI().template operator()<TO>();
					for(int j=0;j<ARRAY_SIZE;j++)
					{
#pragma HLS UNROLL
						typename TDstI::type  elem = localC[loc][j];
						ap_uint<TDstI::width> elem_packed = *reinterpret_cast<ap_uint<TDstI::width>*>(&elem);
						outElem(j,0,1) = elem_packed;
					}
					attn_scores.write(outElem);
				}
				output_idx+=ARRAY_SIZE;
			}
			else
			{
				auto  outElem1 = TDstI().template operator()<TO>();
				auto  outElem2 = TDstI().template operator()<TO>();
				for(int j=0;j<ARRAY_SIZE;j++){
#pragma HLS UNROLL
					typename TDstI::type elem1 = localC[j][ARRAY_SIZE-1];
					typename TDstI::type elem2 = localC[ARRAY_SIZE-1][j];
					ap_uint<TDstI::width> elem1_packed = *reinterpret_cast<ap_uint<TDstI::width>*>(&elem1);
					ap_uint<TDstI::width> elem2_packed = *reinterpret_cast<ap_uint<TDstI::width>*>(&elem2);
					outElem1(j,0,1) = elem1_packed; // last column
					outElem2(j,0,1) = elem2_packed; //last row
				}
				attn_scores.write(outElem1);
				attn_scores.write(outElem2);
			}
		}

	}
	output_idx = 0;
}

#endif
