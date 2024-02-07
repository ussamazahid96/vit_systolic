#ifndef __LAYERNORM__HPP__
#define __LAYERNORM__HPP__

#include "hls_stream.h"
#include "interpret.hpp"
#include "hls_math.h"

template<unsigned int DIM, unsigned int TOKENS, 
         typename TSrcI = Identity, typename TWeiI = Identity, typename TBiasI = Identity, typename TDstI = Identity,
         typename TI, typename TW, typename TB, typename TO>
void LayerNorm_Batch(hls::stream<TI> &input, hls::stream<TO> &output,
					 const TW normW[DIM * TSrcI::width/ TI::width],
					 const TB normB[DIM * TSrcI::width/ TI::width],
					 unsigned int numReps)
{
    const unsigned int FOLD = DIM * TSrcI::width/ TI::width;
    const unsigned int ELEMSPERIN = TI::width / TSrcI::width;
    const ap_ufixed<16,0> scale = (double) 1/DIM;
    const ap_ufixed<18,0> scale2 = (double) 1/(DIM-1);
    TI inputBuffer[FOLD];

    for(unsigned i = 0; i < TOKENS*numReps; i++)
    {

        ap_fixed<16,0>  running_mean = 0, prev_mean = 0;
        ap_ufixed<18,2> running_var = 1e-5;
        ap_ufixed<18,2> running_std = 0, running_invstd = 0;
        unsigned int read_idx = 0;

    	for(unsigned f = 0; f < 2*FOLD; f++)
        {
            if(f<FOLD)
            {
                TI  iElem = input.read();
                inputBuffer[f] = iElem;
                auto const in = TSrcI()(iElem,0);
                for(int k=0;k<ELEMSPERIN;k++){
#pragma HLS UNROLL
                    typename TSrcI::type inElem = in(k,0);
                    prev_mean = running_mean;
                    running_mean += (inElem * scale);
                    running_var  += (inElem-running_mean)*(inElem-prev_mean)*scale2;
                }
                running_std    = hls::sqrt(running_var);
                running_invstd = hls::recip(running_std);
            }
            else
            {
            	TI iElem = inputBuffer[read_idx];
            	TW wElem = normW[read_idx];
            	TB bElem = normB[read_idx];
            	read_idx++;
                auto const in = TSrcI()(iElem,0);
                auto const w  = TWeiI()(wElem,0);
                auto const b  = TBiasI()(bElem,0);
                auto  outElem = TDstI().template operator()<TO>();
                for(int k=0;k<ELEMSPERIN;k++){
#pragma HLS UNROLL
                    ap_fixed<24,8> in_normalized = (in(k,0)-running_mean)*running_invstd;
                    ap_fixed<24,8> in_scaled = in_normalized*w(k,0) + b(k,0);
                    typename TDstI::type in_scaled_casted = (typename TDstI::type) in_scaled;
                    outElem(k,0,1) = in_scaled_casted*(1<<(TDstI::type::width-TDstI::type::iwidth));
                }
                output.write(outElem);
            }
        }
    }
}


#endif
