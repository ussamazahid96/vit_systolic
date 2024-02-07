#ifndef __MEM_INIT__HPP__

#define __MEM_INIT__HPP__


#include "hls_stream.h"
#include "interpret.hpp"

template<unsigned int DIM, unsigned int DATAWIDTH, typename TW>
void StreamingNormMemInit(hls::stream<TW> &normwStream,
                          TW nw_buffer[DIM * DATAWIDTH/ TW::width], TW nb_buffer[DIM * DATAWIDTH/ TW::width])
{
    const unsigned int FOLD = DIM * DATAWIDTH/ TW::width;
    unsigned int w_idx = 0, b_idx=0;
    
    for(unsigned i=0;i<2*FOLD;i++)
    {
#pragma HLS PIPELINE II=1
        TW wElem = normwStream.read();
        if(i<FOLD)
        {
            nw_buffer[w_idx] = wElem;
            w_idx++;
        }
        else
        {
            nb_buffer[b_idx] = wElem;
            b_idx++;
        }
    }
}


template<unsigned int numPEs, unsigned int FOLDs, typename TWS, typename TW>
void StreamingAttnMemInit(hls::stream<TWS> &input_stream,  TW &query_weights)
{
    for(unsigned p=0;p<numPEs;p++)
    {
    	for(unsigned f=0;f<FOLDs;f++)
    	{
#pragma HLS PIPELINE II=1
    		TWS wElem = input_stream.read();
    		query_weights.m_weights[p][f] = wElem;
    	}
    }
}


template<unsigned int MatrixW, unsigned int MatrixH, unsigned numPEs, unsigned int numSIMDs, unsigned int TILES,
         typename TSrcI, typename TDstI, typename TI, typename TO>
void StreamingAttnBufInit(hls::stream<TI> &input, hls::stream<TO> &output_tiled)
{
    static_assert(TI::width == numSIMDs * TSrcI::width, "");
    const unsigned int total_read_cycles  = MatrixH*MatrixW*TSrcI::width/TI::width;
    const unsigned int total_write_cycles = TILES;
    const unsigned int ITERATIONS = total_read_cycles + total_write_cycles*MatrixH;
    const unsigned int SFold = MatrixW / numSIMDs;
    unsigned int pe = 0, folds = 0, current_fold = 0, tile = 0;

    ap_uint<numSIMDs*TSrcI::width> Buffer[numPEs][TILES];
#pragma HLS ARRAY_PARTITION dim=1 type=complete variable=Buffer

    for(unsigned iter = 0; iter < ITERATIONS; iter++){
#pragma HLS PIPELINE II=1
    	if(iter < total_read_cycles){
			TI elem = input.read();
            unsigned int wmem_addr = folds+current_fold;
			Buffer[pe][wmem_addr] = elem;
			if(++current_fold == SFold){
				current_fold = 0;
				if(++pe == numPEs){
					pe = 0;
					folds+=SFold;
				}
			}
    	}else{
    		auto outElem = TDstI().template operator()<TO>();
    	    for(unsigned p = 0; p < numPEs; p++){
#pragma HLS UNROLL
    	    	auto const wgt = Buffer[p][tile];
    	    	outElem(p,0,1) = wgt;
    	    }
    	    output_tiled.write(outElem);
    	    if(++tile == TILES){
    	    	tile=0;
    	    }
    	}
    }
}


template<unsigned int MatrixW, unsigned int MatrixH, unsigned numPEs, unsigned int numSIMDs, unsigned int TILES,
         typename TSrcI, typename TDstI, typename TI, typename TO>
void StreamingTransBufInit(hls::stream<TI> &input, hls::stream<TO> &output_tiled)
{
    static_assert(TI::width == numPEs * TSrcI::width, "");
    const unsigned int total_read_cycles  = MatrixH*MatrixW*TSrcI::width/TI::width;
    const unsigned int total_write_cycles = TILES;
    const unsigned int ITERATIONS = total_read_cycles + total_write_cycles*MatrixH;
    const unsigned int SFold = MatrixW / numPEs;
    unsigned int simd = 0, folds = 0, current_fold = 0, tile_offset=0, col = 0;

    FixedPointWeights<numPEs, typename TSrcI::type, numSIMDs, TILES> Buffer;
#pragma HLS ARRAY_PARTITION dim=1 type=complete variable=Buffer.m_weights

    for(unsigned iter = 0; iter < ITERATIONS; iter++){
#pragma HLS PIPELINE II=1
        if(iter < total_read_cycles){
            TI elem = input.read();
            unsigned int wmem_addr = folds+current_fold;
            Buffer.m_weights[simd][wmem_addr] = elem;
            if(++current_fold == SFold){
                current_fold = 0;
                if(++simd == numSIMDs){
                    simd = 0;
                    folds+=SFold;
                }
            }
        }else{
        	unsigned int tile = tile_offset + col;
        	auto const &w = Buffer.weights(tile);
			auto outElem = TDstI().template operator()<TO>();
        	for(unsigned p = 0; p < numPEs;p++){
#pragma HLS UNROLL
        		auto const simd_pack = TSrcI().template operator()<ap_uint<numSIMDs*TSrcI::width>>();
				for(unsigned s = 0; s < numSIMDs; s++){
#pragma HLS UNROLL
					auto const wgt = w[s];
					simd_pack(s,0,1) = wgt[p];
				}
				outElem(p,0,1) = simd_pack[0];
        	}
			output_tiled.write(outElem);
			tile_offset += SFold;
			if(tile_offset == TILES){
				tile_offset = 0;
				if(++col == SFold){
					col = 0;
				}
			}
        }
    }
}


template<unsigned int DATAWIDTH, unsigned int PACKEDDATAWIDTH, unsigned int NUM_HEADS, unsigned int NUM_BYTES>
void SplitStream(ap_uint<DATAWIDTH> *input, hls::stream<ap_uint<PACKEDDATAWIDTH>> output_streams[HEADS])
{
    const unsigned int numWords = NUM_BYTES / (DATAWIDTH / 8);
    for (unsigned int i = 0; i < numWords; i++) {
#pragma HLS pipeline style=flp II=1
        ap_uint<DATAWIDTH> el = input[i];
        for(unsigned int h = 0; h<HEADS;h++){
#pragma HLS UNROLL
        	ap_uint<PACKEDDATAWIDTH> oElem = el((h+1)*PACKEDDATAWIDTH-1,h*PACKEDDATAWIDTH);
			output_streams[h].write(oElem);
        }
    }
}

template<unsigned int DATAWIDTH, unsigned int PACKEDDATAWIDTH, unsigned int NUM_HEADS, unsigned int AttnBytes, unsigned int HFCBytes>
void SplitStream_hfc(ap_uint<DATAWIDTH> *input,
					 hls::stream<ap_uint<PACKEDDATAWIDTH>> output_streams[HEADS],
					 hls::stream<ap_uint<DATAWIDTH>> &hfc_weights_stream)
{
    const unsigned int AttnnumWords = AttnBytes / (DATAWIDTH / 8);
    const unsigned int HFCnumWords  = HFCBytes / (DATAWIDTH / 8);
    for (unsigned int i = 0; i < (AttnnumWords+HFCnumWords); i++) {
#pragma HLS pipeline style=flp II=1
        ap_uint<DATAWIDTH> el = input[i];
		if(i<AttnnumWords){
			for(unsigned int h = 0; h<HEADS;h++){
	#pragma HLS UNROLL
				ap_uint<PACKEDDATAWIDTH> oElem = el((h+1)*PACKEDDATAWIDTH-1,h*PACKEDDATAWIDTH);
				output_streams[h].write(oElem);
			}
		}
		else{
			hfc_weights_stream.write(el);
		}
    }
}
void WeightsSplit(ap_uint<BUSWIDTH> *input_and_weights,
				  ap_uint<BUSWIDTH> *query_weights,
				  ap_uint<BUSWIDTH> *key_weights,
			      ap_uint<BUSWIDTH> *value_weights,

				  hls::stream<ap_uint<BUSWIDTH>>    input_and_weights_stream[HEADS],
				  hls::stream<ap_uint<PACKEDWIDTH>> query_weights_stream[HEADS],
				  hls::stream<ap_uint<PACKEDWIDTH>> key_weights_stream[HEADS],
				  hls::stream<ap_uint<PACKEDWIDTH>> value_weights_stream[HEADS],
				  hls::stream<ap_uint<BUSWIDTH>> &hfc_weights_stream)
{

	const unsigned int normwBits  = EMB_DIM*norm_w_t::width;
	const unsigned int normbBits  = EMB_DIM*norm_b_t::width;
	const unsigned int attnwBits  = HEADS*PEs*WMEM*PACKEDWIDTH;
	const unsigned int hfcwBits	  = HFCPEs*HFCWMEM*PACKEDWIDTH;

	//SplitStream<BUSWIDTH, (normwBits+normbBits)/8>(input_and_weights, input_and_weights_stream);
	SplitStream_hfc<BUSWIDTH, PACKEDWIDTH, HEADS, attnwBits/8, hfcwBits/8>(query_weights, query_weights_stream, hfc_weights_stream);
	SplitStream<BUSWIDTH, PACKEDWIDTH, HEADS, attnwBits/8>(key_weights, key_weights_stream);
	SplitStream<BUSWIDTH, PACKEDWIDTH, HEADS, attnwBits/8>(value_weights, value_weights_stream);
}

template<unsigned int BusWidth, unsigned int Heads, unsigned int inBits>
void InputSplit(ap_uint<BusWidth> *input_and_weights, hls::stream<ap_uint<BusWidth>> input_and_weights_stream[HEADS],
				hls::stream<ap_uint<BusWidth>> &input_skip_stream)
{
	const unsigned int words = inBits/BusWidth;
	for(unsigned int i = 0; i< 2*words;i++){
#pragma HLS pipeline style=flp II=1
		ap_uint<BusWidth> elem = input_and_weights[i];
		if(i<words)
		{
			for(unsigned int j=0;j<Heads;j++){
	#pragma HLS UNROLL
				input_and_weights_stream[j].write(elem);
			}
		}
		else
		{
			input_skip_stream.write(elem);
		}
	}

}


template<unsigned int Heads, unsigned int DIM, unsigned int Tokens, typename TI, typename TO>
void CombineStream(hls::stream<TI> in_stream[HEADS], hls::stream<TO> & out_stream)
{
	static_assert(TO::width == Heads*TI::width, "");
	for(unsigned int i=0;i<Tokens;i++)
	{
#pragma HLS pipeline II=1
		TO outElem = 0;
		for(unsigned int j=0;j<Heads;j++){
#pragma HLS UNROLL
			TI elem = in_stream[j].read();
			outElem((j+1)*TI::width-1, j*TI::width) = elem;
		}
		out_stream.write(outElem);
	}
}


template<unsigned int BusWidth, unsigned int outputBits,
		 typename TSrcI1 = Identity, typename TSrcI2 = Identity, typename TDstI = Identity,
		 typename TS>
void AddStreams(hls::stream<TS> &input1, hls::stream<TS> &input2, hls::stream<TS> &output)
{
	const unsigned int ITERATIONS = outputBits/BusWidth;
	const unsigned int ELEMPERITER = BusWidth/TSrcI1::width;
	for(unsigned int i=0;i<ITERATIONS;i++)
	{
#pragma HLS pipeline II=1
		TS elem1 = input1.read();
		TS elem2 = input2.read();
		auto const el1 = TSrcI1()(elem1, 0);
		auto const el2 = TSrcI2()(elem2, 0);
		auto outElem = TDstI().template operator()<TS>();
		for(unsigned int j =0; j<ELEMPERITER;j++)
		{
#pragma HLS UNROLL
			typename TDstI::type sum = el1(j,0) + el2(j,0);
			outElem(j,0,1) = sum*(1<<(TDstI::type::width-TDstI::type::iwidth));
		}
		output.write(outElem);
	}
}





#endif
