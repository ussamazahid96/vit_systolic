/******************************************************************************
 *  Copyright (c) 2019, Xilinx, Inc.
 *  Copyright (c) 2022, Advanced Micro Devices, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *  1.  Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2.  Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *
 *  3.  Neither the name of the copyright holder nor the names of its
 *      contributors may be used to endorse or promote products derived from
 *      this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 *  THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 *  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 *  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 *  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 *  OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 *  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 *  OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 *  ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/
/******************************************************************************
 *
 *  Authors: Giulio Gambardella <giuliog@xilinx.com>
 *
 *  \file input_gen_kernelstride.cpp
 *
 *  HLS Top function with a single HLS sliding-window generator block (when kernel%stride !=0) unit testing
 *
 *****************************************************************************/
#include <hls_stream.h>
using namespace hls;
#include "ap_int.h"
#include "bnn-library.h"
#include "data/input_gen_kernelstride.h"

void Testbench(stream<ap_uint<IFM_Channels*INPUT_PRECISION> > & in, stream<ap_uint<IFM_Channels*INPUT_PRECISION> > & out, unsigned int numReps)
{
#pragma HLS DATAFLOW
stream<ap_uint<SIMD*INPUT_PRECISION> > in_simd("in_simd");
stream<ap_uint<SIMD*INPUT_PRECISION> > out_simd("out_simd");

StreamingDataWidthConverter_Batch<IFM_Channels*INPUT_PRECISION, SIMD*INPUT_PRECISION, IFMDim*IFMDim>(in, in_simd, numReps);

ConvolutionInputGenerator_kernel_stride<KERNEL_DIM,
	IFM_Channels,
	INPUT_PRECISION,
	IFMDim, 
	OFMDim, 
	SIMD,
	STRIDE>(in_simd, out_simd, numReps, ap_resource_dflt());
	
StreamingDataWidthConverter_Batch<SIMD*INPUT_PRECISION, IFM_Channels*INPUT_PRECISION, KERNEL_DIM*KERNEL_DIM*OFMDim*OFMDim*IFM_Channels/SIMD>(out_simd, out, numReps);

}

