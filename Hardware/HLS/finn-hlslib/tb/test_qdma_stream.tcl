open_project hls-syn-qdma
add_files qdma_stream_top.cpp -cflags "-std=c++14 -I$::env(FINN_HLS_ROOT) -I$::env(FINN_HLS_ROOT)/tb" 
add_files -tb qdma_stream_tb.cpp -cflags "-std=c++14 -I$::env(FINN_HLS_ROOT) -I$::env(FINN_HLS_ROOT)/tb" 
set_top qdma_conv_top
open_solution sol1
set_part {xczu3eg-sbva484-1-i}
create_clock -period 5 -name default
csim_design
csynth_design
cosim_design
exit
