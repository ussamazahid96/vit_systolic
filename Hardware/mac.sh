CURR_PATH=$(pwd)

g++ HLS/vitW1A2/tb_cosim.cpp -o csim.exe \
    -I$CURR_PATH/../../include/ \
    -I$CURR_PATH/HLS/tpu/ \
    -I$CURR_PATH/../Training/export/vitW1A2/hw/ \
    -I$CURR_PATH/HLS/finn-hlslib/ \
    -I$CURR_PATH/HLS/vitW1A2/top.cpp \
    -std=c++17
./csim.exe