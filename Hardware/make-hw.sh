#!/bin/bash

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <network> <platform> <mode>" >&2
  echo "where <network> = vit" >&2
  echo "<platform> = ultra96, zcu104" >&2
  echo "<mode> = regenerate (h)ls only, (b)itstream only, (a)ll" >&2
  exit 1
fi

NETWORK=$1
PLATFORM=$2
MODE=$3
PATH_TO_VIVADO=$(which vivado)
PATH_TO_VITIS_HLS=$(which vitis_hls)

if [ -z "$ROOT" ]; then
    export ROOT="$( ( cd "$(dirname "$0")/.."; pwd) )"
fi

if [ -z "$PATH_TO_VIVADO" ]; then
    echo "Error: Vivado not found."
    exit 1
fi

if [ -z "$PATH_TO_VITIS_HLS" ]; then
    echo "Error: Vivado HLS not found."
    exit 1
fi

if [ ! -d "HLS/$NETWORK" ]; then
    echo "Error: Network is not available in HLS/."
    exit 1
fi

TARGET_NAME="$NETWORK-$PLATFORM"

# HLS Directories
HLS_OUT_DIR="$ROOT/Hardware/output/HLS/$TARGET_NAME"
HLS_SCRIPT=$ROOT/Hardware/HLS/hls-syn.tcl
REPORT_OUT_DIR="$ROOT/Hardware/output/Reports/"
HLS_SRC_DIR="$ROOT/Hardware/HLS/$NETWORK"

# reports/logs
HLS_IP_REPO="$HLS_OUT_DIR/ips/"
VITIS_HLS_LOG="$ROOT/Hardware/output/HLS/vitis_hls.log"
HLS_REPORT_PATH="$HLS_OUT_DIR/sol1/syn/report/AttnHead_csynth.rpt"

# regenerate HLS if requested
if [[ ("$MODE" == "h") || ("$MODE" == "a")  ]]; then
  mkdir -p $HLS_OUT_DIR
  mkdir -p $REPORT_OUT_DIR
  OLDDIR=$(pwd)
  echo "Calling Vivado HLS for hardware synthesis..."
  cd $HLS_OUT_DIR/..
	PARAMS="$ROOT/Training/export/$NETWORK"
	TEST_INPUT="$ROOT/Training/export/input.npy"
  HLS_OUTPUT_PATH="$ROOT/debug/hls_output_$NETWORK.txt"
  if [[ ("$PLATFORM" == "ultra96") ]]; then
    PLATFORM_PART="xczu3eg-sbva484-1-i"
    TARGET_CLOCK=10
  elif [[ ("$PLATFORM" == "zcu104") ]]; then
    PLATFORM_PART="xczu7ev-ffvc1156-2-e"
    TARGET_CLOCK=10
  else
	  echo "Error: Platform not supported for now. Please choose ultra96."
	  exit 1
  fi
  if [ ! -d "$PARAMS" ]; then
	echo "Error: Please export the weight and threshold parameters in $PARAMS"
	exit 1
  fi
  vitis_hls -f $HLS_SCRIPT -tclargs $NETWORK-$PLATFORM $HLS_SRC_DIR $PARAMS $TEST_INPUT $HLS_OUTPUT_PATH $PLATFORM_PART $TARGET_CLOCK
  if cat $VITIS_HLS_LOG | grep "ERROR"; then
    echo "Error in ViTIS_HLS"
    exit 1	
  fi
  if cat $VITIS_HLS_LOG | grep "CRITICAL WARNING"; then
    echo "Critical warning in ViTIS_HLS"
    exit 1	
  fi
  cat $HLS_REPORT_PATH | grep "Utilization Estimates" -A 20 > $REPORT_OUT_DIR/$TARGET_NAME-hls.txt
  cat $REPORT_OUT_DIR/$TARGET_NAME-hls.txt
  echo "HLS synthesis complete"
  echo "HLS-generated IP is at $HLS_IP_REPO"
  cd $HLS_IP_REPO
  unzip TransRDMA.zip -d TransRDMA/
  unzip AttnHead.zip -d AttnHead/
  unzip HeadFC.zip -d HeadFC/
  cd $OLDDIR
fi

# generate bitstream if requested

VIVADO_SCRIPT_DIR=$ROOT/Hardware/Vivado/$PLATFORM
VIVADO_SCRIPT=$VIVADO_SCRIPT_DIR/make-vivado-proj.tcl

VIVADO_OUT_DIR="$ROOT/Hardware/output/Vivado/$TARGET_NAME/"
BITSTREAM_PATH="$ROOT/Hardware/output/Bitstreams/"
BITSTREAM="$BITSTREAM_PATH/$TARGET_NAME.bit"
BITSTREAM_PATH_PYNQ="$ROOT/Hardware/Pynq/Bitstreams/"
HWH="$BITSTREAM_PATH/$TARGET_NAME.hwh"

if [[ ("$MODE" == "b") || ("$MODE" == "a")  ]]; then
  mkdir -p "$ROOT/Hardware/output/Vivado"
  mkdir -p $BITSTREAM_PATH
  echo "Setting up Vivado project..."
  if [ -d "$VIVADO_OUT_DIR" ]; then
  read -p "Remove existing project at $VIVADO_OUT_DIR (y/n)? " -n 1 -r
  echo    # (optional) move to a new line
    if [[ $REPLY =~ ^[Nn]$ ]]
    then
      echo "Cancelled"
      exit 1
    fi
  rm -rf $VIVADO_OUT_DIR
  fi
  vivado -mode batch -notrace -source $VIVADO_SCRIPT -tclargs $HLS_IP_REPO $TARGET_NAME $VIVADO_OUT_DIR $VIVADO_SCRIPT_DIR

  cp -f "$VIVADO_OUT_DIR/$TARGET_NAME.runs/impl_1/procsys_wrapper.bit" $BITSTREAM
  cp -f "$VIVADO_OUT_DIR/$TARGET_NAME.gen/sources_1/bd/procsys/hw_handoff/procsys.hwh" $HWH

  # extract parts of the post-implementation reports
  cat "$VIVADO_OUT_DIR/$TARGET_NAME.runs/impl_1/procsys_wrapper_timing_summary_routed.rpt" | grep "| Design Timing Summary" -B 3 -A 10 > $REPORT_OUT_DIR/$TARGET_NAME.txt
  cat "$VIVADO_OUT_DIR/$TARGET_NAME.runs/impl_1/procsys_wrapper_utilization_placed.rpt" | grep "| Slice LUTs" -B 3 -A 11 >> $REPORT_OUT_DIR/$TARGET_NAME.txt
  cat "$VIVADO_OUT_DIR/$TARGET_NAME.runs/impl_1/procsys_wrapper_utilization_placed.rpt" | grep "| CLB LUTs" -B 3 -A 11 >> $REPORT_OUT_DIR/$TARGET_NAME.txt
  cat "$VIVADO_OUT_DIR/$TARGET_NAME.runs/impl_1/procsys_wrapper_utilization_placed.rpt" |  grep "| Block RAM Tile" -B 3 -A 5 >> $REPORT_OUT_DIR/$TARGET_NAME.txt
  cat "$VIVADO_OUT_DIR/$TARGET_NAME.runs/impl_1/procsys_wrapper_utilization_placed.rpt" |  grep "| DSPs" -B 3 -A 3 >> $REPORT_OUT_DIR/$TARGET_NAME.txt  

  cp -f $BITSTREAM $BITSTREAM_PATH_PYNQ
  cp -f $HWH $BITSTREAM_PATH_PYNQ  
  cp -f $REPORT_OUT_DIR/$TARGET_NAME.txt $BITSTREAM_PATH
  cp -f $REPORT_OUT_DIR/$TARGET_NAME.txt $BITSTREAM_PATH_PYNQ
  cat $REPORT_OUT_DIR/$TARGET_NAME.txt

  echo "Bitstream copied to $BITSTREAM"

fi

echo "Done!"
exit 0
