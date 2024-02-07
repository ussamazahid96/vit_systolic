###############################################################################
 #  Copyright (c) 2016, Xilinx, Inc.
 #  All rights reserved.
 #
 #  Redistribution and use in source and binary forms, with or without
 #  modification, are permitted provided that the following conditions are met:
 #
 #  1.  Redistributions of source code must retain the above copyright notice,
 #     this list of conditions and the following disclaimer.
 #
 #  2.  Redistributions in binary form must reproduce the above copyright
 #      notice, this list of conditions and the following disclaimer in the
 #      documentation and/or other materials provided with the distribution.
 #
 #  3.  Neither the name of the copyright holder nor the names of its
 #      contributors may be used to endorse or promote products derived from
 #      this software without specific prior written permission.
 #
 #  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 #  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 #  THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 #  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 #  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 #  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 #  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 #  OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 #  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 #  OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 #  ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 #
###############################################################################
###############################################################################
 #
 #
 # @file make-vivado-proj.tcl
 #
 # tcl script for block design and bitstream generation. Automatically 
 # launched by make-hw.sh. Tested with Vivado 2018.2
 #
 #
###############################################################################

# Creates a Vivado project ready for synthesis and launches bitstream generation
if {$argc != 4} {
  puts "Expected: <jam repo> <proj name> <proj dir> <xdc_dir>"
  exit
}

# paths to donut and jam IP folders
set config_jam_repo [lindex $argv 0]

# project name, target dir and FPGA part to use
set config_proj_name [lindex $argv 1]
set config_proj_dir [lindex $argv 2]
set config_proj_part "xczu3eg-sbva484-1-i"

# other project config

set xdc_dir [lindex $argv 3]

# set up project
create_project $config_proj_name $config_proj_dir -part $config_proj_part
set_property ip_repo_paths [list $config_jam_repo] [current_project]
update_ip_catalog

# create block design
create_bd_design "procsys"

source "${xdc_dir}/ultra96.tcl"

# instantiate ips
create_bd_cell -type ip -vlnv xilinx.com:hls:TransRDMA:1.0 TransRDMA_0
create_bd_cell -type ip -vlnv xilinx.com:hls:AttnHead:1.0 AttnHead_0
create_bd_cell -type ip -vlnv xilinx.com:hls:TransWDMA:1.0 TransWDMA_0

# set ports
set_property -dict [list CONFIG.PSU__USE__S_AXI_GP1 {1} CONFIG.PSU__USE__S_AXI_GP2 {1} CONFIG.PSU__USE__S_AXI_GP3 {1} CONFIG.PSU__USE__S_AXI_GP4 {1}] [get_bd_cells zynq_ultra_ps_e_0]

# apply connections
# AXI Slave
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} Master {/zynq_ultra_ps_e_0/M_AXI_HPM0_FPD} Slave {/AttnHead_0/s_axi_control} ddr_seg {Auto} intc_ip {New AXI Interconnect} master_apm {0}}  [get_bd_intf_pins AttnHead_0/s_axi_control]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} Master {/zynq_ultra_ps_e_0/M_AXI_HPM0_FPD} Slave {/TransRDMA_0/s_axi_control} ddr_seg {Auto} intc_ip {New AXI Interconnect} master_apm {0}}  [get_bd_intf_pins TransRDMA_0/s_axi_control]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} Master {/zynq_ultra_ps_e_0/M_AXI_HPM0_FPD} Slave {/TransWDMA_0/s_axi_control} ddr_seg {Auto} intc_ip {New AXI Interconnect} master_apm {0}}  [get_bd_intf_pins TransWDMA_0/s_axi_control]

# AXI Master
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {/zynq_ultra_ps_e_0/pl_clk0 (100 MHz)} Clk_slave {Auto} Clk_xbar {Auto} Master {/TransRDMA_0/m_axi_input_and_weights_mem} Slave {/zynq_ultra_ps_e_0/S_AXI_HP0_FPD} ddr_seg {Auto} intc_ip {New AXI SmartConnect} master_apm {0}}  [get_bd_intf_pins zynq_ultra_ps_e_0/S_AXI_HP0_FPD]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {/zynq_ultra_ps_e_0/pl_clk0 (100 MHz)} Clk_slave {Auto} Clk_xbar {Auto} Master {/TransRDMA_0/m_axi_query_weights_mem} Slave {/zynq_ultra_ps_e_0/S_AXI_HP1_FPD} ddr_seg {Auto} intc_ip {New AXI SmartConnect} master_apm {0}}  [get_bd_intf_pins zynq_ultra_ps_e_0/S_AXI_HP1_FPD]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {/zynq_ultra_ps_e_0/pl_clk0 (100 MHz)} Clk_slave {Auto} Clk_xbar {Auto} Master {/TransRDMA_0/m_axi_key_weights_mem} Slave {/zynq_ultra_ps_e_0/S_AXI_HP2_FPD} ddr_seg {Auto} intc_ip {New AXI SmartConnect} master_apm {0}}  [get_bd_intf_pins zynq_ultra_ps_e_0/S_AXI_HP2_FPD]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {/zynq_ultra_ps_e_0/pl_clk0 (100 MHz)} Clk_slave {Auto} Clk_xbar {Auto} Master {/TransRDMA_0/m_axi_value_weights_mem} Slave {/zynq_ultra_ps_e_0/S_AXI_HPC0_FPD} ddr_seg {Auto} intc_ip {New AXI SmartConnect} master_apm {0}}  [get_bd_intf_pins zynq_ultra_ps_e_0/S_AXI_HPC0_FPD]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {/zynq_ultra_ps_e_0/pl_clk0 (100 MHz)} Clk_slave {Auto} Clk_xbar {Auto} Master {/TransWDMA_0/m_axi_output_mem} Slave {/zynq_ultra_ps_e_0/S_AXI_HPC1_FPD} ddr_seg {Auto} intc_ip {New AXI SmartConnect} master_apm {0}}  [get_bd_intf_pins zynq_ultra_ps_e_0/S_AXI_HPC1_FPD]

# AXI Stream
connect_bd_intf_net [get_bd_intf_pins TransRDMA_0/input_and_weights_stream] [get_bd_intf_pins AttnHead_0/input_and_weights_stream]
connect_bd_intf_net [get_bd_intf_pins TransRDMA_0/query_weights_stream] [get_bd_intf_pins AttnHead_0/query_weights_stream]
connect_bd_intf_net [get_bd_intf_pins TransRDMA_0/key_weights_stream] [get_bd_intf_pins AttnHead_0/key_weights_stream]
connect_bd_intf_net [get_bd_intf_pins TransRDMA_0/value_weights_stream] [get_bd_intf_pins AttnHead_0/value_weights_stream]
connect_bd_intf_net [get_bd_intf_pins AttnHead_0/output_stream] [get_bd_intf_pins TransWDMA_0/output_stream]

add_files -fileset constrs_1 -norecurse "${xdc_dir}/ultra96.xdc"

# create HDL wrapper for the block design
save_bd_design
validate_bd_design
make_wrapper -files [get_files $config_proj_dir/$config_proj_name.srcs/sources_1/bd/procsys/procsys.bd] -top
add_files -norecurse $config_proj_dir/$config_proj_name.srcs/sources_1/bd/procsys/hdl/procsys_wrapper.v
update_compile_order -fileset sources_1
update_compile_order -fileset sim_1


set_property strategy Flow_PerfOptimized_high [get_runs synth_1]
set_property STEPS.SYNTH_DESIGN.ARGS.DIRECTIVE AlternateRoutability [get_runs synth_1]
set_property STEPS.SYNTH_DESIGN.ARGS.RETIMING true [get_runs synth_1]

set_property strategy Performance_ExtraTimingOpt [get_runs impl_1]
set_property STEPS.OPT_DESIGN.ARGS.DIRECTIVE Explore [get_runs impl_1]
set_property STEPS.POST_ROUTE_PHYS_OPT_DESIGN.ARGS.DIRECTIVE AggressiveExplore [get_runs impl_1]
set_property STEPS.PHYS_OPT_DESIGN.ARGS.DIRECTIVE AggressiveExplore [get_runs impl_1]
set_property STEPS.POST_ROUTE_PHYS_OPT_DESIGN.IS_ENABLED true [get_runs impl_1]
set_property STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE AlternateCLBRouting [get_runs impl_1]
set_property STEPS.OPT_DESIGN.ARGS.DIRECTIVE ExploreWithRemap [get_runs impl_1]

# set_property strategy Flow_RuntimeOptimized [get_runs synth_1]
# set_property STEPS.SYNTH_DESIGN.ARGS.DIRECTIVE AlternateRoutability [get_runs synth_1]
# set_property STEPS.SYNTH_DESIGN.ARGS.RETIMING true [get_runs synth_1]

# set_property strategy Flow_Quick [get_runs impl_1]
# set_property STEPS.OPT_DESIGN.ARGS.DIRECTIVE ExploreWithRemap [get_runs impl_1]
# set_property STEPS.PHYS_OPT_DESIGN.ARGS.DIRECTIVE AggressiveExplore [get_runs impl_1]
# set_property STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE AlternateCLBRouting [get_runs impl_1]
# set_property STEPS.POST_ROUTE_PHYS_OPT_DESIGN.IS_ENABLED true [get_runs impl_1]
# set_property STEPS.POST_ROUTE_PHYS_OPT_DESIGN.ARGS.DIRECTIVE AggressiveExplore [get_runs impl_1]


# launch bitstream generation
launch_runs impl_1 -to_step write_bitstream -jobs 32
wait_on_run impl_1