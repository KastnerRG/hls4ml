set tcldir [file dirname [info script]]
source [file join $tcldir project.tcl]

add_files ${project_name}_prj/solution1/syn/verilog
synth_design -top ${project_name} -part $part
opt_design -retarget -propconst -sweep -shift_register_opt
report_utilization -file vivado_synth.rpt
