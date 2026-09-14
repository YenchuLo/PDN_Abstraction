#!/usr/bin/env tclsh
# Localized Pixel-R abstraction (local_flow).
# Reuses spec_flow stages 01–03 (same C4-overlap ports / G'), then fits
# region-wise (Rx, Ry, Rz) via uniform IR init + Feng block τ.
#   tclsh local_flow/run_flow.tcl
#   tclsh local_flow/run_flow.tcl ibmpg4
#   tclsh local_flow/run_flow.tcl ibmpg4 --block 1
#   tclsh local_flow/run_flow.tcl ibmpg2 --block 0 --seed 0
#   tclsh local_flow/run_flow.tcl ibmpg2 --block 1 --uniform-current
#   tclsh local_flow/run_flow.tcl ibmpg2 --block 1 --uniform-current --fit eigen
#   tclsh local_flow/run_flow.tcl ibmpg2 ir --block 1 --uniform-current
#   tclsh local_flow/run_flow.tcl ibmpg2 --block 1 --uniform-current --pitch 463

# === user parameters (edit these) ===
set ibm_case    "ibmpg2"
set chip_pitch  0
set seed        0
set block       2
# fit_method: ir (Feng, default) | eigen (spectral LS)
set fit_method  "ir"
# Uniform sink current: "" / 0 / off = real lumped I;
# "conserve" (or 1) = equal share of total I on every grid cell;
# positive number = fixed per-sink draw in amps.
set uniform_current ""
# When on: every grid cell gets a voltage pad (C4 overlap + nearest top-metal fill).
set full_pads     0

# === derived (do not edit) ===
set here [file dirname [file normalize [info script]]]
set src [file join $here src]
set spec_src [file join [file dirname $here] spec_flow src]
set repo [file dirname $here]

set positionals {}
for {set i 0} {$i < [llength $argv]} {incr i} {
    set a [lindex $argv $i]
    if {$a eq "--seed"} {
        incr i
        if {$i >= [llength $argv]} {
            puts stderr "error: --seed requires an integer"
            exit 1
        }
        set seed [lindex $argv $i]
    } elseif {$a eq "--block"} {
        incr i
        if {$i >= [llength $argv]} {
            puts stderr "error: --block requires an integer (0, 1, or K>=2)"
            exit 1
        }
        set block [lindex $argv $i]
    } elseif {[string match "--block=*" $a]} {
        set block [string range $a 8 end]
    } elseif {[string match "--seed=*" $a]} {
        set seed [string range $a 7 end]
    } elseif {$a eq "--fit"} {
        incr i
        if {$i >= [llength $argv]} {
            puts stderr "error: --fit requires eigen or ir"
            exit 1
        }
        set fit_method [lindex $argv $i]
    } elseif {[string match "--fit=*" $a]} {
        set fit_method [string range $a 6 end]
    } elseif {$a eq "--uniform-current"} {
        set nxt [expr {$i + 1}]
        set take 0
        if {$nxt < [llength $argv]} {
            set cand [lindex $argv $nxt]
            set cl [string tolower $cand]
            if {$cl in {conserve on off true false yes no none} \
                    || [string is double -strict $cand]} {
                set take 1
            }
        }
        if {$take} {
            incr i
            set uniform_current [lindex $argv $i]
        } else {
            set uniform_current "conserve"
        }
    } elseif {[string match "--uniform-current=*" $a]} {
        set uniform_current [string range $a 18 end]
        if {$uniform_current eq ""} {
            set uniform_current "conserve"
        }
    } elseif {$a eq "--full-pads"} {
        set full_pads 1
    } elseif {$a eq "--pitch" || $a eq "--chip-pitch"} {
        incr i
        if {$i >= [llength $argv]} {
            puts stderr "error: --pitch requires a number (0 = auto)"
            exit 1
        }
        set chip_pitch [lindex $argv $i]
    } elseif {[string match "--pitch=*" $a]} {
        set chip_pitch [string range $a 8 end]
    } elseif {[string match "--chip-pitch=*" $a]} {
        set chip_pitch [string range $a 13 end]
    } elseif {[string match "--*" $a]} {
        puts stderr "error: unknown option: $a"
        exit 1
    } else {
        lappend positionals $a
    }
}
if {[llength $positionals] >= 1} {
    set ibm_case [lindex $positionals 0]
}
if {[llength $positionals] >= 2} {
    set fit_method [lindex $positionals 1]
}
if {[llength $positionals] >= 3} {
    puts stderr "error: unexpected positional args: [lrange $positionals 2 end]"
    exit 1
}
set fit_method [string tolower [string trim $fit_method]]
if {$fit_method ne "ir" && $fit_method ne "eigen"} {
    puts stderr "error: fit_method must be ir or eigen (got: $fit_method)"
    exit 1
}

proc resolve_spice {repo case} {
    set c [string tolower [string trim $case]]
    set n ""
    if {[string match "*.sp" $c] || [string match "*.spice" $c] || [file exists $case]} {
        if {[file pathtype $case] eq "absolute"} {
            return [file normalize $case]
        }
        return [file normalize [file join $repo $case]]
    }
    if {$c eq "tsmc1" || $c eq "tsmc/tc1" || $c eq "tsmc_tc1"} {
        return [file normalize [file join $repo Benchmarks TSMC TC1 \
            myALL_nportModel_without_package.sp]]
    }
    if {[regexp {^tc([1-6])$} $c -> n]} {
    } elseif {[regexp {^ibmpg([1-6])$} $c -> n]} {
    } elseif {[regexp {^([1-6])$} $c -> n]} {
    } else {
        puts stderr "error: case must be ibmpg1..6, TC1..6, tsmc1, or a .sp path (got: $case)"
        exit 1
    }
    return [file normalize [file join $repo Benchmarks IBM "TC$n" "ibmpg$n.spice"]]
}

set spice_abs [resolve_spice $repo $ibm_case]
if {![file exists $spice_abs]} {
    puts stderr "error: spice file not found: $spice_abs"
    exit 1
}

set stem [file rootname [file tail $spice_abs]]
if {$chip_pitch <= 0} {
    set pitch_tag "auto"
} else {
    set pitch_tag $chip_pitch
}
set u_tag ""
set uc [string tolower [string trim $uniform_current]]
if {$uc ne "" && $uc ne "0" && $uc ne "off" && $uc ne "false" && $uc ne "none"} {
    if {$uc eq "conserve" || $uc eq "1" || $uc eq "true" || $uc eq "yes" || $uc eq "on"} {
        set u_tag "_uI"
        set uniform_current "conserve"
    } else {
        set u_tag "_uI${uniform_current}"
    }
} else {
    set uniform_current ""
}
set fp_tag ""
if {$full_pads} {
    set fp_tag "_fullPads"
}
set f_tag ""
if {$fit_method eq "eigen"} {
    set f_tag "_eigen"
}
set OUT [file join $here outputs ${stem}_n${pitch_tag}${u_tag}${fp_tag}_b${block}${f_tag}]

puts "ibm_case    = $ibm_case"
puts "spice_path  = $spice_abs"
puts "chip_pitch  = $chip_pitch"
puts "seed        = $seed"
puts "block       = $block"
puts "fit_method  = $fit_method"
puts "uniform_current = $uniform_current"
puts "full_pads   = $full_pads"
puts "OUT         = $OUT"

file mkdir $OUT

proc run_stage {label cmd} {
    puts "\n=== $label ==="
    if {[catch {exec {*}$cmd >@ stdout 2>@ stderr} err]} {
        puts stderr "error: stage failed: $label"
        puts stderr $err
        exit 1
    }
}

set py python3
if {[info exists ::env(MY_FLOW_PYTHON)] && $::env(MY_FLOW_PYTHON) ne ""} {
    set py $::env(MY_FLOW_PYTHON)
} elseif {![catch {exec python -c "import scipy"}]} {
    set py python
}
puts "python      = $py"

run_stage "01_parse" [list $py [file join $spec_src stage01_parse.py] $spice_abs $OUT]
if {$chip_pitch <= 0} {
    set ports_cmd [list $py [file join $spec_src stage02_ports.py] $OUT]
} else {
    set ports_cmd [list $py [file join $spec_src stage02_ports.py] $OUT $chip_pitch]
}
if {$uniform_current ne ""} {
    if {$uniform_current eq "conserve"} {
        lappend ports_cmd --uniform-current
    } else {
        lappend ports_cmd --uniform-current $uniform_current
    }
}
if {$full_pads} {
    lappend ports_cmd --full-pads
}
run_stage "02_ports" $ports_cmd

set comps {}
set comp_json [file join $OUT components.json]
if {![file exists $comp_json]} {
    puts stderr "error: missing $comp_json"
    exit 1
}
set fh [open $comp_json r]
set raw [read $fh]
close $fh
foreach {_ path} [regexp -all -inline {"path"\s*:\s*"(comp[0-9]+)"} $raw] {
    lappend comps $path
}
if {[llength $comps] == 0} {
    puts stderr "error: no components listed in $comp_json"
    exit 1
}
puts "\nProcessing [llength $comps] VDD component(s): $comps"

foreach COMP $comps {
    set COMP_OUT [file join $OUT $COMP]
    puts "\n######## $COMP ########"
    run_stage "${COMP}_03_assemble_kron" \
        [list $py [file join $spec_src stage03_assemble_kron.py] $COMP_OUT]
    run_stage "${COMP}_04_localized" \
        [list $py [file join $src stage04_model.py] $COMP_OUT --block $block]
    if {$fit_method eq "eigen"} {
        run_stage "${COMP}_05_fit_eigen" \
            [list $py [file join $src stage05_fit_eigen.py] $COMP_OUT]
    } else {
        run_stage "${COMP}_05_fit_ir" \
            [list $py [file join $src stage05_fit_ir.py] $COMP_OUT $seed]
    }
    run_stage "${COMP}_08_correlate" \
        [list $py [file join $src stage08_correlate.py] $COMP_OUT $seed]
}

puts "\nDone. Results in: $OUT"
foreach COMP $comps {
    set m [file join $OUT $COMP metrics.json]
    if {[file exists $m]} {
        puts "$COMP metrics.json written"
    }
}
