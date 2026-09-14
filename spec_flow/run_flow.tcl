#!/usr/bin/env tclsh
# TSMC Pixel-R abstraction flow driver.
# Edit parameters below, then:
#   tclsh spec_flow/run_flow.tcl
#   tclsh spec_flow/run_flow.tcl ibmpg3
#   tclsh spec_flow/run_flow.tcl ibmpg3 ir     ;# optional fit_method
#   tclsh spec_flow/run_flow.tcl ibmpg3 --simulator spectre
#   tclsh spec_flow/run_flow.tcl ibmpg3 ir --simulator spectre

# === user parameters (edit these) ===
# Case: ibmpg1..ibmpg6 / TC1..TC6 (IBM), or tsmc1 / tsmc/tc1 (TSMC n-port
# under Benchmarks/TSMC/TC1/), or an explicit .sp/.spice path relative to
# the repo root.
set ibm_case    "ibmpg2"
# Grid cell side length. 0 = auto: C4 lattice pitch (median unique-row/col
# spacing), clamped to chip_pitch > max VDD metal stripe pitch.
set chip_pitch  0
set seed        0
# fit_method: eigen (default, TSMC spec) | ir (mixed-BC IR response fit)
set fit_method  "eigen"
# simulator: ngspice (default) | spectre
set simulator   "ngspice"
# Uniform sink current: "" / 0 / off = real lumped I;
# "conserve" (or 1) = equal share of total I on every grid cell;
# positive number = fixed per-sink draw in amps.
set uniform_current ""
# When on: every grid cell gets a voltage pad (C4 overlap + nearest top-metal fill).
set full_pads     0
# Grid-cell R perturbation: "" = off; metal layer int (e.g. 1 or 5); or "all"
# for every metal in the current VDD rail stack. One U per port cell; vias untouched.
set perturb_layer ""
set perturb_amp   0.2

# === derived (do not edit) ===
set here [file dirname [file normalize [info script]]]
set src [file join $here src]
set repo [file dirname $here]

set positionals {}
for {set i 0} {$i < [llength $argv]} {incr i} {
    set a [lindex $argv $i]
    if {$a eq "--simulator"} {
        incr i
        if {$i >= [llength $argv]} {
            puts stderr "error: --simulator requires a value (ngspice|spectre)"
            exit 1
        }
        set simulator [lindex $argv $i]
    } elseif {$a eq "--seed"} {
        incr i
        if {$i >= [llength $argv]} {
            puts stderr "error: --seed requires an integer"
            exit 1
        }
        set seed [lindex $argv $i]
    } elseif {$a eq "--perturb-layer"} {
        incr i
        if {$i >= [llength $argv]} {
            puts stderr "error: --perturb-layer requires a metal layer int or 'all'"
            exit 1
        }
        set perturb_layer [lindex $argv $i]
    } elseif {[string match "--perturb-layer=*" $a]} {
        set perturb_layer [string range $a 16 end]
    } elseif {$a eq "--perturb-amp"} {
        incr i
        if {$i >= [llength $argv]} {
            puts stderr "error: --perturb-amp requires a number"
            exit 1
        }
        set perturb_amp [lindex $argv $i]
    } elseif {[string match "--perturb-amp=*" $a]} {
        set perturb_amp [string range $a 14 end]
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

proc resolve_spice {repo case} {
    set c [string tolower [string trim $case]]
    set n ""
    # Explicit path (absolute or repo-relative)
    if {[string match "*.sp" $c] || [string match "*.spice" $c] || [file exists $case]} {
        if {[file pathtype $case] eq "absolute"} {
            return [file normalize $case]
        }
        return [file normalize [file join $repo $case]]
    }
    # TSMC n-port TC1 under Benchmarks/TSMC/TC1/
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

if {$fit_method ne "eigen" && $fit_method ne "ir"} {
    puts stderr "error: fit_method must be eigen or ir (got: $fit_method)"
    exit 1
}
if {$simulator ne "ngspice" && $simulator ne "spectre"} {
    puts stderr "error: simulator must be ngspice or spectre (got: $simulator)"
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
set p_tag ""
set pl [string tolower [string trim $perturb_layer]]
if {$pl ne "" && $pl ne "off" && $pl ne "none" && $pl ne "false"} {
    if {$pl ne "all" && ![string is integer -strict $pl]} {
        puts stderr "error: perturb_layer must be an integer metal layer or 'all' (got: $perturb_layer)"
        exit 1
    }
    if {![string is double -strict $perturb_amp] || $perturb_amp < 0} {
        puts stderr "error: perturb_amp must be >= 0 (got: $perturb_amp)"
        exit 1
    }
    set perturb_layer $pl
    set p_tag "_pL${pl}a${perturb_amp}s${seed}"
} else {
    set perturb_layer ""
}
if {$fit_method eq "ir"} {
    set OUT [file join $here outputs ${stem}_n${pitch_tag}${u_tag}${fp_tag}${p_tag}_ir]
} else {
    set OUT [file join $here outputs ${stem}_n${pitch_tag}${u_tag}${fp_tag}${p_tag}]
}

puts "ibm_case    = $ibm_case"
puts "spice_path  = $spice_abs"
puts "chip_pitch  = $chip_pitch"
puts "seed        = $seed"
puts "fit_method  = $fit_method"
puts "simulator   = $simulator"
puts "uniform_current = $uniform_current"
puts "full_pads   = $full_pads"
puts "perturb_layer = $perturb_layer"
puts "perturb_amp   = $perturb_amp"
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

set parse_cmd [list $py [file join $src stage01_parse.py] $spice_abs $OUT]
run_stage "01_parse" $parse_cmd
if {$chip_pitch <= 0} {
    set ports_cmd [list $py [file join $src stage02_ports.py] $OUT]
} else {
    set ports_cmd [list $py [file join $src stage02_ports.py] $OUT $chip_pitch]
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
if {$perturb_layer ne ""} {
    lappend ports_cmd --perturb-layer $perturb_layer --perturb-amp $perturb_amp \
        --seed $seed
}
run_stage "02_ports" $ports_cmd

# Discover component subdirs from components.json
set comps {}
set comp_json [file join $OUT components.json]
if {![file exists $comp_json]} {
    puts stderr "error: missing $comp_json"
    exit 1
}
set fh [open $comp_json r]
set raw [read $fh]
close $fh
# Minimal JSON list extraction for "path": "compN"
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
        [list $py [file join $src stage03_assemble_kron.py] $COMP_OUT]
    run_stage "${COMP}_04_pixel_model" \
        [list $py [file join $src stage04_pixel_model.py] $COMP_OUT]
    if {$fit_method eq "ir"} {
        run_stage "${COMP}_05_fit_ir" \
            [list $py [file join $src stage05_fit_ir.py] $COMP_OUT $seed]
    } else {
        run_stage "${COMP}_05_fit_eigen" \
            [list $py [file join $src stage05_fit_eigen.py] $COMP_OUT]
    }
    run_stage "${COMP}_06_emit_spice" \
        [list $py [file join $src stage06_emit_spice.py] $COMP_OUT \
             --simulator $simulator]
    run_stage "${COMP}_07_spice_solve" \
        [list $py [file join $src stage07_spice_solve.py] $COMP_OUT \
             --simulator $simulator]
    run_stage "${COMP}_08_correlate_ir" \
        [list $py [file join $src stage08_correlate_ir.py] $COMP_OUT $seed]
}

puts "\nDone. Results in: $OUT"
foreach COMP $comps {
    set m [file join $OUT $COMP metrics.json]
    if {[file exists $m]} {
        puts "$COMP metrics.json written"
    }
    set pr [file join $OUT $COMP pixel_r.json]
    if {[file exists $pr]} {
        puts "$COMP pixel_r.json written"
    }
}
