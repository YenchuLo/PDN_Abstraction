#!/usr/bin/env tclsh
# my_flow: parse → ports → Kron → tri-stagger + tri-square.
# Edit parameters below, then:
#   tclsh my_flow/run_flow.tcl
#   tclsh my_flow/run_flow.tcl ibmpg3
#   tclsh my_flow/run_flow.tcl ibmpg3 --simulator spectre

# === user parameters (edit these) ===
# Case: ibmpg1..ibmpg6 / TC1..TC6 (IBM), or tsmc1 / tsmc/tc1 (TSMC n-port
# under Benchmarks/TSMC/TC1/), or an explicit .sp/.spice path relative to
# the repo root.
set ibm_case    "ibmpg2"
# Fine mesh pitch; 0 = auto C4 lattice (same rule as spec_flow), then clamp
# to > max VDD metal stripe pitch.
set pitch_bot   0
set coarsen_k   1
# Divide C4 lattice pitch by sqrt(ratio): 1.0 = match bump spacing; 4.0 denser.
set grid_to_pad_ratio 1.0
set seed        0
# Tri-square L3 denser than L2 by k (pitch_l = pitch_bot/k).
set tri_square_k 2
# Optional SPICE decks for stagger (--spice). simulator: ngspice | spectre
set emit_spice  0
set simulator   "ngspice"
# Uniform sink current: "" / 0 / off = real lumped I;
# "conserve" (or 1) = equal share of total I on every fine-grid cell;
# positive number = fixed per-sink draw in amps.
set uniform_current ""
# Grid-cell R perturbation: "" = off; metal layer int (e.g. 1 or 5); or "all"
# for every metal in the current VDD rail stack. One U per port cell; vias untouched.
set perturb_layer ""
set perturb_amp   0.2
# Pad/sink attach: rxry (default L-bend ∝ Rx/Ry) | zero (snap to nearest grid)
set via_stub      "rxry"
# When on: every fine-grid cell gets a voltage pad (C4 + nearest top-metal fill).
set full_pads     0

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
    } elseif {$a eq "--spice"} {
        set emit_spice 1
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
    } elseif {$a eq "--via-stub"} {
        incr i
        if {$i >= [llength $argv]} {
            puts stderr "error: --via-stub requires zero or rxry"
            exit 1
        }
        set via_stub [lindex $argv $i]
    } elseif {[string match "--via-stub=*" $a]} {
        set via_stub [string range $a 11 end]
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
    puts stderr "error: unexpected positional args: [lrange $positionals 1 end]"
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
if {$simulator ne "ngspice" && $simulator ne "spectre"} {
    puts stderr "error: simulator must be ngspice or spectre (got: $simulator)"
    exit 1
}
if {[info exists ::env(SPICE_SIMULATOR)] && $::env(SPICE_SIMULATOR) ne ""} {
    set simulator $::env(SPICE_SIMULATOR)
}

set stem [file rootname [file tail $spice_abs]]
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
set vs [string tolower [string trim $via_stub]]
if {$vs eq "" || $vs in {rxry nonzero non-zero on true yes 1 r}} {
    set via_stub "rxry"
    set s_tag ""
} elseif {$vs in {zero 0 off false no none short}} {
    set via_stub "zero"
    set s_tag "_nostub"
} else {
    puts stderr "error: via_stub must be zero or rxry (got: $via_stub)"
    exit 1
}
set fp_tag ""
if {$full_pads} {
    set fp_tag "_fullPads"
}
if {$pitch_bot <= 0} {
    set OUT [file join $here outputs ${stem}_auto_k${coarsen_k}${u_tag}${fp_tag}${p_tag}${s_tag}]
} else {
    set OUT [file join $here outputs ${stem}_pb${pitch_bot}_k${coarsen_k}${u_tag}${fp_tag}${p_tag}${s_tag}]
}

puts "ibm_case   = $ibm_case"
puts "spice_path  = $spice_abs"
puts "pitch_bot   = $pitch_bot"
puts "coarsen_k   = $coarsen_k"
puts "grid_to_pad_ratio = $grid_to_pad_ratio"
puts "seed        = $seed"
puts "tri_square_k = $tri_square_k"
puts "emit_spice  = $emit_spice"
puts "simulator   = $simulator"
puts "uniform_current = $uniform_current"
puts "full_pads   = $full_pads"
puts "perturb_layer = $perturb_layer"
puts "perturb_amp   = $perturb_amp"
puts "via_stub    = $via_stub"
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
set ports_cmd [list $py [file join $src stage02_ports.py] $OUT $pitch_bot $coarsen_k \
     --grid-to-pad-ratio $grid_to_pad_ratio]
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
        [list $py [file join $src stage03_assemble_kron.py] $COMP_OUT]

    set tri_stagger_cmd [list $py [file join $src run_tri_stagger.py] $COMP_OUT $seed \
        --via-stub $via_stub]
    if {$emit_spice} {
        lappend tri_stagger_cmd --spice
    }
    run_stage "${COMP}_tri_stagger" $tri_stagger_cmd

    run_stage "${COMP}_tri_square" \
        [list $py [file join $src run_tri_square.py] $COMP_OUT $seed \
             --k $tri_square_k --via-stub $via_stub]
}

puts "\nDone. Results in: $OUT"
foreach COMP $comps {
    set sr [file join $OUT $COMP tri_stagger_r.json]
    set tr [file join $OUT $COMP tri_square_r.json]
    if {[file exists $sr]} { puts "$COMP tri_stagger_r.json written" }
    if {[file exists $tr]} { puts "$COMP tri_square_r.json written" }
}
