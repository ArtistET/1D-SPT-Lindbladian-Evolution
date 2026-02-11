using ITensors, ITensorMPS
using LinearAlgebra
using JLD2
using ArgParse
import Base.Filesystem.mkpath

include("AKLT_GS.jl")

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--load"
            help = "load init ground state or not"
            default = true
            arg_type = Bool
        "-N"
            help = "Half of the system size, which is the size of one branch of the ladder, N is recommanded to be even"
            arg_type = Int
        "--t1"
            help = "The in-ladder1 hopping t1"
            default = 0.1
            arg_type = Float64
        "--t2"
            help = "The in-ladder2 hopping t2"
            default = 0.2
            arg_type = Float64
        "--tR"
            help = "The in-site hopping tR"
            default = 1
            arg_type = Float64
        "--tD"
            help = "The diagonal hopping tD"
            arg_type = Float64
        "-J"
            help = "The coupling J"
            default = 0
            arg_type = Float64
        "--Dmax"
            help = "The maximum bond dimension loaded"
            default = 100
            arg_type = Int
        "--Dstep"
            help = "The Dstep loaded"
            default = 20
            arg_type = Int
        "-U"
            help = "The repulsive interaction relative to t"
            arg_type = Float64
        # "-f"
        #     help = "The filling of electron"
        #     arg_type = Float64
    end

    return parse_args(s)
end

function main()
    args = parse_commandline()
    @show args
    load  = args["load"]
    N     = args["N"]
    t1    = args["t1"]
    t2    = args["t2"]
    tR    = args["tR"]
    tD    = args["tD"]
    J     = args["J"]
    Dmax  = args["Dmax"]
    Dstep = args["Dstep"]
    U     = args["U"]
    load_path = generate_mps_path(N, t1, t2, tR, tD, J, U, Dmax, Dstep)
    sites, psi = create_psi0_for_dmrg(N, load, load_path)

    idx_st = div(N,4)
    idx_ed = N-div(N,4)
    SO_h_odd, SO_b_odd, SO_t_odd    = create_SO(sites, idx_st, idx_ed, N, "odd")
    SO_h_even, SO_b_even, SO_t_even = create_SO(sites, idx_st, idx_ed, N, "even")


    C_odd, SOV_odd     = measure(SO_h_odd, SO_b_odd, SO_t_odd, psi)
    C_even, SOV_even   = measure(SO_h_even, SO_b_even, SO_t_even, psi)

    check_entanglement(psi, N)

    println("Dmid= ", dim(linkind(psi, N))) #Get the linkdim of the middle of psi (N~N+1), which is also the maximum bond dimension in a finite MPS.
    println("complex SO_odd= ", C_odd, "  SO_odd= ", SOV_odd)
    println("complex SO_even= ", C_even, "  SO_even= ", SOV_even)
    if SOV_odd > SOV_even
        println("This phase is SPT")
    elseif SOV_odd==SOV_even
        println("Now maybe at the critical point")
    else
        println("This phase is trivial")
    end
end


if abspath(PROGRAM_FILE) == @__FILE__ # only run this code when directly running, including will not trigger main()
    main()
end
