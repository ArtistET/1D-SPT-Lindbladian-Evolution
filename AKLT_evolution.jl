using ITensors, ITensorMPS
using LinearAlgebra
using JLD2
using ArgParse
import Base.Filesystem.mkpath

include("AKLT_GS.jl")

# explanation for the model ------------can also see in FIG.1 of http://arxiv.org/abs/cond-mat/0609051v2  ---
#        t1
#      o---o---o---o  alpha=1
#      |  /|  /|  /|
#   tR | / | / | / |   
#      |/tD|/  |/  |
#      o---o---o---o  alpha=2
#        t2
#------------------------------------------------------------------------------------------------------------
function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--load"
            help = "load init ground state or not"
            default = true
            arg_type = Bool
        "--loadsl"
            help = "load a evolution slice or not"
            default = false
            arg_type = Bool
        "--loadt"
            help = "the t of the slice loaded"
            arg_type = Float64
        "-N"
            help = "Half of the system size, which is the size of one branch of the ladder, N is recommanded to be even"
            arg_type = Int
        "--Dmax"
            help = "The maximum bond dimension"
            default = 100
            arg_type = Int
        "--Dstep"
            help = "The step of increasing maxdim in DMRG"
            default = 20
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
        "--initD"
            help = "The initial bond dimension"
            default = 10
            arg_type = Int
        "--Dload"
            help = "The maximum bond dimension loaded"
            default = 100
            arg_type = Int
        "--Dstepload"
            help = "The Dstep loaded"
            default = 20
            arg_type = Int
        "-U"
            help = "The repulsive interaction relative to t"
            arg_type = Float64
        "--dt"
            help = "The step of time"
            arg_type = Float64
        "--tsmax"
            help = "The maximum step number of t"
            arg_type = Int
        # "-f"
        #     help = "The filling of electron"
        #     arg_type = Float64
    end

    return parse_args(s)
end

function load_slice(slice_path, t)
    println("Load from time slice where t= ", t)
    @load slice_path psi0
    return psi0
end

function generate_slice_path(t, N, t1, t2, tR, tD, J, U, Dmax, Dstep)
    if !isdir("./psi_evolution/T$(t)_N$(N)_t($(t1),$(t2))_tR$(tR)_tD$(tD)_J$(J)_U$(U)/Dmax$(Dmax)/Dstep$(Dstep)")
         mkpath("./psi_evolution/T$(t)_N$(N)_t($(t1),$(t2))_tR$(tR)_tD$(tD)_J$(J)_U$(U)/Dmax$(Dmax)/Dstep$(Dstep)")
    end
    mps_path="./psi_evolution/T$(t)_N$(N)_t($(t1),$(t2))_tR$(tR)_tD$(tD)_J$(J)_U$(U)/Dmax$(Dmax)/Dstep$(Dstep)/AKLT_T$(t)__N$(N)_t($(t1),$(t2))_tR$(tR)_tD$(tD)_J$(J)_U$(U)_Dmax$(Dmax).jld2"
    return mps_path
end

function create_psi0_for_evolution(N::Int, load::Bool, loadsl, loadt, HS, mps_path, slice_path, psi0)
    if load
        if loadsl
            psi     = load_slice(slice_path, loadt)
    else
        energy, psi = dmrg_GS(false, N, HS, mps_path, psi0, initD, Dstep, Dmax)
    end
    return psi
end

function main()
    args = parse_commandline()
    @show args
    load  = args["load"]
    loadsl= args["loadsl"]
    loadt = args["loadt"]
    N     = args["N"]
    Dmax  = args["Dmax"]
    t1    = args["t1"]
    t2    = args["t2"]
    tR    = args["tR"]
    tD    = args["tD"]
    J     = args["J"]
    initD = args["initD"]
    Dstep = args["Dstep"]
    Dload = args["Dload"]
    Dstepload = args["Dstepload"]
    U     = args["U"]
    dt    = args["dt"]
    tsmax = args["tsmax"]
    load_path = generate_mps_path(N, t1, t2, tR, tD, J, U, Dload, Dstepload)
    slice_path= generate_slice_path(t, N, t1, t2, tR, tD, J, U, Dmax, Dstep)
    slice_load_path= generate_slice_path(t, N, t1, t2, tR, tD, J, U, Dload, Dstepload)
    slice_path
    sites, psi0  = create_psi0_for_dmrg(N, load, load_path)
    os    = system_ham(N, t1, t2, tR, tD, J, U)
    HS    = MPO(os, sites)
    psi0  = create_psi0_for_evolution(N, load, loadsl, loadt, HS, mps_path, slice_load_path, psi0)
end
main()