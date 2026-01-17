using ITensors, ITensorMPS
using LinearAlgebra
using ITensors.HDF5
using JLD
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
            default = false
            arg_type = Bool
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
            arg_type = Float64
        "--initD"
            help = "The initial bond dimension"
            default = 10
            arg_type = Int
        "-U"
            help = "The repulsive interaction relative to t"
            arg_type = Float64
        "-f"
            help = "The filling of electron"
            arg_type = Float64
    end

    return parse_args(s)
end

function create_psi0_for_evolution(N::Int, load::Bool, HS, mps_path)
    if load
        psi0         = load_mps(mps_path)
    else
        energy, psi0 = dmrg_GS(false, HS, mps_path, initD, Dstep, Dmax)
    end
    return psi0
end

function main()
    args = parse_commandline()
    @show args
    load  = args["load"]
    N     = args["N"]
    Dmax  = args["Dmax"]
    t1    = args["t1"]
    t2    = args["t2"]
    tR    = args["tR"]
    tD    = args["tD"]
    J     = args["J"]
    initD = args["initD"]
    Dstep = args["Dstep"]
    U     = args["U"]
    mps_path = generate_mps_path(N, t1, t2, tR, tD, J, U, Dmax, Dstep)
    sites = create_sites(N)
    os    = system_ham(N, t1, t2, tR, tD, J, U)
    HS    = MPO(os, sites)
    psi0  = create_psi0_for_evolution(N, load, HS, mps_path)
end
main()