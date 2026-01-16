using ITensors
using LinearAlgebra
using ITensors.HDF5
using JLD
using ArgParse
import Base.Filesystem.mkpath

# explanation for the model ------------can also see in FIG.1 of http://arxiv.org/abs/cond-mat/0609051v2  ---
#        t1
#      o---o---o---o
#      |  /|  /|  /|
#   tR | / | / | / |   
#      |/tD|/  |/  |
#      o---o---o---o
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
        "-D"
            help = "The maximum bond dimension"
            default = 100
            arg_type = Int
        "--Dstep"
            help = "The step of increasing maxdim in DMRG"
            default = 100
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

function generate_mps_path( N, t1, t2, tR, tD, J, U, D)
    if !isdir("./psi/N$(N)_t($(t1),$(t2))_tR$(tR)_tD$(tD)_J$(J)_U$(U)/Dmax$(D)")
         mkpath("./psi/N$(N)_t($(t1),$(t2))_tR$(tR)_tD$(tD)_J$(J)_U$(U)/Dmax$(D)")
    end
    mps_path="./psi/N$(N)_t($(t1),$(t2))_tR$(tR)_tD$(tD)_J$(J)_U$(U)/Dmax$(D)/AKLT_NN$(N)_t($(t1),$(t2))_tR$(tR)_tD$(tD)_J$(J)_U$(U)_Dmax$(D).h5"
    return mps_path
end

function load_mps(mps_path)
    println("load from init")
    f = h5open(mps_path, "r")
    psi0 = read(f, "psi0", MPS)
    close(f)
    return psi0
end

function save_checkpoint(psi0, mps_path)
    f = h5open(mps_path,"w")
    write(f,"psi0",psi0)
    close(f)
end

function create_sites(N::Int64)
    sites = siteinds("Electron", 2*N; conserve_qns=false)#, conserve_nf=true
    return sites
end

function lpos(i::Int64, alpha::Int64) #using the coordinate of ladder to read out the position in the sites
    return 2*i-2+alpha
end

function system_ham(N::Int64, t1::Float64, t2::Float64, tR::Float64, tD::Float64, J::Float64, U::Float64)
    t = [t1, t2]
    os = OpSum()
    for j=1:N
        for alpha = 1:2
            idx   = lpos(i,alpha)
            idx_a = lpos(i%N+1,alpha)        #next idx in the same ladder
            idx_d = (idx-3+2*alpha+N)%(2*N)  #next idx for tD (excepte for site(1,1))
            idx_r = idx+3-2*alpha            #next idx for tR
            #in-ladder terms
            os += -t[alpha], "Cdagup", idx, "Cup" ,idx_a
            os += -t[alpha], "Cdagup", idx_a, "Cup" ,idx
            os += -U/2, "Nup",idx
            os += -t[alpha], "Cdagdn", idx, "Cdn" ,idx_a
            os += -t[alpha], "Cdagdn", idx_a, "Cdn" ,idx
            os += -U/2, "Ndn",idx
            #inter-ladder terms
            os += -tR, "Cdagup", idx, "Cup", idx_r
            os += -tR, "Cdagdn", idx, "Cdn", idx_r
            if idx ==1
                os += -tD, "Cdagup", idx, "Cup", 2*N
                os += -tD, "Cdagdn", idx, "Cdn", 2*N
            else
                os += -tD, "Cdagup", idx, "Cup", idx_d
                os += -tD, "Cdagdn", idx, "Cdn", idx_d
            end
            #repulsive interaction terms
            os += U, "Nup", idx, "Ndn", idx
        end
            #spin coupling terms
            idx = lpos(i,1)
            os += J/2,"S+",idx,"S-",idx+1
            os += J/2,"S-",idx,"S+",idx+1
            os += J,  "Sz",idx,"Sz",idx+1
    end
    return os
end

function create_psi0(N::Int, load::Bool, mps_path)
    if load
        psi0     = load_mps(mps_path)
    else
        state1   = [isodd(n) ? "Up" : "Dn" for n=1:2*N] #按奇偶分up/down
        state2   = [isodd(n) ? "Dn" : "Up" for n=1:2*N]
        psi0     = productMPS(sites, state1) + productMPS(sites, state2)
    end
    return psi0
end

function dmrg_GS(psi0, H, mps_path; eps=1e-10)
    nsweeps = 1
    Elast   = Inf
    noise   = [1e-6]
    cutoff  = [0]
    for n = 1:400
        maxdim = initD *(n-1)
        energy, psi = dmrg(H, psi0; nsweeps, maxdim, cutoff, noise)
        if abs(Elast-energy) < eps
            break
        end
        Elast = energy
        save_checkpoint(psi, mps_path)
    end
end

function main()
    args = parse_commandline()
    @show args
    load  = args["load"]
    N     = args["N"]
    D     = args["D"]
    t1    = args["t1"]
    t2    = args["t2"]
    tR    = args["tR"]
    tD    = args["tD"]
    J     = args["J"]
    initD = args["initD"]
    U     = args["U"]
    mps_path = generate_mps_path(N, t1, t2, tR, tD, J, U, D)
    sites = create_sites(N)
    os    = system_ham(N, t1, t2, tR, tD, J, U)
    HS    = MPO(os, sites)
    psi0  = create_psi0(N, load, mps_path)

    dmrg_GS(psi0, HS, mps_path)
end
main()
