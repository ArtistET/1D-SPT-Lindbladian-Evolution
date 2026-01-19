using ITensors, ITensorMPS
using LinearAlgebra
using JLD2
using ArgParse
import Base.Filesystem.mkpath

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
        "-U"
            help = "The repulsive interaction relative to t"
            arg_type = Float64
        # "-f"
        #     help = "The filling of electron"
        #     arg_type = Float64
    end

    return parse_args(s)
end

function generate_mps_path( N, t1, t2, tR, tD, J, U, Dmax, Dstep)
    if !isdir("./ground_states/N$(N)_t($(t1),$(t2))_tR$(tR)_tD$(tD)_J$(J)_U$(U)/Dmax$(Dmax)/Dstep$(Dstep)")
         mkpath("./ground_states/N$(N)_t($(t1),$(t2))_tR$(tR)_tD$(tD)_J$(J)_U$(U)/Dmax$(Dmax)/Dstep$(Dstep)")
    end
    mps_path="./ground_states/N$(N)_t($(t1),$(t2))_tR$(tR)_tD$(tD)_J$(J)_U$(U)/Dmax$(Dmax)/Dstep$(Dstep)/AKLT_NN$(N)_t($(t1),$(t2))_tR$(tR)_tD$(tD)_J$(J)_U$(U)_Dmax$(Dmax).jld2"
    return mps_path
end

function load_mps(mps_path)
    println("load from init")
    @load mps_path psi0
    return psi0
end

function save_checkpoint(psi0, mps_path)
    @save mps_path psi0
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
            idx   = lpos(j,alpha)
            idx_a = lpos(j%N+1,alpha)        #next idx in the same ladder (a for same "alpha")
            idx_d = (idx-3+2*alpha)%(2*N)    #next idx for tD (excepte for site(1,1)) 
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
                os += -tD, "Cdagup", 1, "Cup", 2*N
                os += -tD, "Cdagdn", 1, "Cdn", 2*N
            else
                os += -tD, "Cdagup", idx, "Cup", idx_d
                os += -tD, "Cdagdn", idx, "Cdn", idx_d
            end
            #repulsive interaction terms
            os += U, "Nup", idx, "Ndn", idx
        end
            #spin coupling terms
            idx = lpos(j,1)
            os += J/2,"S+",idx,"S-",idx+1
            os += J/2,"S-",idx,"S+",idx+1
            os += J,  "Sz",idx,"Sz",idx+1
    end
    return os
end

function create_psi0_for_dmrg(sites, N::Int, load::Bool, mps_path)
    if load
        psi0     = load_mps(mps_path)
    else
        state1   = [isodd(n) ? "Up" : "Dn" for n=1:2*N] #按奇偶分up/down
        state2   = [isodd(n) ? "Dn" : "Up" for n=1:2*N]
        psi0     = productMPS(sites, state1) + productMPS(sites, state2)
    end
    return psi0
end

function dmrg_GS(load, sites, N, H, mps_path, initD, Dstep, Dmax; eps=1e-10)
    psi0  = create_psi0_for_dmrg(sites, N, load, mps_path)
    normalize!(psi0)
    orthogonalize!(psi0, 1)
    psi = psi0
    nsweeps = 1
    energy   = Inf
    noise   = [1e-6]
    cutoff  = [0]
    for n = 1:400
        maxdim = min(initD+Dstep*(n-1) , Dmax)
        E_now, psi = dmrg(H, psi; nsweeps, maxdim, cutoff, noise)
        E_diff      = abs(E_now-energy)
        println("Now step No.", n, " , now energy difference ", E_diff)
        if E_diff < eps
            break
        end
        energy = E_now
        save_checkpoint(psi, mps_path)
    end
    normalize!(psi)
    orthogonalize!(psi, 1)
    save_checkpoint(psi, mps_path)
    return energy, psi
end

function rotation_Z2(sites, idx) # to generate the Z2 rotation which needed in stirng order
    Sz = op("Sz", sites, idx)
    Rz = exp(1im * pi * Sz)
    return Rz
end

function create_SO(sites, i_st, i_end, N, odd_even::String) #observe the string order, have to claim it's even(trivial) or odd(SPT),1<=i_st<i_end<=N and keep i_end>=i_st+2, so that N>=3 is recommended
    os_head = OpSum()
    SO_body = ITensor[]
    os_tail = OpSum()
    if odd_even == "odd" #odd string for SPT 
        os_head += "Sz", 2*i_st-1
        os_head += "Sz", 2*i_st
        os_tail += "Sz", 2*i_end-1
        os_tail += "Sz", 2*i_end
        SO_head =  MPO(os_head, sites)
        SO_tail =  MPO(os_tail, sites)
        for j=i_st+1:i_end-1
            for alpha=1:2
                Rz = rotation_Z2(sites, lpos(j, alpha))
                push!(SO_body, Rz)
            end
        end
    else               #even string for trivial 
        os_head += "Sz", 2*i_st
        os_head += "Sz", 2*i_st+1
        os_tail += "Sz", 2*i_end
        os_tail += "Sz", (2*i_end)%(2*N)+1
        SO_head =  MPO(os_head, sites)
        SO_tail =  MPO(os_tail, sites)
        for j=i_st+1:i_end-1
            for alpha=1:2
                Rz = rotation_Z2(sites, lpos(j-alpha+2, alpha))
                push!(SO_body, Rz)
            end
        end
    end
    return SO_head, SO_body, SO_tail
end

function measure(SO_head, SO_body, SO_tail, psi; cutoff=1e-10) # apply the operator to MPS to get the expectation
    psi_after = apply(SO_head, psi)
    for single_Rz in SO_body
        psi_after = apply(single_Rz, psi_after)
    end
    psi_after = apply(SO_tail, psi_after; cutoff=cutoff)
    C_value = -inner(psi, psi_after)
    SO_value = real(C_value)
    return C_value, SO_value
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
    Dload = args["Dload"]
    U     = args["U"]
    mps_path = generate_mps_path(N, t1, t2, tR, tD, J, U, Dmax, Dstep)
    sites = create_sites(N)
    os    = system_ham(N, t1, t2, tR, tD, J, U)
    HS    = MPO(os, sites)

    SO_h_odd, SO_b_odd, SO_t_odd    = create_SO(sites, 1, N, N, "odd")
    SO_h_even, SO_b_even, SO_t_even = create_SO(sites, 1, N, N, "even")

    energy,psi = dmrg_GS(load, sites, N, HS, mps_path, initD, Dstep, Dmax)

    C_odd, SOV_odd     = measure(SO_h_odd, SO_b_odd, SO_t_odd, psi)
    C_even, SOV_even   = measure(SO_h_even, SO_b_even, SO_t_even, psi)
    println("E= ", energy)
    println("complex SO_odd= ", C_odd, "SO_odd= ", SOV_odd, "SO_even= ", SOV_even)
    println("complex SO_even= ", C_even, "SO_even= ", SOV_even)
    if SOV_odd > SOV_even
        println("This phase is SPT")
    elseif SOV_odd==SOV_even
        println("Now maybe at the critical point")
    else
        println("This phase is trivial")
    end
end
main()
