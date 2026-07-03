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
            default = 0.0
            arg_type = Float64
        "-N"
            help = "Half of the system size, which is the size of one branch of the ladder, N is recommanded to be even"
            arg_type = Int
        "--Dmax"
            help = "The maximum bond dimension"
            default = 400
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
        "--I1"
            help = "The intensity of in-ladder1 hopping operators"
            default = 0.1
            arg_type = Float64
        "--I2"
            help = "The intensity of in-ladder2 hopping operators"
            default = 0.1
            arg_type = Float64
        "--IR"
            help = "The intensity of in-site hopping operators"
            default = 0.1
            arg_type = Float64
        "--ID"
            help = "The intensity of diagonal hopping operators"
            default = 0.1
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
            default  = 0.01
            arg_type = Float64
        "--tsmax"
            help = "The maximum step number of t"
            default  = 20
            arg_type = Int
        # "-f"
        #     help = "The filling of electron"
        #     arg_type = Float64
    end

    return parse_args(s)
end

function load_slice(slice_path, t)
    println("Load from time slice where t= ", t)
    @load slice_path rho0
    return rho0
end

function save_slice(rho0, slice_path)
    @save slice_path rho0
end

function generate_slice_path(t, N, t1, t2, tR, tD, J, U, I1, I2, IR, ID, Dmax)
    if !isdir("./psi_evolution/T$(t)_N$(N)_t($(t1),$(t2))_tR$(tR)_tD$(tD)_J$(J)_U$(U)_I1$(I1)_I2$(I2)_IR$(IR)_ID$(ID)/Dmax$(Dmax)")
         mkpath("./psi_evolution/T$(t)_N$(N)_t($(t1),$(t2))_tR$(tR)_tD$(tD)_J$(J)_U$(U)_I1$(I1)_I2$(I2)_IR$(IR)_ID$(ID)/Dmax$(Dmax)")
    end
    mps_path="./psi_evolution/T$(t)_N$(N)_t($(t1),$(t2))_tR$(tR)_tD$(tD)_J$(J)_U$(U)_I1$(I1)_I2$(I2)_IR$(IR)_ID$(ID)/Dmax$(Dmax)/AKLT_T$(t)__N$(N)_t($(t1),$(t2))_tR$(tR)_tD$(tD)_J$(J)_U$(U)_I1$(I1)_I2$(I2)_IR$(IR)_ID$(ID)_Dmax$(Dmax).jld2"
    return mps_path
end

# function create_rho0_for_evolution(N::Int, load::Bool, loadsl, loadt, HS, mps_path, slice_path, psi0, initD, Dstep, Dmax)
#     if load
#         if loadsl
#             rho0    = load_slice(slice_path, loadt)
#         else
#             rho0    = outer(psi0', psi0;maxdim=Dmax, cutoff=1e-6)
#         end
#     else
#         energy, psi = dmrg_GS(N, HS, mps_path, psi0, initD, Dstep, Dmax)
#         rho0        = outer(psi', psi;maxdim=Dmax, cutoff=1e-6)
#     end
#     return rho0
# end

function SO_MPO(sites,SO_head,SO_body,SO_tail;cutoff=1e-8,maxdim=400)
    so_mpo = MPO(sites,"Id")
    so_mpo = apply(SO_head, so_mpo;cutoff=cutoff,maxdim=maxdim)
    for local_rot in SO_body
        so_mpo = apply(local_rot, so_mpo;cutoff=cutoff,maxdim=maxdim)
    end
    so_mpo = apply(SO_tail, so_mpo;cutoff=cutoff,maxdim=maxdim)
    return so_mpo
end

function measure_SO_for_rho(SO_head,SO_body,SO_tail,IN,rho_t)
    rho_t_after = apply(SO_head, rho_t)
    rho_t_after = apply(SO_body, rho_t_after)
    rho_t_after = apply(SO_tail, rho_t_after)
    C_value     = -inner(IN, rho_t_after)/inner(IN, rho_t)
    SO_value    = real(C_value)
    return C_value, SO_value
end

function get_sites_rho0_HS(N::Int, t1, t2, tR, tD, J, U, load::Bool, loadsl::Bool, loadt, mps_path, load_path, slice_load_path, initD, Dstep, Dmax)
    if load 
        if loadsl
            rho0    = load_slice(slice_load_path, loadt)
            psi0    = nothing
            sites   = firstsiteinds(rho0; plev=0)
            os      = system_ham(N, t1, t2, tR, tD, J, U)
            HS      = MPO(os, sites)
        else
            sites, psi0  = create_psi0_for_dmrg(N, load, load_path)
            os           = system_ham(N, t1, t2, tR, tD, J, U)
            HS           = MPO(os, sites)
            rho0         = outer(psi0', psi0;maxdim=Dmax, cutoff=1e-6)
        end
    else
        sites, psi00  = create_psi0_for_dmrg(N, load, load_path)
        os            = system_ham(N, t1, t2, tR, tD, J, U)
        HS            = MPO(os, sites)
        energy, psi0  = dmrg_GS(N, HS, mps_path, psi00, initD, Dstep, Dmax)
        rho0          = outer(psi', psi;maxdim=Dmax, cutoff=1e-6)
    end
    return sites,psi0 , rho0, HS
end

function create_hopping(sites, HS, dt,N::Int64, I1::Float64, I2::Float64, IR::Float64, ID::Float64) #hopping operators for Lindblad
    I         = [I1, I2]
    IN        = MPO(sites,"Id")
    sdt       = sqrt(dt)
    K_list    = MPO[]
    Kdag_list = MPO[]
    os = OpSum()
    for j=1:N
        for alpha = 1:2
            idx   = lpos(j,alpha)
            idx_a = lpos(j%N+1,alpha)        #next idx in the same ladder (a for same "alpha")
            #in-ladder terms
            os += I[alpha]^2, "Cdagup", idx_a, "Cup" ,idx, "Cdagup", idx, "Cup" ,idx_a
            os += I[alpha]^2, "Cdagup", idx, "Cup" ,idx_a, "Cdagup", idx_a, "Cup" ,idx
            os += I[alpha]^2, "Cdagdn", idx_a, "Cdn" ,idx, "Cdagdn", idx, "Cdn" ,idx_a
            os += I[alpha]^2, "Cdagdn", idx, "Cdn" ,idx_a, "Cdagdn", idx_a, "Cdn" ,idx
        end
        idx1 = lpos(j,1)
        idx2 = lpos(j,2)
        idxr = lpos(j%N+1,1)
        #inter-ladder terms
        os += IR^2, "Cdagup", idx2, "Cup", idx1, "Cdagup", idx1, "Cup", idx2
        os += IR^2, "Cdagup", idx1, "Cup", idx2, "Cdagup", idx2, "Cup", idx1
        os += IR^2, "Cdagdn", idx2, "Cdn", idx1, "Cdagdn", idx1, "Cdn", idx2
        os += IR^2, "Cdagdn", idx1, "Cdn", idx2, "Cdagdn", idx2, "Cdn", idx1
        os += ID^2, "Cdagup", idx2, "Cup", idxr, "Cdagup", idxr, "Cup", idx2
        os += ID^2, "Cdagup", idxr, "Cup", idx2, "Cdagup", idx2, "Cup", idxr
        os += ID^2, "Cdagdn", idx2, "Cdn", idxr, "Cdagdn", idxr, "Cdn", idx2
        os += ID^2, "Cdagdn", idxr, "Cdn", idx2, "Cdagdn", idx2, "Cdn", idxr
    end
    Lsum = MPO(os, sites)
    K0=IN+dt*(-1im*HS-0.5*Lsum)
    K0_dag=IN+dt*(1im*HS-0.5*Lsum)
    for j=1:N
        for alpha = 1:2
            idx   = lpos(j,alpha)
            idx_a = lpos(j%N+1,alpha)        #next idx in the same ladder (a for same "alpha")
            idx_d = (idx-3+2*alpha)%(2*N)    #next idx for tD (excepte for site(1,1)) 
            idx_r = idx+3-2*alpha            #next idx for tR
            #in-ladder terms
            oh =  OpSum()
            oh += sdt*I[alpha], "Cdagup", idx, "Cup" ,idx_a
            push!(K_list,MPO(oh,sites))
            
            ohd =  OpSum()
            ohd += sdt*I[alpha], "Cdagup", idx_a, "Cup" ,idx
            push!(Kdag_list,MPO(ohd,sites))

            oh =  OpSum()
            oh += sdt*I[alpha], "Cdagup", idx_a, "Cup" ,idx
            push!(K_list,MPO(oh,sites))

            ohd =  OpSum()
            ohd += sdt*I[alpha], "Cdagup", idx, "Cup" ,idx_a
            push!(Kdag_list,MPO(ohd,sites))

            oh =  OpSum()
            oh += sdt*I[alpha], "Cdagdn", idx, "Cdn" ,idx_a
            push!(K_list,MPO(oh,sites))

            ohd =  OpSum()
            ohd += sdt*I[alpha], "Cdagdn", idx_a, "Cdn" ,idx
            push!(Kdag_list,MPO(ohd,sites))

            oh =  OpSum()
            oh += sdt*I[alpha], "Cdagdn", idx_a, "Cdn" ,idx
            push!(K_list,MPO(oh,sites))

            ohd =  OpSum()
            ohd += sdt*I[alpha], "Cdagdn", idx, "Cdn" ,idx_a
            push!(Kdag_list,MPO(ohd,sites))

            #inter-ladder terms
            oh =  OpSum()
            oh += sdt*IR, "Cdagup", idx, "Cup", idx_r
            push!(K_list,MPO(oh,sites))
            oh =  OpSum()
            oh += sdt*IR, "Cdagdn", idx, "Cdn", idx_r
            push!(K_list,MPO(oh,sites))

            ohd =  OpSum()
            ohd += sdt*IR, "Cdagup", idx_r, "Cup", idx
            push!(Kdag_list,MPO(ohd,sites))
            ohd =  OpSum()
            ohd += sdt*IR, "Cdagdn", idx_r, "Cdn", idx
            push!(Kdag_list,MPO(ohd,sites))

            if idx ==1
                oh =  OpSum()
                oh += sdt*ID, "Cdagup", 1, "Cup", 2*N
                push!(K_list,MPO(oh,sites))
                oh =  OpSum()
                oh += sdt*ID, "Cdagdn", 1, "Cdn", 2*N
                push!(K_list,MPO(oh,sites))

                ohd =  OpSum()
                ohd += sdt*ID, "Cdagup", 2*N, "Cup", 1
                push!(Kdag_list,MPO(ohd,sites))
                ohd =  OpSum()
                ohd += sdt*ID, "Cdagdn", 2*N, "Cdn", 1
                push!(Kdag_list,MPO(ohd,sites))

            else
                oh =  OpSum()
                oh += sdt*ID, "Cdagup", idx, "Cup", idx_d
                push!(K_list,MPO(oh,sites))
                oh =  OpSum()
                oh += sdt*ID, "Cdagdn", idx, "Cdn", idx_d
                push!(K_list,MPO(oh,sites))

                ohd =  OpSum()
                ohd += sdt*ID, "Cdagup", idx_d, "Cup", idx
                push!(Kdag_list,MPO(ohd,sites))
                ohd =  OpSum()
                ohd += sdt*ID, "Cdagdn", idx_d, "Cdn", idx
                push!(Kdag_list,MPO(ohd,sites))
            end
        end
    end
    
    return K0,K0_dag,K_list,Kdag_list
end
function format_hms(sec) # format a duration (in seconds) as HH:MM:SS.mmm
    ms   = round(Int, (sec - floor(sec)) * 1000)
    isec = floor(Int, sec)
    h    = div(isec, 3600)
    m    = div(isec % 3600, 60)
    s    = isec % 60
    return string(lpad(h,2,'0'), ":", lpad(m,2,'0'), ":", lpad(s,2,'0'), ".", lpad(ms,3,'0'))
end
function tree_sum(terms::Vector{MPO};cutoff=1e-8,maxdim=400)
    current = copy(terms)
    while length(current) >1
        next = MPO[]
        i=1
        while i< length(current)
            tmp = current[i]+current[i+1]
            tbefore = time() #for testing 
            truncate!(tmp;cutoff=cutoff,maxdim=maxdim)
            push!(next,tmp)
            println("One sum step over ,time cost=", format_hms(time()-tbefore)," (hh:mm:ss)") #for testing 
            i +=2
        end
        if isodd(length(current))
            push!(next,current[length(current)])
        end
        current = next
    end
    println("Tree sum over") #for testing 
    return current[1]
end
function single_op_Lindblad(A,rho,Adag;cutoff=1e-8,maxdim=400)
    # tmp = apply(rho,Adag)
    # println("maxdim in contraction = ", maxlinkdim(tmp))
    # rho1 = apply(A,tmp;cutoff=cutoff,maxdim=maxdim)
    # rho1 = apply(A,apply(rho,Adag);cutoff=cutoff,maxdim=maxdim)  # more precise
    rho1 = apply(A,apply(rho,Adag;cutoff=cutoff,maxdim=maxdim);cutoff=cutoff,maxdim=maxdim)  # more available
    return rho1
end
function check_maxdim(K0,K_list)
    maxdim_list=[]
    push!(maxdim_list, maxlinkdim(K0))
    for Ki in K_list
        push!(maxdim_list, maxlinkdim(Ki))
    end
    println("bond dimensions for operators are ", maxdim_list," with total operator number =", length(maxdim_list))
    return 0
end
function single_step_Lindblad(rho,K0,K0_dag,K_list::Vector{MPO},Kdag_list::Vector{MPO};block_size=4,cutoff=1e-8,maxdim=400)
    buffer   = MPO[]
    rho_list = MPO[]
    rho_0    = single_op_Lindblad(K0,rho,K0_dag;cutoff=cutoff,maxdim=maxdim)
    println("K0 applied, progress 1/", 1+length(K_list))  #for tesing 
    push!(buffer, rho_0)
    for i = 1:length(K_list)
        rho_i = single_op_Lindblad(K_list[i],rho,Kdag_list[i];cutoff=cutoff,maxdim=maxdim)
        push!(buffer,rho_i)
        if length(buffer)== block_size || i==length(K_list)
            rho_sum = tree_sum(buffer;cutoff=cutoff,maxdim=maxdim)
            push!(rho_list,rho_sum)
            empty!(buffer)
        end
        println("K",i," applied, progress ", i+1,"/", 1+length(K_list)) #for testing
    end
    rho_ans = tree_sum(rho_list;cutoff=cutoff,maxdim=maxdim)
    return rho_ans
end
function Lindblad_evolution(dt,tsmax,init_t, N, t1, t2, tR, tD, J, U, I1, I2, IR, ID, Dmax,rho,K0,K0_dag,K_list,Kdag_list,SO_h_odd, SO_b_odd, SO_t_odd,SO_h_even, SO_b_even, SO_t_even,IN;block_size=4,cutoff=1e-8)
    for i = 1:tsmax
        slice_path_t = generate_slice_path(init_t+dt*(i-1), N, t1, t2, tR, tD, J, U, I1, I2, IR, ID, Dmax)
        save_slice(rho, slice_path_t)
        C_odd, SOV_odd = measure_SO_for_rho(SO_h_odd, SO_b_odd, SO_t_odd,IN,rho)
        C_even, SOV_even = measure_SO_for_rho(SO_h_even, SO_b_even, SO_t_even,IN,rho)
        println("At T = ", init_t+dt*(i-1),"="^30)
        println("complex SO_odd= ", C_odd, "  SO_odd= ", SOV_odd)
        println("complex SO_even= ", C_even, "  SO_even= ", SOV_even)
        t_step = time()
        rho = single_step_Lindblad(rho,K0,K0_dag,K_list,Kdag_list;block_size=block_size,cutoff=cutoff,maxdim=Dmax)
        println("Step ", i, "/", tsmax, " (T ", round(init_t+dt*(i-1), digits=6), " -> ", round(init_t+dt*i, digits=6), ") took ", format_hms(time()-t_step), " (hh:mm:ss) , maxlinkdim(rho)= ", maxlinkdim(rho))
    end
    slice_path_t = generate_slice_path(init_t+dt*tsmax, N, t1, t2, tR, tD, J, U, I1, I2, IR, ID, Dmax)
    save_slice(rho, slice_path_t)
    C_odd, SOV_odd = measure_SO_for_rho(SO_h_odd, SO_b_odd, SO_t_odd,IN,rho)
    C_even, SOV_even = measure_SO_for_rho(SO_h_even, SO_b_even, SO_t_even,IN,rho)
    println("At T_final = ", init_t+dt*tsmax,"="^30)
    println("complex SO_odd= ", C_odd, "  SO_odd= ", SOV_odd)
    println("complex SO_even= ", C_even, "  SO_even= ", SOV_even)
    return nothing
end

function test_tr_rho(SO_mpo,IN,rho)
    rho_after= apply(SO_mpo,rho)
    ans = -inner(IN,rho_after)/inner(IN, rho)
    return ans
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
    I1    = args["I1"]
    I2    = args["I2"]
    IR    = args["IR"]
    ID    = args["ID"]
    J     = args["J"]
    initD = args["initD"]
    Dstep = args["Dstep"]
    Dload = args["Dload"]
    Dstepload = args["Dstepload"]
    U     = args["U"]
    dt    = args["dt"]
    tsmax = args["tsmax"]
    init_t= 0.0
    if load && loadsl     # if load time slice, init t will be adjusted to loadt, or else init t = 0
        init_t = loadt
    end
    mps_path  = generate_mps_path(N, t1, t2, tR, tD, J, U, Dmax, Dstep)
    load_path = generate_mps_path(N, t1, t2, tR, tD, J, U, Dload, Dstepload)
    # slice_path= generate_slice_path(t, N, t1, t2, tR, tD, J, U, Dmax, Dstep)
    slice_load_path= generate_slice_path(loadt, N, t1, t2, tR, tD, J, U, I1, I2, IR, ID, Dload)
    # slice_path
    # sites, psi0  = create_psi0_for_dmrg(N, load, load_path)
    t_load = time()
    sites, psi0 ,rho0, HS  = get_sites_rho0_HS(N, t1, t2, tR, tD, J, U,load,loadsl,loadt, mps_path,load_path,slice_load_path, initD, Dstep, Dmax)
    # os       = system_ham(N, t1, t2, tR, tD, J, U)
    # HS       = MPO(os, sites)
    # rho0     = create_rho0_for_evolution(N, load, loadsl, loadt, HS, mps_path, slice_load_path, psi0, initD, Dstep, Dmax)
    println("Initial density matrix loaded in ", format_hms(time()-t_load), " (hh:mm:ss)")
    t_ops = time()
    idx_st = div(N,4)
    idx_ed = N-div(N,4)
    SO_h_odd, SO_b_odd, SO_t_odd    = create_SO(sites, idx_st, idx_ed, N, "odd")
    SO_h_even, SO_b_even, SO_t_even = create_SO(sites, idx_st, idx_ed, N, "even")

    # ===============testing===============================================
    SO_mpo_odd                      = SO_MPO(sites, SO_h_odd, SO_b_odd, SO_t_odd; maxdim = Dmax)
    SO_mpo_even                     = SO_MPO(sites, SO_h_even, SO_b_even, SO_t_even; maxdim = Dmax)
    SO_0_odd = -inner(psi0',SO_mpo_odd,psi0)
    SO_0_even = -inner(psi0',SO_mpo_even,psi0)
    println("Initial SO_odd = ", SO_0_odd, " SO_even= ", SO_0_even)
    IN = MPO(sites,"Id")
    println("For density matrix method","="^30)
    SO_1_odd = test_tr_rho(SO_mpo_odd,IN,rho0)
    SO_1_even = test_tr_rho(SO_mpo_even,IN,rho0)
    println("Initial SO_odd = ", SO_1_odd, " SO_even= ", SO_1_even)
    # ======================================================================

    K0,K0_dag,K_list,Kdag_list      = create_hopping(sites, HS, dt,N, I1, I2, IR, ID)
    check_maxdim(K0,K_list)
    println("SO / Lindblad operators built in ", format_hms(time()-t_ops), " (hh:mm:ss)")
    Lindblad_evolution(dt,tsmax,init_t,N, t1, t2, tR, tD, J, U, I1, I2, IR, ID, Dmax,rho0,K0,K0_dag,K_list,Kdag_list,SO_h_odd, SO_b_odd, SO_t_odd,SO_h_even, SO_b_even, SO_t_even,IN)
end

if abspath(PROGRAM_FILE) == @__FILE__ # only run this code when directly running, including will not trigger main()
    main()
end