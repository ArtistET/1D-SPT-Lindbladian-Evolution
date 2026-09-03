using ITensors, ITensorMPS
using LinearAlgebra
using Statistics
using Random
using JLD2
using ArgParse
import Base.Filesystem.mkpath

include("AKLT_GS.jl")

struct JumpChannel
    target::Int
    source::Int
    create_op::String
    destroy_op::String
    number_op::String
    rate::Float64
    label::String
end

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--load"
            help = "load initial ground state or not"
            default = true
            arg_type = Bool
        "--loadsl"
            help = "resume trajectory checkpoints or not"
            default = false
            arg_type = Bool
        "--loadt"
            help = "time of the trajectory checkpoints loaded"
            default = 0.0
            arg_type = Float64
        "-N"
            help = "half of the system size"
            required = true
            arg_type = Int
        "--Dmax"
            help = "maximum MPS bond dimension during trajectory evolution"
            default = 400
            arg_type = Int
        "--Dstep"
            help = "step of increasing maxdim in DMRG"
            default = 20
            arg_type = Int
        "--t1"
            help = "in-leg hopping t1"
            default = 0.1
            arg_type = Float64
        "--t2"
            help = "in-leg hopping t2"
            default = 0.2
            arg_type = Float64
        "--tR"
            help = "rung hopping tR"
            default = 1.0
            arg_type = Float64
        "--tD"
            help = "diagonal hopping tD"
            required = true
            arg_type = Float64
        "-J"
            help = "spin coupling J"
            default = 0.0
            arg_type = Float64
        "--I1"
            help = "intensity of in-leg1 jump operators"
            default = 0.1
            arg_type = Float64
        "--I2"
            help = "intensity of in-leg2 jump operators"
            default = 0.1
            arg_type = Float64
        "--IR"
            help = "intensity of rung jump operators"
            default = 0.1
            arg_type = Float64
        "--ID"
            help = "intensity of diagonal jump operators"
            default = 0.1
            arg_type = Float64
        "--initD"
            help = "initial DMRG bond dimension"
            default = 10
            arg_type = Int
        "--Dload"
            help = "bond dimension of loaded ground state or trajectory checkpoint"
            default = 100
            arg_type = Int
        "--Dstepload"
            help = "Dstep of loaded ground state"
            default = 20
            arg_type = Int
        "-U"
            help = "repulsive interaction"
            required = true
            arg_type = Float64
        "--dt"
            help = "time step"
            default = 0.01
            arg_type = Float64
        "--tsmax"
            help = "number of time steps"
            default = 20
            arg_type = Int
        "--ntraj"
            help = "number of trajectories handled sequentially by this process"
            default = 1
            arg_type = Int
        "--traj-start"
            help = "global id of the first trajectory"
            default = 1
            arg_type = Int
        "--seed"
            help = "base random seed"
            default = 1234
            arg_type = Int
        "--cutoff"
            help = "MPS truncation cutoff"
            default = 1e-8
            arg_type = Float64
        "--save-traj"
            help = "save final MPS of every trajectory for continuation"
            default = false
            arg_type = Bool
    end
    return parse_args(s)
end

function format_hms(sec)
    ms = round(Int, (sec - floor(sec)) * 1000)
    isec = floor(Int, sec)
    h = div(isec, 3600)
    m = div(isec % 3600, 60)
    s = isec % 60
    return string(lpad(h, 2, '0'), ":", lpad(m, 2, '0'), ":", lpad(s, 2, '0'), ".", lpad(ms, 3, '0'))
end

time_label(t) = string(round(t; digits=12))

function add_bond_channels!(channels, a, b, rate, label)
    for (spin, create_op, destroy_op, number_op) in
        (("up", "Cdagup", "Cup", "Nup"), ("dn", "Cdagdn", "Cdn", "Ndn"))
        push!(channels, JumpChannel(a, b, create_op, destroy_op, number_op, rate, "$(label)_$(spin)_$(b)to$(a)"))
        push!(channels, JumpChannel(b, a, create_op, destroy_op, number_op, rate, "$(label)_$(spin)_$(a)to$(b)"))
    end
    return channels
end

function create_jump_channels(N, I1, I2, IR, ID)
    channels = JumpChannel[]
    intensities = (I1, I2)
    for j in 1:N
        next_j = j % N + 1
        for alpha in 1:2
            add_bond_channels!(channels, lpos(j, alpha), lpos(next_j, alpha), intensities[alpha], "leg$(alpha)_$(j)")
        end
        add_bond_channels!(channels, lpos(j, 1), lpos(j, 2), IR, "rung_$(j)")
        add_bond_channels!(channels, lpos(j, 2), lpos(next_j, 1), ID, "diag_$(j)")
    end
    @assert length(channels) == 16N
    return channels
end

function create_nojump_operator(sites, hamiltonian, dt, channels; energy_shift=0.0)
    # Construct K0 as one OpSum. Adding already-built MPOs triggers an expensive
    # generic MPO decomposition, and the paired directions obey exactly
    # L†_ab L_ab + L†_ba L_ba = rate²*(n_a+n_b-2*n_a*n_b).
    # A real scalar shift changes only the exact no-jump state's global phase.
    # Removing the extensive ground-state energy greatly reduces Euler error.
    os = (-1im * dt) * hamiltonian + (1.0 + 1im * dt * energy_shift, "Id", 1)
    @assert length(channels) % 4 == 0
    for first_channel in 1:4:length(channels)
        up_forward, up_reverse, dn_forward, dn_reverse = channels[first_channel:(first_channel + 3)]
        a, b = up_forward.target, up_forward.source
        @assert (up_reverse.target, up_reverse.source) == (b, a)
        @assert (dn_forward.target, dn_forward.source) == (a, b)
        @assert (dn_reverse.target, dn_reverse.source) == (b, a)
        @assert up_forward.number_op == up_reverse.number_op == "Nup"
        @assert dn_forward.number_op == dn_reverse.number_op == "Ndn"
        @assert all(channel.rate == up_forward.rate for channel in (up_reverse, dn_forward, dn_reverse))
        for spin_offset in (0, 2)
            channel = channels[first_channel + spin_offset]
            coefficient = dt * channel.rate^2
            os += -0.5 * coefficient, channel.number_op, a
            os += -0.5 * coefficient, channel.number_op, b
            os += coefficient, channel.number_op, a, channel.number_op, b
        end
    end
    return MPO(os, sites)
end

function create_jump_operator(sites, dt, channel)
    os = OpSum()
    os += sqrt(dt) * channel.rate,
        channel.create_op, channel.target,
        channel.destroy_op, channel.source
    return MPO(os, sites)
end

function jump_probabilities(psi, dt, channels; negative_tolerance=1e-9)
    # For L = rate*c†_target*c_source,
    # ‖sqrt(dt)Lψ‖² = dt*rate²*<n_source*(1-n_target)>.
    # Two density correlation matrices therefore replace 16N MPO applications.
    correlations = Dict(
        "Nup" => correlation_matrix(psi, "Nup", "Nup"),
        "Ndn" => correlation_matrix(psi, "Ndn", "Ndn"),
    )
    probabilities = Vector{Float64}(undef, length(channels))
    most_negative = 0.0
    for (i, channel) in pairs(channels)
        corr = correlations[channel.number_op]
        occupation = real(corr[channel.source, channel.source])
        joint_occupation = real(corr[channel.source, channel.target])
        probability = dt * channel.rate^2 * (occupation - joint_occupation)
        most_negative = min(most_negative, probability)
        probabilities[i] = max(probability, 0.0)
    end
    most_negative < -negative_tolerance && @warn "Negative jump probability before clipping" most_negative
    return probabilities
end

function trajectory_step(psi, rng, K0, channels, sites, dt; cutoff, maxdim)
    jump_weights = jump_probabilities(psi, dt, channels)
    total_jump_probability = sum(jump_weights)
    isfinite(total_jump_probability) && total_jump_probability >= 0 ||
        error("Invalid total jump probability: $total_jump_probability")
    total_jump_probability <= 1 ||
        error("Total jump probability is $total_jump_probability > 1; reduce dt")

    draw = rand(rng)
    if draw >= total_jump_probability
        nojump_state = apply(K0, psi; cutoff=cutoff, maxdim=maxdim)
        nojump_weight = real(inner(nojump_state, nojump_state))
        nojump_weight > 0 || error("No-jump branch has zero norm")
        normalize!(nojump_state)
        return nojump_state, 0, total_jump_probability, nojump_weight
    end

    cumulative = 0.0
    selected = 0
    for i in eachindex(channels)
        cumulative += jump_weights[i]
        if draw < cumulative
            selected = i
            break
        end
    end
    selected != 0 || error("Failed to select a jump channel: draw=$draw cumulative=$cumulative")

    jump_state = apply(create_jump_operator(sites, dt, channels[selected]), psi; cutoff=cutoff, maxdim=maxdim)
    actual_weight = real(inner(jump_state, jump_state))
    actual_weight > 0 || error("Selected zero-norm jump channel $(channels[selected].label)")
    if !isapprox(actual_weight, jump_weights[selected]; rtol=1e-5, atol=1e-12)
        @warn "Selected jump weight changed by MPS truncation" channel=channels[selected].label predicted=jump_weights[selected] actual=actual_weight
    end
    normalize!(jump_state)
    return jump_state, selected, total_jump_probability, actual_weight
end

function SO_MPO(sites, SO_head, SO_body, SO_tail; cutoff=1e-10, maxdim=400)
    so_mpo = MPO(sites, "Id")
    so_mpo = apply(SO_head, so_mpo; cutoff=cutoff, maxdim=maxdim)
    so_mpo = apply(SO_body, so_mpo; cutoff=cutoff, maxdim=maxdim)
    so_mpo = apply(SO_tail, so_mpo; cutoff=cutoff, maxdim=maxdim)
    return so_mpo
end

function measure_string_orders(psi, SO_odd, SO_even)
    C_odd = -inner(psi', SO_odd, psi)
    C_even = -inner(psi', SO_even, psi)
    return C_odd, C_even
end

function trajectory_base_path(N, t1, t2, tR, tD, J, U, I1, I2, IR, ID, Dmax, dt, seed)
    return "./trajectory_evolution/N$(N)_t($(t1),$(t2))_tR$(tR)_tD$(tD)_J$(J)_U$(U)_I1$(I1)_I2$(I2)_IR$(IR)_ID$(ID)/Dmax$(Dmax)_dt$(dt)_seed$(seed)"
end

function checkpoint_path(t, trajectory_id, N, t1, t2, tR, tD, J, U, I1, I2, IR, ID, Dmax, dt, seed; create=false)
    base = trajectory_base_path(N, t1, t2, tR, tD, J, U, I1, I2, IR, ID, Dmax, dt, seed)
    directory = joinpath(base, "checkpoints", "T$(time_label(t))")
    create && mkpath(directory)
    return joinpath(directory, "trajectory_$(trajectory_id).jld2")
end

function result_path(init_t, final_t, traj_start, traj_stop, N, t1, t2, tR, tD, J, U, I1, I2, IR, ID, Dmax, dt, seed)
    base = trajectory_base_path(N, t1, t2, tR, tD, J, U, I1, I2, IR, ID, Dmax, dt, seed)
    directory = joinpath(base, "results")
    mkpath(directory)
    return joinpath(directory, "T$(time_label(init_t))_to_T$(time_label(final_t))_traj$(traj_start)-$(traj_stop).jld2")
end

function save_trajectory(psi, path)
    @save path psi
end

function load_trajectory(path)
    isfile(path) || error("Trajectory checkpoint does not exist: $path")
    @load path psi
    return psi
end

function trajectory_rng(seed, trajectory_id, completed_steps)
    rng = MersenneTwister(seed + trajectory_id - 1)
    for _ in 1:completed_steps
        rand(rng)
    end
    return rng
end

function validate_args(args)
    args["N"] >= 4 || error("N must be at least 4 for the present string-order interval")
    args["Dmax"] > 0 || error("Dmax must be positive")
    args["dt"] > 0 || error("dt must be positive")
    args["tsmax"] >= 0 || error("tsmax must be nonnegative")
    args["ntraj"] > 0 || error("ntraj must be positive")
    args["traj-start"] > 0 || error("traj-start must be positive")
    args["cutoff"] >= 0 || error("cutoff must be nonnegative")
    args["loadsl"] && !args["load"] && error("loadsl=true requires load=true")
    return nothing
end

function prepare_system(args)
    N = args["N"]
    load = args["load"]
    loadsl = args["loadsl"]
    loadt = args["loadt"]
    Dmax = args["Dmax"]
    Dstep = args["Dstep"]
    Dload = args["Dload"]
    Dstepload = args["Dstepload"]
    t1, t2, tR, tD = args["t1"], args["t2"], args["tR"], args["tD"]
    J, U = args["J"], args["U"]

    mps_path = generate_mps_path(N, t1, t2, tR, tD, J, U, Dmax, Dstep)
    load_path = generate_mps_path(N, t1, t2, tR, tD, J, U, Dload, Dstepload)

    if loadsl
        trajectory_id = args["traj-start"]
        path = checkpoint_path(loadt, trajectory_id, N, t1, t2, tR, tD, J, U,
            args["I1"], args["I2"], args["IR"], args["ID"], Dload, args["dt"], args["seed"])
        psi_initial = load_trajectory(path)
        sites = siteinds(psi_initial)
    else
        sites, psi_initial = create_psi0_for_dmrg(N, load, load_path)
    end

    HS = MPO(system_ham(N, t1, t2, tR, tD, J, U), sites)
    if !load
        _, psi_initial = dmrg_GS(N, HS, mps_path, psi_initial, args["initD"], Dstep, Dmax)
    end
    normalize!(psi_initial)
    if load && !hasqns(first(siteinds(psi_initial)))
        @warn "The loaded state has no QN blocks. Rerun with load=false to obtain the memory benefit of conserve_qns=true."
    end
    return sites, psi_initial, HS
end

function save_results(path, args, times, C_odd_samples, C_even_samples, jump_indices,
    total_jump_probabilities, branch_weights, bond_dimensions, measurement_seconds, evolution_seconds,
    checkpoint_seconds, completed_trajectories; final=false)
    if final
        SO_odd_mean = vec(mean(real.(C_odd_samples); dims=1))
        SO_even_mean = vec(mean(real.(C_even_samples); dims=1))
        if size(C_odd_samples, 1) > 1
            SO_odd_stderr = vec(std(real.(C_odd_samples); dims=1, corrected=true)) ./ sqrt(size(C_odd_samples, 1))
            SO_even_stderr = vec(std(real.(C_even_samples); dims=1, corrected=true)) ./ sqrt(size(C_even_samples, 1))
        else
            SO_odd_stderr = fill(NaN, length(times))
            SO_even_stderr = fill(NaN, length(times))
        end
        @save path args times C_odd_samples C_even_samples SO_odd_mean SO_even_mean SO_odd_stderr SO_even_stderr jump_indices total_jump_probabilities branch_weights bond_dimensions measurement_seconds evolution_seconds checkpoint_seconds completed_trajectories
    else
        @save path args times C_odd_samples C_even_samples jump_indices total_jump_probabilities branch_weights bond_dimensions measurement_seconds evolution_seconds checkpoint_seconds completed_trajectories
    end
end

function run_trajectories(args, sites, psi_initial, HS)
    N = args["N"]
    dt = args["dt"]
    tsmax = args["tsmax"]
    init_t = args["load"] && args["loadsl"] ? args["loadt"] : 0.0
    final_t = init_t + dt * tsmax
    Dmax = args["Dmax"]
    cutoff = args["cutoff"]
    ntraj = args["ntraj"]
    traj_start = args["traj-start"]
    traj_stop = traj_start + ntraj - 1
    seed = args["seed"]
    t1, t2, tR, tD = args["t1"], args["t2"], args["tR"], args["tD"]
    J, U = args["J"], args["U"]
    I1, I2, IR, ID = args["I1"], args["I2"], args["IR"], args["ID"]

    idx_st = div(N, 4)
    idx_ed = N - div(N, 4)
    SO_h_odd, SO_b_odd, SO_t_odd = create_SO(sites, idx_st, idx_ed, N, "odd")
    SO_h_even, SO_b_even, SO_t_even = create_SO(sites, idx_st, idx_ed, N, "even")
    SO_odd = SO_MPO(sites, SO_h_odd, SO_b_odd, SO_t_odd; cutoff=cutoff, maxdim=Dmax)
    SO_even = SO_MPO(sites, SO_h_even, SO_b_even, SO_t_even; cutoff=cutoff, maxdim=Dmax)

    direct_odd, direct_even = measure_string_orders(psi_initial, SO_odd, SO_even)
    apply_odd, _ = measure(SO_h_odd, SO_b_odd, SO_t_odd, psi_initial; cutoff=cutoff, maxdim=Dmax)
    apply_even, _ = measure(SO_h_even, SO_b_even, SO_t_even, psi_initial; cutoff=cutoff, maxdim=Dmax)
    println("Initial SO consistency: odd direct=", direct_odd, " apply=", apply_odd,
        " |diff|=", abs(direct_odd - apply_odd))
    println("Initial SO consistency: even direct=", direct_even, " apply=", apply_even,
        " |diff|=", abs(direct_even - apply_even))
    isapprox(direct_odd, apply_odd; rtol=1e-6, atol=1e-9) || error("Odd string-order constructions disagree")
    isapprox(direct_even, apply_even; rtol=1e-6, atol=1e-9) || error("Even string-order constructions disagree")

    channels = create_jump_channels(N, I1, I2, IR, ID)
    energy_shift = real(inner(psi_initial', HS, psi_initial))
    K0 = create_nojump_operator(sites, system_ham(N, t1, t2, tR, tD, J, U), dt, channels;
        energy_shift=energy_shift)
    println("Built ", length(channels), " jump channels as metadata; maxlinkdim(K0)=", maxlinkdim(K0),
        "; no-jump energy shift=", energy_shift)

    times = collect(range(init_t; step=dt, length=tsmax + 1))
    C_odd_samples = fill(ComplexF64(NaN, NaN), ntraj, tsmax + 1)
    C_even_samples = fill(ComplexF64(NaN, NaN), ntraj, tsmax + 1)
    jump_indices = zeros(Int, ntraj, tsmax)
    total_jump_probabilities = fill(NaN, ntraj, tsmax)
    branch_weights = fill(NaN, ntraj, tsmax)
    bond_dimensions = zeros(Int, ntraj, tsmax + 1)
    measurement_seconds = zeros(ntraj, tsmax + 1)
    evolution_seconds = zeros(ntraj, tsmax)
    checkpoint_seconds = zeros(ntraj)
    output_path = result_path(init_t, final_t, traj_start, traj_stop, N, t1, t2, tR, tD, J, U, I1, I2, IR, ID, Dmax, dt, seed)
    completed_trajectories = 0

    completed_steps = round(Int, init_t / dt)
    isapprox(completed_steps * dt, init_t; rtol=0, atol=1e-10) || error("loadt must be an integer multiple of dt")

    for local_id in 1:ntraj
        trajectory_id = traj_start + local_id - 1
        trajectory_wall = time()
        if args["load"] && args["loadsl"]
            if local_id == 1
                psi = psi_initial
                psi_initial = nothing
            else
                load_path = checkpoint_path(init_t, trajectory_id, N, t1, t2, tR, tD, J, U, I1, I2, IR, ID, args["Dload"], dt, seed)
                psi = load_trajectory(load_path)
                all(siteinds(psi) .== sites) || error("Site indices differ in trajectory checkpoint $load_path")
                normalize!(psi)
            end
        else
            psi = copy(psi_initial)
        end
        rng = trajectory_rng(seed, trajectory_id, completed_steps)
        println("Trajectory ", trajectory_id, " started; maxlinkdim=", maxlinkdim(psi))

        for time_index in 1:(tsmax + 1)
            measurement_start = time()
            C_odd_samples[local_id, time_index], C_even_samples[local_id, time_index] =
                measure_string_orders(psi, SO_odd, SO_even)
            measurement_seconds[local_id, time_index] = time() - measurement_start
            bond_dimensions[local_id, time_index] = maxlinkdim(psi)
            println("Trajectory ", trajectory_id, " T=", time_label(times[time_index]),
                " SO_odd=", real(C_odd_samples[local_id, time_index]),
                " SO_even=", real(C_even_samples[local_id, time_index]),
                " maxlinkdim=", bond_dimensions[local_id, time_index],
                " measurement=", format_hms(measurement_seconds[local_id, time_index]))

            time_index > tsmax && break
            evolution_start = time()
            psi, jump_index, total_jump_probability, branch_weight = trajectory_step(
                psi, rng, K0, channels, sites, dt; cutoff=cutoff, maxdim=Dmax)
            evolution_seconds[local_id, time_index] = time() - evolution_start
            jump_indices[local_id, time_index] = jump_index
            total_jump_probabilities[local_id, time_index] = total_jump_probability
            branch_weights[local_id, time_index] = branch_weight
            event = jump_index == 0 ? "no jump" : channels[jump_index].label
            println("Trajectory ", trajectory_id, " step ", time_index, "/", tsmax,
                " event=", event, " jump_probability=", total_jump_probability,
                " branch_norm=", branch_weight,
                " maxlinkdim=", maxlinkdim(psi),
                " evolution=", format_hms(evolution_seconds[local_id, time_index]))
        end

        if args["save-traj"]
            checkpoint_start = time()
            path = checkpoint_path(final_t, trajectory_id, N, t1, t2, tR, tD, J, U, I1, I2, IR, ID, Dmax, dt, seed; create=true)
            save_trajectory(psi, path)
            checkpoint_seconds[local_id] = time() - checkpoint_start
            println("Trajectory ", trajectory_id, " checkpoint saved in ", format_hms(checkpoint_seconds[local_id]))
        end
        completed_trajectories = local_id
        save_results(output_path, args, times, C_odd_samples, C_even_samples, jump_indices,
            total_jump_probabilities, branch_weights, bond_dimensions, measurement_seconds, evolution_seconds,
            checkpoint_seconds, completed_trajectories)
        println("Trajectory ", trajectory_id, " finished in ", format_hms(time() - trajectory_wall))
        GC.gc()
    end

    save_start = time()
    save_results(output_path, args, times, C_odd_samples, C_even_samples, jump_indices,
        total_jump_probabilities, branch_weights, bond_dimensions, measurement_seconds, evolution_seconds,
        checkpoint_seconds, completed_trajectories; final=true)
    println("Final ensemble result saved to ", output_path, " in ", format_hms(time() - save_start))
    return output_path
end

function main()
    args = parse_commandline()
    validate_args(args)
    @show args
    println("Julia threads=", Threads.nthreads(), " BLAS threads=", BLAS.get_num_threads())

    load_start = time()
    sites, psi_initial, HS = prepare_system(args)
    println("Initial MPS/system prepared in ", format_hms(time() - load_start),
        "; QN conserving=", hasqns(first(siteinds(psi_initial))), "; maxlinkdim=", maxlinkdim(psi_initial))
    run_trajectories(args, sites, psi_initial, HS)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
