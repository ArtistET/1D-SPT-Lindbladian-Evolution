using Test

isdefined(Main, :trajectory_step) || include("AKLT_evolution.jl")

@testset "quantum trajectory building blocks" begin
    N = 4
    dt = 1e-5
    sites = create_sites(N)
    # Fixed Nf and Sz sector, with a nonzero odd string-order value.
    state = ["Up", "Up", "Up", "Dn", "Dn", "Dn", "Up", "Dn"]
    psi = productMPS(sites, state)
    normalize!(psi)

    hamiltonian = system_ham(N, 0.1, 0.2, 1.0, 0.98, 0.0, 10.0)
    channels = create_jump_channels(N, 0.1, 0.1, 0.1, 0.1)
    @test length(channels) == 16N

    predicted = jump_probabilities(psi, dt, channels)
    for i in eachindex(channels)
        if iszero(predicted[i])
            @test predicted[i] == 0
            continue
        end
        jump_state = apply(create_jump_operator(sites, dt, channels[i]), psi; cutoff=1e-12, maxdim=40)
        actual = real(inner(jump_state, jump_state))
        @test actual ≈ predicted[i] rtol=1e-9 atol=1e-12
    end

    K0 = create_nojump_operator(sites, hamiltonian, dt, channels)
    nojump_state = apply(K0, psi; cutoff=1e-12, maxdim=40)
    reference_k0_sum = (-1im * dt) * hamiltonian + (1.0, "Id", 1)
    for channel in channels
        reference_k0_sum += -0.5 * dt * channel.rate^2,
            channel.create_op, channel.source,
            channel.destroy_op, channel.target,
            channel.create_op, channel.target,
            channel.destroy_op, channel.source
    end
    K0_reference = MPO(reference_k0_sum, sites)
    reference_state = apply(K0_reference, psi; cutoff=1e-12, maxdim=40)
    state_difference2 = real(inner(nojump_state, nojump_state) + inner(reference_state, reference_state) -
        2 * inner(nojump_state, reference_state))
    # This subtracts three O(1) contractions, so use a tolerance above
    # Float64 cancellation noise rather than interpreting it as a state error.
    @test abs(state_difference2) < 1e-10
    total_weight = real(inner(nojump_state, nojump_state)) + sum(predicted)
    @test total_weight ≈ 1.0 atol=1e-6

    next_psi, selected, reported_jump_probability, _ = trajectory_step(
        psi, MersenneTwister(1), K0, channels, sites, dt; cutoff=1e-12, maxdim=40)
    @test selected in 0:length(channels)
    @test reported_jump_probability ≈ sum(predicted) rtol=1e-10
    @test real(inner(next_psi, next_psi)) ≈ 1.0 atol=1e-10

    idx_st = div(N, 4)
    idx_ed = N - div(N, 4)
    SO_h, SO_b, SO_t = create_SO(sites, idx_st, idx_ed, N, "odd")
    SO = SO_MPO(sites, SO_h, SO_b, SO_t; cutoff=1e-12, maxdim=40)
    direct_mps, _ = measure_string_orders(psi, SO, SO)
    applied_mps, _ = measure(SO_h, SO_b, SO_t, psi; cutoff=1e-12, maxdim=40)
    @test direct_mps ≈ applied_mps rtol=1e-10 atol=1e-12

    rho = outer(psi', psi; cutoff=1e-12, maxdim=40)
    identity_mpo = MPO(sites, "Id")
    applied_mpdo = -inner(identity_mpo, apply(SO, rho; cutoff=1e-12, maxdim=40)) / inner(identity_mpo, rho)
    direct_mpdo_left = -inner(rho, SO) / inner(rho, identity_mpo)
    direct_mpdo_right = -inner(SO, rho) / inner(identity_mpo, rho)
    @info "MPO–MPO measurement comparison" applied_mpdo direct_mpdo_left direct_mpdo_right
    @test applied_mpdo ≈ direct_mps rtol=1e-9 atol=1e-11
    @test direct_mpdo_left ≈ applied_mpdo rtol=1e-9 atol=1e-11
    @test direct_mpdo_right ≈ applied_mpdo rtol=1e-9 atol=1e-11
end
