# Exact density-matrix cross-check of the trajectory generator on a 3-rung,
# one-up/one-down sector (36 states). No MPS truncation enters the reference.
using Test, LinearAlgebra, Printf
include("AKLT_evolution.jl")

N = 3
nsites = 2N
sites = create_sites(N)
labels = [(u, d) for u in 1:nsites for d in 1:nsites]
index = Dict(label => i for (i, label) in enumerate(labels))
basis = [productMPS(sites, [s == u ? (s == d ? "UpDn" : "Up") :
    (s == d ? "Dn" : "Emp") for s in 1:nsites]) for (u, d) in labels]
dim = length(basis)
H_mpo = MPO(system_ham(N, 0.1, 0.2, 1.0, 0.98, 0.0, 10.0), sites)
H = [inner(basis[j]', H_mpo, basis[i]) for j in 1:dim, i in 1:dim]
channels = create_jump_channels(N, 0.1, 0.1, 0.1, 0.1)
Ls = Matrix{ComplexF64}[]
for ch in channels
    L = zeros(ComplexF64, dim, dim)
    op = create_jump_operator(sites, 1.0, ch)
    for (i, (u, d)) in enumerate(labels)
        spin_up = ch.number_op == "Nup"
        occupied = spin_up ? u : d
        occupied == ch.source && occupied != ch.target || continue
        j = spin_up ? index[(ch.target, d)] : index[(u, ch.target)]
        L[j, i] = inner(basis[j]', op, basis[i])
    end
    push!(Ls, L)
end
A = sum(L' * L for L in Ls)

sz(s, u, d) = ((s == u) - (s == d)) / 2
function string_value(u, d, parity)
    head, tail, body = parity == :odd ? ((1, 2), (5, 6), (3, 4)) :
        ((2, 3), (6, 1), (5, 4))
    return -(sz(head[1], u, d) + sz(head[2], u, d)) *
        (sz(tail[1], u, d) + sz(tail[2], u, d)) *
        exp(1im * pi * (sz(body[1], u, d) + sz(body[2], u, d)))
end
SO = Dict(p => Diagonal(ComplexF64[string_value(u, d, p) for (u, d) in labels])
    for p in (:odd, :even))

function rhs(rho)
    out = -1im * (H * rho - rho * H) - 0.5 * (A * rho + rho * A)
    for L in Ls
        out += L * rho * L'
    end
    return out
end

function rk4(rho, step, nsteps)
    for _ in 1:nsteps
        k1 = rhs(rho)
        k2 = rhs(rho + step / 2 * k1)
        k3 = rhs(rho + step / 2 * k2)
        k4 = rhs(rho + step * k3)
        rho += step / 6 * (k1 + 2k2 + 2k3 + k4)
    end
    return rho
end

function branch_map(rho, dt, nsteps)
    K = exp(-1im * dt * (H - 0.5im * A))
    for _ in 1:nsteps
        next = K * rho * K'
        for L in Ls
            next += dt * (L * rho * L')
        end
        rho = next / real(tr(next))
    end
    return rho
end

function expectation(rho, p)
    return real(tr(SO[p] * rho))
end

@testset "small-sector exact Lindblad and TDVP branches" begin
    @test dim == 36
    @test norm(H - H') < 1e-11
    @test norm(A - A') < 1e-11
    @test norm(rhs(Matrix{ComplexF64}(I, dim, dim))) < 1e-11

    # Independent diagonal formula must match the project's string MPO.
    for p in (:odd, :even)
        head, body, tail = create_SO(sites, 1, 3, N, String(p))
        mpo = SO_MPO(sites, head, body, tail; cutoff=1e-13, maxdim=64)
        for (i, psi) in enumerate(basis)
            @test -inner(psi', mpo, psi) ≈ SO[p][i, i] atol=1e-10
        end
    end

    initial = index[(1, 5)]
    psi = basis[initial]
    dt = 0.025
    predicted = jump_probabilities(psi, dt, channels)
    exact = [dt * real((L' * L)[initial, initial]) for L in Ls]
    @test maximum(abs.(predicted .- exact)) < 1e-12
    E = real(H[initial, initial])
    Heff = create_effective_hamiltonian(sites,
        system_ham(N, 0.1, 0.2, 1.0, 0.98, 0.0, 10.0), channels, E)
    heff_action = [inner(b', Heff, psi) for b in basis]
    dense_action = (H - E * I - 0.5im * A)[:, initial]
    @printf("Heff generator column norm_error=%.3g\n", norm(heff_action - dense_action))
    @test norm(heff_action - dense_action) < 1e-10
    tdvp_psi = tdvp(Heff, -1im * dt, psi; nsite=2, maxdim=64,
        cutoff=1e-13, normalize=false,
        updater_kwargs=(; ishermitian=false, tol=1e-10, krylovdim=15, maxiter=30, eager=true))
    dense_psi = exp(-1im * dt * (H - E * I - 0.5im * A))[:, initial]
    mps_psi = [inner(b, tdvp_psi) for b in basis]
    fidelity_loss = 1 - abs(dot(mps_psi, dense_psi))^2 /
        (real(dot(mps_psi, mps_psi)) * real(dot(dense_psi, dense_psi)))
    @printf("no-jump one-step vector_error=%.3g infidelity=%.3g\n",
        norm(mps_psi - dense_psi), fidelity_loss)
    @test fidelity_loss < 1e-5

    # Force a nonzero jump and compare the post-jump no-jump propagation.
    selected = findfirst(>(0.0), exact)
    post = apply(create_jump_operator(sites, dt, channels[selected]), psi;
        cutoff=1e-13, maxdim=64)
    normalize!(post)
    exact_post = Ls[selected][:, initial]
    exact_post /= norm(exact_post)
    post_mps = [inner(b, post) for b in basis]
    @test abs(abs(dot(post_mps, exact_post)) - 1) < 1e-10
    post_tdvp = tdvp(Heff, -1im * dt, post; nsite=2, maxdim=64,
        cutoff=1e-13, normalize=false,
        updater_kwargs=(; ishermitian=false, tol=1e-10, krylovdim=15, maxiter=30, eager=true))
    post_dense = exp(-1im * dt * (H - E * I - 0.5im * A)) * exact_post
    post_vec = [inner(b, post_tdvp) for b in basis]
    @test 1 - abs(dot(post_vec, post_dense))^2 /
        (real(dot(post_vec, post_vec)) * real(dot(post_dense, post_dense))) < 1e-5

    rho0 = zeros(ComplexF64, dim, dim)
    rho0[initial, initial] = 1
    ref = rk4(copy(rho0), 0.001, 200)
    @test abs(real(tr(ref)) - 1) < 1e-10
    for step in (0.025, 0.0125)
        approx = branch_map(copy(rho0), step, round(Int, 0.2 / step))
        for p in (:odd, :even)
            @printf("initial dt=%.4f %s SO_exact=%.9f SO_branch=%.9f abs_error=%.3g\n",
                step, p, expectation(ref, p), expectation(approx, p),
                abs(expectation(ref, p) - expectation(approx, p)))
        end
        rho_jump = exact_post * exact_post'
        ref_jump = rk4(copy(rho_jump), 0.001, 200)
        approx_jump = branch_map(copy(rho_jump), step, round(Int, 0.2 / step))
        for p in (:odd, :even)
            @printf("postjump dt=%.4f %s SO_exact=%.9f SO_branch=%.9f abs_error=%.3g\n",
                step, p, expectation(ref_jump, p), expectation(approx_jump, p),
                abs(expectation(ref_jump, p) - expectation(approx_jump, p)))
        end
    end
end
