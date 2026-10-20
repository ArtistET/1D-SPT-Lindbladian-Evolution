using ITensors, ITensorMPS, JLD2, LinearAlgebra
include("AKLT_GS.jl")

length(ARGS) == 3 || error("usage: julia analyze_u0_qn_ground_states.jl Dmax Dstep OUTPUT.csv")
Dmax, Dstep = parse.(Int, ARGS[1:2])
output = ARGS[3]
isfile(output) && error("refusing to overwrite $output")

function exact_u0_energy(tD)
    n = 10
    h = zeros(Float64, 2n, 2n)
    for j in 1:n, alpha in 1:2
        idx = lpos(j, alpha)
        next_idx = lpos(j % n + 1, alpha)
        hop = (0.1, 0.2)[alpha]
        h[idx, next_idx] -= hop
        h[next_idx, idx] -= hop
        h[idx, idx + 3 - 2alpha] -= 1.0
        diagonal = idx == 1 ? 2n : (idx - 3 + 2alpha) % (2n)
        h[idx, diagonal] -= tD
    end
    isapprox(h, h'; atol=1e-12) || error("single-particle Hamiltonian is not Hermitian")
    levels = eigvals(Symmetric(h))
    return 2sum(levels[1:n]), levels[n+1] - levels[n]
end

data = []
for tD in (0.98, 0.99, 1.0, 1.01, 1.02)
    state = "ground_states/N10_t(0.1,0.2)_tR1.0_tD$(tD)_J0.0_U0.0/Dmax$(Dmax)/Dstep$(Dstep)/AKLT_N10_t(0.1,0.2)_tR1.0_tD$(tD)_J0.0_U0.0_Dmax$(Dmax).jld2"
    isfile(replace(state, ".jld2" => ".complete")) || error("incomplete state: $state")
    psi = load(state, "psi0")
    hasqns(siteind(psi, 1)) || error("state is not QN conserving: $state")
    norm_error = abs(norm(psi) - 1)
    norm_error < 1e-8 || error("state is not normalized: $state")
    sites = siteinds(psi)
    hamiltonian = MPO(system_ham(10, 0.1, 0.2, 1.0, tD, 0.0, 0.0), sites)
    energy = real(inner(psi, hamiltonian, psi))
    exact_energy, fermi_gap = exact_u0_energy(tD)
    energy + 1e-8 >= exact_energy || error("MPS energy below exact ground energy: $state")
    so = Float64[]
    so_imag = Float64[]
    for parity in ("odd", "even")
        head, body, tail = create_SO(sites, 2, 8, 10, parity)
        complex_so, _ = measure(head, body, tail, psi)
        isfinite(real(complex_so)) && isfinite(imag(complex_so)) || error("non-finite SO: $state")
        push!(so, real(complex_so))
        push!(so_imag, imag(complex_so))
    end
    orthogonalize!(psi, 10)
    _, s, _ = svd(psi[10], (linkind(psi, 9), siteind(psi, 10)))
    smallest_singular_value = minimum(abs(s[k, k]) for k in 1:dim(s, 1))
    result = (tD, Dmax, so[1], so[2], so_imag[1], so_imag[2], energy, exact_energy, energy - exact_energy,
              fermi_gap, maxlinkdim(psi), dim(linkind(psi, 10)),
              smallest_singular_value, norm_error, true)
    push!(data, result)
    println(result)
end

open(output, "w") do io
    println(io, "tD,Dmax,odd,even,odd_imag,even_imag,energy,exact_energy,energy_error,fermi_gap,maxlinkdim,Dmid,min_mid_singular_value,norm_error,hasqns")
    for row in data
        println(io, join(row, ","))
    end
end
println("Wrote $output")
