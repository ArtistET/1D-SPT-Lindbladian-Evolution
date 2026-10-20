using ITensors, ITensorMPS, LinearAlgebra, JLD2
include("AKLT_GS.jl")

length(ARGS) == 4 || error("usage: julia AKLT_GS_U0_QN.jl tD Dmax Dload Dstepload (Dload=0 starts anew)")
tD = parse(Float64, ARGS[1])
Dmax = parse(Int, ARGS[2])
Dload = parse(Int, ARGS[3])
Dstepload = parse(Int, ARGS[4])
Dmax > Dload >= 0 || error("need Dmax > Dload >= 0")

state = generate_mps_path(10, 0.1, 0.2, 1.0, tD, 0.0, 0.0, Dmax, 100)
marker = replace(state, ".jld2" => ".complete")
isfile(marker) && error("state already marked complete: $state")
if isfile(state)
    psi = load_mps(state)
    starting_dim = Dmax
elseif Dload > 0
    psi = load_mps(generate_mps_path(10, 0.1, 0.2, 1.0, tD, 0.0, 0.0, Dload, Dstepload))
    starting_dim = Dload
else
    _, psi = create_psi0_for_dmrg(10, false, "")
    starting_dim = 0
end
hasqns(siteind(psi, 1)) || error("initial state is not QN conserving")
normalize!(psi)
sites = siteinds(psi)
H = MPO(system_ham(10, 0.1, 0.2, 1.0, tD, 0.0, 0.0), sites)

# At U=0 the Hamiltonian is quadratic. Its half-filled exact energy is an
# independent check on the DMRG approximation, not a DMRG stopping shortcut.
h = zeros(Float64, 20, 20)
for j in 1:10, alpha in 1:2
    idx = lpos(j, alpha)
    idx_a = lpos(j % 10 + 1, alpha)
    hop = (0.1, 0.2)[alpha]
    h[idx, idx_a] -= hop
    h[idx_a, idx] -= hop
    h[idx, idx + 3 - 2alpha] -= 1.0
    idx_d = idx == 1 ? 20 : (idx - 3 + 2alpha) % 20
    h[idx, idx_d] -= tD
end
isapprox(h, h'; atol=1e-12) || error("single-particle Hamiltonian is not Hermitian")
exact_energy = 2sum(eigvals(Symmetric(h))[1:10])
println("exact U=0 energy = $exact_energy")

previous_energy = Inf
stable = 0
for sweep in 1:40
    global psi, stable, previous_energy
    maxdim = min(max(100, starting_dim + 100sweep), Dmax)
    noise = sweep <= 4 ? 1e-6 : sweep <= 7 ? 1e-8 : 0.0
    energy, psi = dmrg(H, psi; nsweeps=1, maxdim=maxdim, cutoff=0.0, noise=noise)
    delta = abs(energy - previous_energy)
    println("sweep=$sweep maxdim=$maxdim noise=$noise energy=$energy exact_gap=$(energy-exact_energy) delta=$delta")
    save_checkpoint(psi, state)
    if maxdim == Dmax && noise == 0.0 && delta < 1e-9
        stable += 1
        stable >= 2 && break
    else
        stable = 0
    end
    previous_energy = energy
end
stable >= 2 || error("DMRG did not stabilize within 40 sweeps; checkpoint retained")
normalize!(psi)
orthogonalize!(psi, 10)
save_checkpoint(psi, state)
println("final norm=$(norm(psi)) maxlinkdim=$(maxlinkdim(psi))")
println("final energy=$(real(inner(psi, H, psi))) exact_gap=$(real(inner(psi, H, psi))-exact_energy)")
