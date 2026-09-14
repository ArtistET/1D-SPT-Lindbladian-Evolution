include("AKLT_evolution.jl")

N, Dload, Dmax, cutoff = 10, 100, 100, 1e-8
t1, t2, tR, tD, J, U = 0.1, 0.2, 1.0, 1.0, 0.0, 10.0
I1 = I2 = IR = ID = 0.1
dt = isempty(ARGS) ? 0.025 : parse(Float64, ARGS[1])

psi = load_mps(generate_mps_path(N, t1, t2, tR, tD, J, U, Dload, 20))
normalize!(psi)
sites = siteinds(psi)
hamiltonian = system_ham(N, t1, t2, tR, tD, J, U)
HS = MPO(hamiltonian, sites)
channels = create_jump_channels(N, I1, I2, IR, ID)
energy_shift = real(inner(psi', HS, psi))
H_eff = create_effective_hamiltonian(sites, hamiltonian, channels, energy_shift)
expected = 1 - sum(jump_probabilities(psi, dt, channels))

for nsite in (1, 2)
    started = time()
    evolved = tdvp(H_eff, -1im * dt, psi;
        nsite=nsite, maxdim=Dmax, cutoff=cutoff, normalize=false,
        updater_kwargs=(; ishermitian=false, tol=1e-8, krylovdim=15, maxiter=30, eager=true))
    actual = real(inner(evolved, evolved))
    println("nsite=", nsite, " seconds=", time() - started,
        " expected=", expected, " actual=", actual,
        " difference=", actual - expected, " maxlinkdim=", maxlinkdim(evolved))
end
