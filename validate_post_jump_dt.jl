include("AKLT_evolution.jl")

N, Dload, cutoff = 10, 100, 1e-8
Dmax = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 100
t1, t2, tR, tD, J, U = 0.1, 0.2, 1.0, 1.0, 0.0, 10.0
I1 = I2 = IR = ID = 0.1

load_path = generate_mps_path(N, t1, t2, tR, tD, J, U, Dload, 20)
psi_initial = load_mps(load_path)
normalize!(psi_initial)
sites = siteinds(psi_initial)
hamiltonian = system_ham(N, t1, t2, tR, tD, J, U)
channels = create_jump_channels(N, I1, I2, IR, ID)
HS = MPO(hamiltonian, sites)
energy_shift = real(inner(psi_initial', HS, psi_initial))
H_eff = create_effective_hamiltonian(sites, hamiltonian, channels, energy_shift)

probe_weights = jump_probabilities(psi_initial, 1.0, channels)
jump_index = findfirst(>(1e-12), probe_weights)
isnothing(jump_index) && error("No nonzero jump channel found")
post_jump = apply(create_jump_operator(sites, 1.0, channels[jump_index]), psi_initial;
    cutoff=cutoff, maxdim=Dmax)
normalize!(post_jump)
println("Forced channel=", channels[jump_index].label,
    " initial_bond_dimension=", maxlinkdim(psi_initial),
    " post_jump_bond_dimension=", maxlinkdim(post_jump),
    " evolution_Dmax=", Dmax)

idx_st, idx_ed = div(N, 4), N - div(N, 4)
odd_parts = create_SO(sites, idx_st, idx_ed, N, "odd")
even_parts = create_SO(sites, idx_st, idx_ed, N, "even")
SO_odd = SO_MPO(sites, odd_parts...; cutoff=cutoff, maxdim=Dmax)
SO_even = SO_MPO(sites, even_parts...; cutoff=cutoff, maxdim=Dmax)

states = MPS[]
dt_values = isempty(ARGS) ? (0.05, 0.025, 0.0125) : (parse(Float64, ARGS[1]),)
for dt in dt_values
    state = copy(post_jump)
    max_survival_error = 0.0
    for _ in 1:round(Int, 0.05 / dt)
        expected_survival = 1 - sum(jump_probabilities(state, dt, channels))
        state = tdvp(H_eff, -1im * dt, state;
            nsite=2, maxdim=Dmax, cutoff=cutoff, normalize=false,
            updater_kwargs=(; ishermitian=false, tol=1e-8, krylovdim=15, maxiter=30, eager=true))
        actual_survival = real(inner(state, state))
        max_survival_error = max(max_survival_error, abs(actual_survival - expected_survival))
        normalize!(state)
    end
    odd, even = measure_string_orders(state, SO_odd, SO_even)
    push!(states, state)
    println("dt=", dt, " T_after_jump=0.05 odd=", real(odd), " even=", real(even),
        " max_survival_error=", max_survival_error, " maxlinkdim=", maxlinkdim(state))
end

for index in 1:(length(states) - 1)
    fidelity = abs(inner(states[index], states[index + 1]))
    odd_a, even_a = measure_string_orders(states[index], SO_odd, SO_even)
    odd_b, even_b = measure_string_orders(states[index + 1], SO_odd, SO_even)
    println("dt comparison ", dt_values[index], " -> ",
        dt_values[index + 1], " infidelity=", 1 - fidelity,
        " odd_difference=", abs(real(odd_a - odd_b)),
        " even_difference=", abs(real(even_a - even_b)))
end
