using ITensors, ITensorMPS, JLD2, LinearAlgebra
include("AKLT_evolution.jl")

output = "experiment_results/ground_state_N10_U10_QN_D400_D800.csv"
isfile(output) && error("refusing to overwrite $output")
rows = []
for td in (0.98, 0.99, 1.0, 1.01, 1.02), (d, step) in ((400, 50), (800, 100))
    path = "ground_states/N10_t(0.1,0.2)_tR1.0_tD$(td)_J0.0_U10.0/Dmax$(d)/Dstep$(step)/AKLT_N10_t(0.1,0.2)_tR1.0_tD$(td)_J0.0_U10.0_Dmax$(d).jld2"
    isfile(replace(path, ".jld2" => ".complete")) || error("incomplete $path")
    psi = load(path, "psi0")
    length(psi) == 20 && all(hasqns, siteinds(psi)) || error("sites/QN $path")
    norm_error = abs(norm(psi) - 1)
    norm_error < 1e-8 || error("norm $path")
    sites = siteinds(psi)
    h = MPO(system_ham(10, 0.1, 0.2, 1.0, td, 0.0, 10.0), sites)
    energy = real(inner(psi, h, psi))
    so = ComplexF64[]
    for parity in ("odd", "even")
        head, body, tail = create_SO(sites, 2, 8, 10, parity)
        op = SO_MPO(sites, head, body, tail)
        direct = -inner(psi', op, psi)
        reference, _ = measure(head, body, tail, psi)
        abs(direct - reference) < 1e-8 || error("SO paths disagree $path $parity")
        push!(so, direct)
    end
    orthogonalize!(psi, 10)
    _, s, _ = svd(psi[10], (linkind(psi, 9), siteind(psi, 10)))
    singular = sort([abs(s[k, k]) for k in 1:dim(s, 1)])
    vals = (energy, real(so[1]), real(so[2]), imag(so[1]), imag(so[2]), norm_error,
        first(singular), sum(abs2, singular[1:min(10, end)]))
    all(isfinite, vals) || error("nonfinite $path")
    maxlinkdim(psi) <= d || error("bond cap $path")
    row = (td, d, vals..., maxlinkdim(psi), dim(linkind(psi, 10)), true)
    push!(rows, row)
    println("PASS ", row)
    flush(stdout)
end
open(output, "w") do io
    println(io, "tD,Dmax,energy,odd,even,odd_imag,even_imag,norm_error,min_mid_singular,tail10_sq,maxlinkdim,Dmid,hasqns")
    foreach(row -> println(io, join(row, ',')), rows)
end
println("VALIDATED 10 ground states; wrote $output")
