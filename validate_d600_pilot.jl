using ITensors, ITensorMPS, JLD2, Statistics, Printf

const TDS = (0.98, 1.0, 1.02)
const IDS = 1:4
const TIMES = collect(0:0.1:12)

function paths(td, id, dmax)
    base = joinpath(@__DIR__, "trajectory_evolution",
        "N10_t(0.1,0.2)_tR1.0_tD$(td)_J0.0_U10.0_I10.1_I20.1_IR0.1_ID0.1",
        "Dmax$(dmax)_dt0.025_tdvp2_seed260903")
    return (joinpath(base, "results", "T0.0_to_T12.0_traj$(id)-$(id).jld2"),
        joinpath(base, "checkpoints", "T12.0", "trajectory_$(id).jld2"))
end

function read_one(td, id, dmax)
    result, checkpoint = paths(td, id, dmax)
    all(isfile, (result, checkpoint)) || error("missing result/checkpoint for tD=$td id=$id D=$dmax")
    d = load(result)
    a = d["args"]
    for (key, value) in (("N", 10), ("U", 10.0), ("tD", td), ("Dmax", dmax),
        ("Dload", 400), ("dt", 0.025), ("tsmax", 480), ("measure-every", 4),
        ("seed", 260903), ("traj-start", id), ("ntraj", 1))
        a[key] == value || error("$result: $key=$(a[key]), expected $value")
    end
    d["completed_trajectories"] == 1 || error("incomplete $result")
    times = d["times"]
    length(times) == 121 && maximum(abs.(times .- TIMES)) < 1e-10 || error("time grid $result")
    odd, even = vec(d["C_odd_samples"]), vec(d["C_even_samples"])
    jumps = vec(d["jump_indices"])
    probs = vec(d["total_jump_probabilities"])
    weights = vec(d["branch_weights"])
    bonds = vec(d["bond_dimensions"])
    length(odd) == length(even) == length(bonds) == 121 || error("measurement dimensions $result")
    length(jumps) == length(probs) == length(weights) == 480 || error("step dimensions $result")
    all(x -> isfinite(real(x)) && isfinite(imag(x)), odd) || error("nonfinite odd $result")
    all(x -> isfinite(real(x)) && isfinite(imag(x)), even) || error("nonfinite even $result")
    all(p -> isfinite(p) && 0 <= p <= 1, probs) || error("invalid jump probabilities $result")
    all(isfinite, weights) && all(weights .> 0) || error("invalid branch weights $result")
    all(j -> 0 <= j <= 160, jumps) || error("invalid jump indices $result")
    all(b -> 1 <= b <= dmax, bonds) || error("invalid bonds $result")
    psi = load(checkpoint, "psi")
    hasqns(first(siteinds(psi))) || error("non-QN checkpoint $checkpoint")
    norm_error = abs(norm(psi) - 1)
    norm_error < 1e-8 || error("norm error $norm_error in $checkpoint")
    maxlinkdim(psi) == bonds[end] || error("checkpoint/result bond mismatch $checkpoint")
    orthogonalize!(psi, 10)
    _, s, _ = svd(psi[10], (linkind(psi, 9), siteind(psi, 10)))
    singular = sort([abs(s[k, k]) for k in 1:dim(s, 1)])
    tail10 = sum(abs2, singular[1:min(10, length(singular))])
    return (odd=real.(odd), even=real.(even), jumps=jumps, probs=probs, weights=weights,
        bonds=bonds, norm_error=norm_error, min_s=first(singular), tail10=tail10,
        center_dim=dim(linkind(psi, 10)))
end

at(t) = round(Int, 10t) + 1
paired(v) = (mean(v), std(v) / sqrt(length(v)))

data = Dict((d, td, id) => read_one(td, id, d) for d in (400, 600), td in TDS, id in IDS)
println("VALIDATED 24 paired result/checkpoint files: 12 D400 + 12 D600")
for d in (400, 600)
    rows = [data[d, td, id] for td in TDS for id in IDS]
    @printf("D=%d max_norm_error=%.3g p_jump=[%.6g,%.6g] branch_weight=[%.6g,%.6g] final_bond=[%d,%d] center_bond=[%d,%d] min_s=[%.3g,%.3g] tail10_sq=[%.3g,%.3g]\n",
        d, maximum(r.norm_error for r in rows), minimum(minimum(r.probs) for r in rows),
        maximum(maximum(r.probs) for r in rows), minimum(minimum(r.weights) for r in rows),
        maximum(maximum(r.weights) for r in rows), minimum(last(r.bonds) for r in rows),
        maximum(last(r.bonds) for r in rows), minimum(r.center_dim for r in rows),
        maximum(r.center_dim for r in rows), minimum(r.min_s for r in rows),
        maximum(r.min_s for r in rows), minimum(r.tail10 for r in rows),
        maximum(r.tail10 for r in rows))
end
println("tD,parity,time,D400_mean,D600_mean,paired_delta,paired_SE,jump_mismatch_steps")
for td in TDS, parity in (:odd, :even), t in (2, 4, 6, 10, 12)
    x = [getproperty(data[400, td, id], parity)[at(t)] for id in IDS]
    y = [getproperty(data[600, td, id], parity)[at(t)] for id in IDS]
    mu, se = paired(y .- x)
    mismatches = sum(count(p -> p[1] != p[2], zip(data[400, td, id].jumps, data[600, td, id].jumps)) for id in IDS)
    @printf("%.2f,%s,%d,%.9g,%.9g,%+.9g,%.9g,%d\n", td, parity, t,
        mean(x), mean(y), mu, se, mismatches)
end
for parity in (:odd, :even), t in (2, 4, 6, 10, 12)
    x = [(getproperty(data[400, 1.02, id], parity)[at(t)] - getproperty(data[400, 0.98, id], parity)[at(t)]) / 0.04 for id in IDS]
    y = [(getproperty(data[600, 1.02, id], parity)[at(t)] - getproperty(data[600, 0.98, id], parity)[at(t)]) / 0.04 for id in IDS]
    mu, se = paired(y .- x)
    @printf("slope,%s,%d,D400=%.9g,D600=%.9g,delta=%+.9g,se=%.9g\n", parity, t, mean(x), mean(y), mu, se)
end
