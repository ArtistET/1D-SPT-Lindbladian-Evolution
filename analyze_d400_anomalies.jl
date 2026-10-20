# Paired, trajectory-level checks for the N=10 U=10 D400 SO anomalies.
using JLD2, Statistics, Printf

const TDS = (0.98, 0.99, 1.0, 1.01, 1.02)
const U0 = Dict(
    0.98 => (0.017961175847, 0.023473487023),
    0.99 => (0.019274007226, 0.022057188588),
    1.0  => (0.020646803789, 0.020646803789),
    1.01 => (0.022050060561, 0.019274271637),
    1.02 => (0.023451386482, 0.017968583696),
)

function one_result(td, id)
    base = "N10_t(0.1,0.2)_tR1.0_tD$(td)_J0.0_U10.0_I10.1_I20.1_IR0.1_ID0.1"
    path = joinpath(@__DIR__, "trajectory_evolution", base,
        "Dmax400_dt0.025_tdvp2_seed260903", "results",
        "T0.0_to_T12.0_traj$(id)-$(id).jld2")
    d = load(path)
    d["completed_trajectories"] == 1 || error("incomplete $path")
    times = d["times"]
    length(times) == 121 && maximum(abs.(times .- collect(0:0.1:12))) < 1e-10 || error("time grid $path")
    odd = vec(real.(d["C_odd_samples"]))
    even = vec(real.(d["C_even_samples"]))
    jumps = vec(d["jump_indices"])
    all(isfinite, odd) && all(isfinite, even) || error("nonfinite $path")
    return (odd=odd, even=even, jumps=jumps)
end

function stats(v)
    n = length(v)
    return mean(v), std(v) / sqrt(n)
end

at(t) = round(Int, 10t) + 1

println("kind,M,tD,parity,t_start,t_end,mean,se,z,frac_positive,extra")
for td in TDS
    rows = [one_result(td, id) for id in 1:64]
    for m in (16, 32, 64), parity in (:odd, :even)
        values = [getproperty(r, parity) for r in rows[1:m]]
        for (ta, tb) in ((2, 4), (4, 6), (2, 6))
            diffs = [v[at(tb)] - v[at(ta)] for v in values]
            mu, se = stats(diffs)
            jumps = [count(!iszero, r.jumps[(40ta+1):(40tb)]) for r in rows[1:m]]
            njump = count(>(0), jumps)
            @printf("paired_change,%d,%.2f,%s,%d,%d,%.9g,%.9g,%.3f,%.4f,%d_with_jump\n",
                m, td, parity, ta, tb, mu, se, mu / se, mean(diffs .> 0), njump)
        end
        gaps = [v[at(12)] - U0[td][parity == :odd ? 1 : 2] for v in values]
        mu, se = stats(gaps)
        @printf("crossing,%d,%.2f,%s,12,12,%.9g,%.9g,%.3f,%.4f,U0_exact_fixed\n",
            m, td, parity, mu, se, mu / se, mean(gaps .< 0))
    end
end
