using JLD2
using Statistics
using Printf

length(ARGS) == 2 || error("usage: julia --project=. analyze_trajectory_results.jl TRAJECTORY_ROOT OUTPUT.csv")

trajectory_root, output_path = ARGS
tD_values = [0.98, 0.99, 1.0, 1.01, 1.02]
sample_sizes = [4, 8, 16, 32]

function result_directory(root, tD)
    parameter_dir = "N10_t(0.1,0.2)_tR1.0_tD$(tD)_J0.0_U10.0_I10.1_I20.1_IR0.1_ID0.1"
    return joinpath(root, parameter_dir, "Dmax100_dt0.05_seed260903", "results")
end

function load_samples(directory)
    isdir(directory) || error("missing result directory: $directory")
    matched = Tuple{Int,Int,String}[]
    for filename in readdir(directory)
        match_result = match(r"^T0\.0_to_T0\.5_traj(\d+)-(\d+)\.jld2$", filename)
        isnothing(match_result) && continue
        first_id, last_id = parse.(Int, match_result.captures)
        last_id - first_id == 3 || continue
        push!(matched, (first_id, last_id, joinpath(directory, filename)))
    end
    sort!(matched; by=first)
    length(matched) == 8 || error("expected 8 four-trajectory files in $directory, found $(length(matched))")

    ids = Int[]
    odd_rows = Vector{Float64}[]
    even_rows = Vector{Float64}[]
    bond_rows = Vector{Int}[]
    jump_rows = Vector{Int}[]
    reference_times = nothing
    for (first_id, last_id, path) in matched
        data = load(path)
        data["completed_trajectories"] == 4 || error("incomplete result: $path")
        times = Float64.(data["times"])
        isnothing(reference_times) ? (reference_times = times) : (times == reference_times || error("time grid mismatch: $path"))
        odd = real.(data["C_odd_samples"])
        even = real.(data["C_even_samples"])
        bonds = data["bond_dimensions"]
        jumps = data["jump_indices"]
        for row in 1:4
            push!(ids, first_id + row - 1)
            push!(odd_rows, vec(odd[row, :]))
            push!(even_rows, vec(even[row, :]))
            push!(bond_rows, vec(bonds[row, :]))
            push!(jump_rows, vec(jumps[row, :]))
        end
        ids[end] == last_id || error("trajectory id mismatch: $path")
    end
    order = sortperm(ids)
    ids = ids[order]
    ids == collect(1:32) || error("expected trajectory ids 1:32 in $directory, found $ids")
    return reference_times, reduce(vcat, permutedims.(odd_rows[order])),
        reduce(vcat, permutedims.(even_rows[order])),
        reduce(vcat, permutedims.(bond_rows[order])),
        reduce(vcat, permutedims.(jump_rows[order]))
end

mkpath(dirname(abspath(output_path)))
open(output_path, "w") do io
    println(io, "tD,tR_over_tD,time,samples,odd_mean,odd_stderr,even_mean,even_stderr,max_bond_dimension,mean_cumulative_jumps")
    for tD in tD_values
        times, odd, even, bonds, jumps = load_samples(result_directory(trajectory_root, tD))
        for sample_count in sample_sizes
            for time_index in eachindex(times)
                odd_values = odd[1:sample_count, time_index]
                even_values = even[1:sample_count, time_index]
                odd_stderr = sample_count > 1 ? std(odd_values) / sqrt(sample_count) : NaN
                even_stderr = sample_count > 1 ? std(even_values) / sqrt(sample_count) : NaN
                cumulative_jumps = time_index == 1 ? zeros(sample_count) :
                    vec(sum(jumps[1:sample_count, 1:(time_index - 1)] .!= 0; dims=2))
                @printf(io, "%.12g,%.12g,%.12g,%d,%.16g,%.16g,%.16g,%.16g,%d,%.16g\n",
                    tD, 1 / tD, times[time_index], sample_count,
                    mean(odd_values), odd_stderr, mean(even_values), even_stderr,
                    maximum(bonds[1:sample_count, time_index]), mean(cumulative_jumps))
            end
        end
    end
end

println("Wrote ", output_path)
