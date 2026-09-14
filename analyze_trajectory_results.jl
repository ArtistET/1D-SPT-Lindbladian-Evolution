using JLD2
using Statistics
using Printf

length(ARGS) in 2:7 || error("usage: julia --project=. analyze_trajectory_results.jl TRAJECTORY_ROOT OUTPUT.csv [FINAL_TIME] [EXPECTED_SAMPLES] [RUN_DIRECTORY] [INITIAL_TIME] [SEGMENT_BOUNDARY]")

trajectory_root, output_path = ARGS[1:2]
final_time = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 0.5
expected_samples = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 64
run_directory = length(ARGS) >= 5 ? ARGS[5] : "Dmax100_dt0.025_dynamicE_seed260903"
initial_time = length(ARGS) >= 6 ? parse(Float64, ARGS[6]) : 0.0
segment_boundary = length(ARGS) >= 7 ? parse(Float64, ARGS[7]) : nothing
tD_values = [0.98, 0.99, 1.0, 1.01, 1.02]
sample_sizes = unique(sort([filter(n -> n <= expected_samples, [4, 8, 16, 32, 64, 128, 256, 512, 1024]); expected_samples]))
final_time_pattern = replace(string(final_time), "." => "\\.")
initial_time_pattern = replace(string(initial_time), "." => "\\.")
result_pattern = Regex("^T$(initial_time_pattern)_to_T$(final_time_pattern)_traj(\\d+)-(\\d+)\\.jld2\$")

function result_directory(root, tD)
    parameter_dir = "N10_t(0.1,0.2)_tR1.0_tD$(tD)_J0.0_U10.0_I10.1_I20.1_IR0.1_ID0.1"
    return joinpath(root, parameter_dir, run_directory, "results")
end

function load_samples(directory)
    isdir(directory) || error("missing result directory: $directory")
    matched = Tuple{Int,Int,String}[]
    for filename in readdir(directory)
        match_result = match(result_pattern, filename)
        isnothing(match_result) && continue
        first_id, last_id = parse.(Int, match_result.captures)
        push!(matched, (first_id, last_id, joinpath(directory, filename)))
    end
    sort!(matched; by=item -> (item[1], item[2]))

    ids = Int[]
    odd_rows = Vector{Float64}[]
    even_rows = Vector{Float64}[]
    bond_rows = Vector{Int}[]
    jump_rows = Vector{Int}[]
    reference_times = nothing
    reference_dt = nothing
    for (first_id, last_id, path) in matched
        data = load(path)
        dt = Float64(data["args"]["dt"])
        isnothing(reference_dt) ? (reference_dt = dt) : (dt == reference_dt || error("dt mismatch: $path"))
        completed = data["completed_trajectories"]
        completed <= last_id - first_id + 1 || error("invalid completed_trajectories in $path")
        times = Float64.(data["times"])
        isnothing(reference_times) ? (reference_times = times) : (times == reference_times || error("time grid mismatch: $path"))
        odd = real.(data["C_odd_samples"])
        even = real.(data["C_even_samples"])
        bonds = data["bond_dimensions"]
        jumps = data["jump_indices"]
        for row in 1:completed
            trajectory_id = first_id + row - 1
            trajectory_id > expected_samples && continue
            push!(ids, trajectory_id)
            push!(odd_rows, vec(odd[row, :]))
            push!(even_rows, vec(even[row, :]))
            push!(bond_rows, vec(bonds[row, :]))
            push!(jump_rows, vec(jumps[row, :]))
        end
    end

    if !isnothing(segment_boundary)
        missing_ids = setdiff(1:expected_samples, ids)
        for trajectory_id in missing_ids
            filenames = [
                "T$(initial_time)_to_T$(segment_boundary)_traj$(trajectory_id)-$(trajectory_id).jld2",
                "T$(segment_boundary)_to_T$(final_time)_traj$(trajectory_id)-$(trajectory_id).jld2",
            ]
            paths = joinpath.(directory, filenames)
            all(isfile, paths) || continue
            first_segment, second_segment = load.(paths)
            all(segment["completed_trajectories"] == 1 for segment in (first_segment, second_segment)) ||
                error("incomplete segmented trajectory $trajectory_id in $directory")
            dts = Float64[segment["args"]["dt"] for segment in (first_segment, second_segment)]
            dts[1] == dts[2] || error("dt mismatch in segmented trajectory $trajectory_id")
            isnothing(reference_dt) ? (reference_dt = dts[1]) :
                (dts[1] == reference_dt || error("dt mismatch in segmented trajectory $trajectory_id"))
            times = vcat(Float64.(first_segment["times"]), Float64.(second_segment["times"])[2:end])
            isnothing(reference_times) ? (reference_times = times) :
                (length(times) == length(reference_times) &&
                    all(isapprox.(times, reference_times; atol=1e-12, rtol=0)) ||
                    error("time grid mismatch in segmented trajectory $trajectory_id"))
            push!(ids, trajectory_id)
            push!(odd_rows, vcat(vec(real.(first_segment["C_odd_samples"][1, :])),
                vec(real.(second_segment["C_odd_samples"][1, 2:end]))))
            push!(even_rows, vcat(vec(real.(first_segment["C_even_samples"][1, :])),
                vec(real.(second_segment["C_even_samples"][1, 2:end]))))
            push!(bond_rows, vcat(vec(first_segment["bond_dimensions"][1, :]),
                vec(second_segment["bond_dimensions"][1, 2:end])))
            push!(jump_rows, vcat(vec(first_segment["jump_indices"][1, :]),
                vec(second_segment["jump_indices"][1, :])))
        end
    end
    order = sortperm(ids)
    ids = ids[order]
    ids == collect(1:expected_samples) || error("expected trajectory ids 1:$expected_samples in $directory, found $ids")
    return reference_times, reduce(vcat, permutedims.(odd_rows[order])),
        reduce(vcat, permutedims.(even_rows[order])),
        reduce(vcat, permutedims.(bond_rows[order])),
        reduce(vcat, permutedims.(jump_rows[order])), reference_dt
end

mkpath(dirname(abspath(output_path)))
datasets = Dict(tD => load_samples(result_directory(trajectory_root, tD)) for tD in tD_values)
open(output_path, "w") do io
    println(io, "tD,tR_over_tD,time,samples,dt,odd_mean,odd_stderr,even_mean,even_stderr,max_bond_dimension,mean_cumulative_jumps")
    for tD in tD_values
        times, odd, even, bonds, jumps, dt = datasets[tD]
        for sample_count in sample_sizes
            for time_index in eachindex(times)
                odd_values = odd[1:sample_count, time_index]
                even_values = even[1:sample_count, time_index]
                odd_stderr = sample_count > 1 ? std(odd_values) / sqrt(sample_count) : NaN
                even_stderr = sample_count > 1 ? std(even_values) / sqrt(sample_count) : NaN
                completed_steps = round(Int, (times[time_index] - first(times)) / dt)
                cumulative_jumps = completed_steps == 0 ? zeros(sample_count) :
                    vec(sum(jumps[1:sample_count, 1:completed_steps] .!= 0; dims=2))
                @printf(io, "%.12g,%.12g,%.12g,%d,%.12g,%.16g,%.16g,%.16g,%.16g,%d,%.16g\n",
                    tD, 1 / tD, times[time_index], sample_count, dt,
                    mean(odd_values), odd_stderr, mean(even_values), even_stderr,
                    maximum(bonds[1:sample_count, time_index]), mean(cumulative_jumps))
            end
        end
    end
end

println("Wrote ", output_path)

slope_output_path = replace(output_path, r"\.csv$" => "_slopes.csv")
times, odd_low, even_low = datasets[1.01][1:3]
_, odd_high, even_high = datasets[0.99][1:3]
ratio_width = 1 / 0.99 - 1 / 1.01
open(slope_output_path, "w") do io
    println(io, "time,samples,odd_slope,odd_slope_stderr,even_slope,even_slope_stderr")
    for sample_count in sample_sizes
        for time_index in eachindex(times)
            odd_slopes = (odd_high[1:sample_count, time_index] .- odd_low[1:sample_count, time_index]) ./ ratio_width
            even_slopes = (even_high[1:sample_count, time_index] .- even_low[1:sample_count, time_index]) ./ ratio_width
            @printf(io, "%.12g,%d,%.16g,%.16g,%.16g,%.16g\n", times[time_index], sample_count,
                mean(odd_slopes), std(odd_slopes) / sqrt(sample_count),
                mean(even_slopes), std(even_slopes) / sqrt(sample_count))
        end
    end
end
println("Wrote ", slope_output_path)
