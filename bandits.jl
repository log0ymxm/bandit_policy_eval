import JSON
using ProgressMeter
using DataStructures: DefaultDict
using StatsBase

function finite_policy_evaluator(A, S)
    h = []
    G = 0
    T = 0

    n = 45811883 # just known based on our dataset
    p = Progress(n, 1)
    for (i, (x, a, r)) in enumerate(S)
        if A(h, x) == a
            push!(h, [x, a, r])
            G += r
            T += 1
        end
        next!(p)
    end

    return G / T
end

function webscope_data()
    open("/mnt/md0/data/webscope_user_click_log/parsed.jl") do f
        for line in eachline(f)
            record = JSON.parse(line)
            produce([record, record["displayed_article"], record["user_clicked"]])
        end
    end
end

function beta_posterior_lower_bounds(n, s, alpha=1, beta=1)
    a = float(alpha .+ s)
    b = float(beta .+ n .- s)
    z = -1.65
    return (a ./ (a .+ b)) .+ z .* sqrt((a.*b) ./ ((a.+b).^2 .* (a .+ b .+ 1)))
end

n_successes = DefaultDict{AbstractString, Int}(0)
n_samples = DefaultDict{AbstractString, Int}(0)

successes(s) = n_successes[s]
samples(s) = n_samples[s]

function epsilon_greedy(history, visit, epsilon=0.1)
    if length(history) > 1
        x, a, r = history[end]
        n_samples[a] += 1
        n_successes[a] += r
    end

    choices = collect(keys(visit["articles"]))

    if rand() < epsilon
        return sample(choices)
    else
        n = samples.(choices)
        s = successes.(choices)
        p = beta_posterior_lower_bounds(n, s)
        _, idx = findmax(p)
        return choices[idx]
    end
end

print("Epsilon Greedy")
print(finite_policy_evaluator(epsilon_greedy, Task(webscope_data)))
