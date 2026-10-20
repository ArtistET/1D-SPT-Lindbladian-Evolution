# Exact U=0 string order from the Slater correlation matrix and determinant
# full-counting statistics; independent of DMRG truncation.
using LinearAlgebra, Printf

function one_body(td)
    n = 10
    h = zeros(Float64, 2n, 2n)
    for j in 1:n, alpha in 1:2
        s = 2j - 2 + alpha
        nxt = 2(j % n + 1) - 2 + alpha
        hop = (0.1, 0.2)[alpha]
        h[s, nxt] -= hop
        h[nxt, s] -= hop
        h[s, s + 3 - 2alpha] -= 1.0
        diag = s == 1 ? 2n : (s - 3 + 2alpha) % (2n)
        h[s, diag] -= td
    end
    isapprox(h, h'; atol=1e-12) || error("non-Hermitian h")
    eig = eigen(Symmetric(h))
    C = eig.vectors[:, 1:n] * eig.vectors[:, 1:n]'
    norm(C * C - C) < 1e-12 || error("bad Slater projector")
    return C, 2sum(eig.values[1:n])
end

function moments(C, body, phase, endpoints)
    factors = ones(ComplexF64, size(C, 1))
    factors[body] .= phase
    F(selected) = begin
        f = copy(factors)
        f[selected] .= 0
        det(I - C + C * Diagonal(f))
    end
    base = F(Int[])
    one = Dict(a => base - F([a]) for a in endpoints)
    two = Dict((a, b) => base - F([a]) - F([b]) + F([a, b])
        for a in endpoints for b in endpoints if a < b)
    return base, one, two
end

function exact_so(C, parity)
    head, tail, body = parity == :odd ?
        ([3, 4], [15, 16], collect(5:14)) :
        ([4, 5], [16, 17], [s for j in 3:7 for s in (2j + 1, 2j)])
    # Even body sites are (7,6), (9,8), ..., (15,14).
    endpoints = vcat(head, tail)
    up0, up1, up2 = moments(C, body, 1im, endpoints)
    dn0, dn1, dn2 = moments(C, body, -1im, endpoints)
    value = 0.0 + 0.0im
    for a in head, b in tail
        pair = (min(a, b), max(a, b))
        value -= (up2[pair] * dn0 + up0 * dn2[pair] -
            up1[a] * dn1[b] - dn1[a] * up1[b]) / 4
    end
    return value
end

println("tD,exact_energy,odd,even,odd_imag,even_imag")
for td in (0.98, 0.99, 1.0, 1.01, 1.02)
    C, energy = one_body(td)
    odd, even = exact_so(C, :odd), exact_so(C, :even)
    @printf("%.2f,%.12f,%.12f,%.12f,%.3g,%.3g\n",
        td, energy, real(odd), real(even), imag(odd), imag(even))
end
