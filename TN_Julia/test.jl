
include("./MPS.jl")
using .MPSforQuantum
using LinearAlgebra

pauliX = convert(Array{ComplexF64,2}, [0 1; 1 0])
pauliY = [0 -im; im 0]
pauliZ = convert(Array{ComplexF64,2}, [1 0; 0 -1])
Hadamard = convert(Array{ComplexF64,2}, [1 1; 1 -1] ./ sqrt(2))
pauliI = convert(Array{ComplexF64,2}, [1 0; 0 1])
zero = convert(Array{ComplexF64,2}, [0 0; 0 0])

N = 20
#C0 = normalize!(rand(ComplexF64, 2^N)) # for random state
C0 = zeros(ComplexF64, 2^N)
C0[1] = 1 # '|000>'

D = 10
eps = 1e-3
mps = MPS(C0, eps)
println("MPS's information")
mps_size(mps)
mps = OneQubitGate(mps, Hadamard, 0) # -> '001' and '000'

println("\nMPS after gate Hadamard[0]")
for i=0:7
    res = restore(mps, i)
    println('|', bitstring(i)[end - 2:end], ">: ",  res)
end
#ans = C0[n + 1]
#println(ans)

println("\nBell state")
N = 2
C0 = zeros(ComplexF64, 2^N)
C0[1] = 1 # '000'
mps1 = MPS(C0, eps)
mps1 = OneQubitGate(mps1, Hadamard, 0) # -> '001' & '000'
arr = CX(mps1, 0, eps)
mps_size(arr)
for i=0:3
    res = restore(mps1, i)
    println('|', bitstring(i)[end - 1:end], ">: ",  res)
end

### MPO
O = []
push!(O, dstack(pauliZ, pauliI))
push!(O, ddstack(pauliZ, zero, zero, pauliZ))
push!(O, dstack(pauliI, pauliZ))

#println("pauliZ", O[1][:, :, 1]) # [site], [phys1, phys2, i(, j)]
#println("pauliI", O[1][:, :, 2]) # [site], [phys1, phys2, i(, j)]