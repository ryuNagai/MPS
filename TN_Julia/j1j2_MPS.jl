include("./MPS.jl")
using .MPSforQuantum
using TensorOperations
using LinearAlgebra
using Plots

L = 4
D = 14
j1 = 1.0
j2 = 0.5
N = L^2

mps = init_rand_MPS(N, D, 'r') # convert to MPS
O = j1j2_2D_Hamiltonian_MPO(L, j1, j2)

(opt_mps, hist) = iterative_ground_state_search(mps, O, D, 2)

println(hist / N)