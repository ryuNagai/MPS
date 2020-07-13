

module MPSforQuantum
    export MPS, restore, OneQubitGate, dstack, ddstack, mps_size
    export CX, inner_product
    using LinearAlgebra

    function SVD(C::Array{ComplexF64,2}, eps::Float64)
        #eps = 1e-2
        A = svd(C)
        filter!((x) -> x > eps, A.S)
        l = length(A.S)
        return A.U[:, 1:l], diagm(A.S) * A.Vt[1:l, :]
    end

    function SVD(C::Array{ComplexF64,2}, D::Int64)
        A = svd(C)
        if length(A.S) > D
            return A.U[:, 1:D], diagm(A.S[1:D]) * A.Vt[1:D, :]
        else
            return A.U[:, :], diagm(A.S) * A.Vt[:, :]
        end
    end

    function MPS(C::Array{ComplexF64,1}, param)
        # C = copy(C0)
        d = 2
        N = Int64(log2(size(C)[1]))
        arrs = []
        #eps = 1e-2
        #D = 550

        r = 1
        for i = 1:N-1
            C = reshape(C, d * r, d^(N - i))
            tmp_A, C = SVD(C, param)
            r = size(tmp_A, 2)
            col = convert(Int64, size(tmp_A, 1) / 2)
            push!(arrs, [tmp_A[1:col, :], tmp_A[(col+1):col*2, :]])
        end

        Ct = transpose(C)
        col2 = convert(Int64, size(C, 2) / 2)
        push!(arrs, [C[:, 1:col2], C[:, (col2+1):col2*2]])
        return arrs
    end

    function mps_size(arrs::Array{Any, 1})
        params = 0
        N = size(arrs)[1]
        for i = 1:N
            println("array $i's size: ", size(arrs[i][1]))
            params += length(arrs[i][1]) * 2
        end
        println("Num of parameters: ", params)
        println("2^N: ", 2^N)
    end

    # Restore prob. ampl. of a state from MPS
    function restore(arrs::Array{Any, 1}, n::Int64) # n is decimal representation of the state '0110...'
        N = Int64(size(arrs)[1])
        s = bitstring(n)[end - (N - 1):end] # 後ろからN bit目まで
        phys_idx = [convert(Int64, s[i]) - 48 for i = length(s):-1:1] # 後ろが最上位ビット
        return prod([arrs[i][phys_idx[i] + 1] for i = 1:length(phys_idx)])[1]
    end

    function OneQubitGate(arrs::Array{Any, 1}, O::Array{Complex{Float64},2}, n::Int64)
        arrs_ = similar(arrs[n + 1])
        arrs__ = copy(arrs)
        arrs_[1] = arrs[n + 1][1] * O[1, 1] + arrs[n + 1][2] * O[2, 1]
        arrs_[2] = arrs[n + 1][1] * O[1, 2] + arrs[n + 1][2] * O[2, 2]
        arrs__[n + 1] = arrs_
        return arrs__
    end

    function dstack(A::Array{ComplexF64,2}, B::Array{ComplexF64,2})
        return cat(A, B, dims = 3)
    end
    
    function ddstack(A::Array{ComplexF64,2}, B::Array{ComplexF64,2}, 
                    C::Array{ComplexF64,2}, D::Array{ComplexF64,2})
        AC = cat(A, C, dims = 3)
        BD = cat(B, D, dims = 3)
        return cat(AC, BD, dims = 4)
    end

    function inner_product(arrs1::Array{Any, 1}, arrs2::Array{Any, 1})
        N = Int64(size(arrs1)[1])
        ip = arrs1[1][1]' * arrs2[1][1] + arrs1[1][2]' * arrs2[1][2]
        for i = 2:N
            phys0 = arrs1[i][1]' * ip * arrs2[i][1]
            phys1 = arrs1[i][2]' * ip * arrs2[i][2]
            ip = phys0 + phys1
        end
        return ip
    end

    function CX(arrs::Array{Any, 1}, n::Int64, t::Number)
        tmp = copy(arrs)
        arrs_ = [tmp[n + 1][1]*tmp[n + 2][1] tmp[n + 1][2]*tmp[n + 2][2]
                tmp[n + 1][1]*tmp[n + 2][2] tmp[n + 1][2]*tmp[n + 2][1]]
        arrs__ = MPSforQuantum.SVD(arrs_, t)
        col = convert(Int64, size(arrs__[1], 1) / 2)
        
        tmp[n + 1][1] = arrs__[1][1:col, :]
        tmp[n + 1][2] = arrs__[1][(col+1):(2*col), :]
        tmp[n + 2][1] = arrs__[2][:, 1:col]
        tmp[n + 2][2] = arrs__[2][:, (col+1):(2*col)]
    
        return tmp
    end

end

