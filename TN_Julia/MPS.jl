

module MPSforQuantum
    export MPS, restore, OneQubitGate, dstack, ddstack, mps_size
    export CX, inner_product, expectation, SVD_R
    using LinearAlgebra
    using TensorOperations

    function SVD(C::Array{ComplexF64,2}, eps::Float64)
        #eps = 1e-2
        A = svd(C)
        filter!((x) -> x > eps, A.S)
        l = length(A.S)
        return A.U[:, 1:l], diagm(A.S), A.Vt[1:l, :]
    end

    function SVD(C::Array{ComplexF64,2}, D::Int64)
        A = svd(C)
        if length(A.S) > D
            return A.U[:, 1:D], diagm(A.S[1:D]), A.Vt[1:D, :]
        else
            return A.U[:, :], diagm(A.S), A.Vt[:, :]
        end
    end

    function SVD_L(C::Array{ComplexF64,2}, eps::Float64)
        U, S, Vt = SVD(C, eps)
        return U, S * Vt
    end

    function SVD_L(C::Array{ComplexF64,2}, D::Int64)
        U, S, Vt = SVD(C, D)
        return U, S * Vt
    end

    function SVD_R(C::Array{ComplexF64,2}, eps::Float64)
        U, S, Vt = SVD(C, eps)
        return U * S, Vt
    end

    function SVD_R(C::Array{ComplexF64,2}, D::Int64)
        U, S, Vt = SVD(C, D)
        return U * S, Vt
    end

    function MPS(C::Array{ComplexF64,1}, param, normalize::Char='l')
        d = 2
        N = Int64(log2(size(C)[1]))
        arrs = []
        r = 1
        if normalize == 'l'
            for i = 1:N-1
                C = reshape(C, d * r, d^(N - i))
                tmp_A, C = SVD_L(C, param)
                r = size(tmp_A, 2)
                col = convert(Int64, size(tmp_A, 1) / 2)
                push!(arrs, [tmp_A[1:col, :], tmp_A[(col+1):col*2, :]])
            end

            Ct = transpose(C)
            col2 = convert(Int64, size(C, 2) / 2)
            push!(arrs, [C[:, 1:col2], C[:, (col2+1):col2*2]])
            return arrs
        elseif normalize == 'r'
            for i = 1:N-1
                C = reshape(C, d^(N - i), d * r)
                C, tmp_B = SVD_R(C, param)
                r = size(tmp_B, 1)
                col = convert(Int64, size(tmp_B, 2) / 2)
                push!(arrs, [tmp_B[:, 1:col], tmp_B[:, (col+1):col*2]])
            end

            Ct = transpose(C)
            col2 = convert(Int64, size(C, 1) / 2)
            push!(arrs, [C[1:col2, :], C[(col2+1):col2*2, :]])
            arrs_ = []
            for i = 1:N
                push!(arrs_, arrs[N - (i-1)])
            end
            return arrs_
        end
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

    function dstack(operators)
        return cat(operators..., dims = 3)
    end
    
    function ddstack(operators)
        m = size(operators)[1]
        tmp2 = []
        for i=1:m
            push!(tmp2, cat(operators[i]..., dims=4))
        end
        tmp3 = Tuple(tmp2)
        mpo = cat(tmp3..., dims=3)
        return mpo
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
        arrs__ = MPSforQuantum.SVD_L(arrs_, t)
        col = convert(Int64, size(arrs__[1], 1) / 2)
        
        tmp[n + 1][1] = arrs__[1][1:col, :]
        tmp[n + 1][2] = arrs__[1][(col+1):(2*col), :]
        tmp[n + 2][1] = arrs__[2][:, 1:col]
        tmp[n + 2][2] = arrs__[2][:, (col+1):(2*col)]
    
        return tmp
    end

    function expectation(mps::Array{Any,1}, O::Array{Any,1})
        N_site = size(mps)[1]
        contracted_sites = []
        n_phys = 2 #ss
        for i=1:N_site
            if i==1
                # first step
                a_len = size(mps[i][1])[2]
                b_len = size(O[i][1, 1, :])[1] # bb0
                arr_0 = zeros(ComplexF64, n_phys, b_len, a_len)
                for sigma1=1:n_phys, a_=1:a_len, b = 1:b_len
                    for sigma2=1:n_phys
                        arr_0[sigma1, b, a_] += O[i][sigma1, sigma2, b] * mps[i][sigma2][1, a_]
                    end
                end
                arr = zeros(ComplexF64, a_len, b_len, a_len)
                for a=1:a_len, a_=1:a_len, b = 1:b_len
                    for sigma1=1:n_phys
                        arr[a, b, a_] += mps[i][sigma1]'[a, 1] * arr_0[sigma1, b, a_]
                    end
                end
            
            elseif i==N_site
                # last step
                a_len1 = size(mps[i-1][1])[2]
                a_len2 = size(mps[i][1])[2]
                b_len2 = size(O[i][1, 1, :])[1]
                arr_0 = zeros(ComplexF64, n_phys, a_len1, b_len2, a_len2)
                for sigma2 = 1:n_phys, a1=1:a_len1, b2 = 1:b_len2, a2 = 1:a_len2
                    for a1_0=1:a_len1
                        arr_0[sigma2, a1, b2, a2] += contracted_sites[i-1][a1, b2, a1_0] * mps[i][sigma2][a1_0, a2]
                    end
                end
    
                arr_1 = zeros(ComplexF64, n_phys, a_len1, a_len2)
                for sigma1 = 1:n_phys, a1=1:a_len1, a2 = 1:a_len2
                    for sigma2 = 1:n_phys, b2 = 1:b_len2
                        arr_1[sigma1, a1, a2] += O[i][sigma1, sigma2, b2] * arr_0[sigma2, a1, b2, a2]
                    end
                end
    
                arr = zeros(ComplexF64, a_len2, a_len2)
                for a2 = 1:a_len2, a2 = 1:a_len2
                    for sigma1 = 1:n_phys, a1=1:a_len1
                        arr[a2, a2] += mps[i][sigma1]'[a2, a1] * arr_1[sigma1, a1, a2]
                    end
                end
            
            else
                # Middle step
                a_len1 = size(mps[i-1][1])[2]
                a_len2 = size(mps[i][1])[2]
                b_len1, b_len2 = size(O[i][1, 1, :, :])
                arr_0 = zeros(ComplexF64, n_phys, a_len1, b_len1, a_len2)
                for sigma2 = 1:n_phys, a1 = 1:a_len1, b1 = 1:b_len1, a2=1:a_len2
                    for a0_0 = 1:a_len1
                        arr_0[sigma2, a1, b1, a2] += contracted_sites[i-1][a1, b1, a0_0] * mps[i][sigma2][a0_0, a2]
                    end
                end
    
                arr_1 = zeros(ComplexF64, n_phys, b_len2, a_len1, a_len2)
                for sigma1 = 1:n_phys, b2 = 1:b_len2, a1 = 1:a_len1, a2=1:a_len2
                    for sigma2=1:n_phys, b1 = 1:b_len1
                        arr_1[sigma1, b2, a1, a2] += O[i][sigma1, sigma2, b1, b2] * arr_0[sigma2, a1, b1, a2]
                    end
                end
    
                arr = zeros(ComplexF64, a_len2, b_len2, a_len2)
                for a2_=1:a_len2, b2 = 1:b_len2, a2=1:a_len2
                    for sigma1 = 1:n_phys, a1 = 1:a_len1
                        arr[a2, b2, a2_] += mps[i][sigma1]'[a2, a1] * arr_1[sigma1, b2, a1, a2_]
                    end
                end
            end
            push!(contracted_sites, arr)
        end
        return contracted_sites[N_site]
    end

end

