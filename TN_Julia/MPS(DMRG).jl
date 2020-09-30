

module MPSforQuantum
    export MPS, restore, OneQubitGate, dstack, ddstack, mps_size, restore,iterative_ground_state_search
    export CX, inner_product, expectation, SVD_R, SVD_L, R_expression, L_expression, iterative_ground_state_search_2site
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
                #C = reshape(C, d^(N - i), d * r)
                l = convert(Int64, size(C, 1) / 2)
                C = cat(C[1:l, :], C[l+1:l*2, :], dims=2)
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
        # println("mps")
        # println(mps)
        for i=1:N_site
            if i==1
                # first step
                tmp_mps = cat(mps[i][1][:], mps[i][2][:], dims=2)
                tmp_mps_dag = cat(mps[i][1]'[:], mps[i][2]'[:], dims=2)
                tmp_O = O[i]
                a_len = size(mps[i][1])[2]
                b_len = size(O[i][1, 1, :])[1] # bb0
                arr_0 = zeros(ComplexF64, n_phys, b_len, a_len)
                arr = zeros(ComplexF64, a_len, b_len, a_len)
                @tensor begin
                    arr_0[sigma1, b, a_] = tmp_O[sigma1, sigma2, b] * tmp_mps[a_, sigma2]
                    arr[a, b, a_] = tmp_mps_dag[a, sigma1] * arr_0[sigma1, b, a_]
                end

            elseif i==N_site
                # last step
                tmp_mps = cat(mps[i][1][:], mps[i][2][:], dims=2)
                tmp_mps_dag = cat(mps[i][1]'[:], mps[i][2]'[:], dims=2)
                tmp_O = O[i]
                tmp_site = contracted_sites[i-1]
                a_len1 = size(mps[i-1][1])[2]
                b_len2 = size(O[i][1, 1, :])[1]
                arr_0 = zeros(ComplexF64, n_phys, a_len1, b_len2)
                arr_1 = zeros(ComplexF64, n_phys, a_len1)
                @tensor begin
                    arr_0[sigma2, a1, b2] = tmp_site[a1, b2, a1_0] * tmp_mps[a1_0, sigma2]
                    arr_1[sigma1, a1] = tmp_O[sigma1, sigma2, b2] * arr_0[sigma2, a1, b2]
                    arr = tmp_mps_dag[a1, sigma1] * arr_1[sigma1, a1]
                end

            else
                # Middle step
                tmp_mps = cat(mps[i][1], mps[i][2], dims=3)
                tmp_mps_dag = cat(mps[i][1]', mps[i][2]', dims=3)
                tmp_O = O[i]
                tmp_site = contracted_sites[i-1]
                a_len1 = size(mps[i-1][1])[2]
                a_len2 = size(mps[i][1])[2]
                b_len1, b_len2 = size(O[i][1, 1, :, :])
                arr_0 = zeros(ComplexF64, n_phys, a_len1, b_len1, a_len2)
                arr_1 = zeros(ComplexF64, n_phys, b_len2, a_len1, a_len2)
                arr = zeros(ComplexF64, a_len2, b_len2, a_len2)
                @tensor begin
                    arr_0[sigma2, a1, b1, a2] = tmp_site[a1, b1, a0_0] * tmp_mps[a0_0, a2, sigma2]
                    arr_1[sigma1, b2, a1, a2] = tmp_O[sigma1, sigma2, b1, b2] * arr_0[sigma2, a1, b1, a2]
                    arr[a2, b2, a2_] = tmp_mps_dag[a2, a1, sigma1] * arr_1[sigma1, b2, a1, a2_]
                end
            end
            push!(contracted_sites, arr)
        end
        return contracted_sites[N_site]
    end

    function R_expression(mps::Array{Any,1}, O::Array{Any,1}, t::Int64)
        # t: target site, sites t+1 to N are merged to R expression
        N_site = size(mps)[1]
        n_phys = 2
        contracted_sites = []
        for i = t+1:N_site
            if i==t+1
                if i==N_site
                    tmp_mps = cat(mps[i][1][:], mps[i][2][:], dims=2)
                    tmp_mps_dag = cat(mps[i][1]'[:], mps[i][2]'[:], dims=2)
                    tmp_O = O[i]
                    a_len1 = size(mps[i-1][1])[2]
                    a_len2 = size(mps[i][1])[2]
                    b_len = size(O[i][1, 1, :])[1]
                    arr = zeros(ComplexF64, a_len1, b_len, a_len1)
                    @tensor begin
                        arr[a0, b0, a0_] = tmp_mps_dag[a0, sigma1] * tmp_O[sigma1, sigma2, b0] * tmp_mps[a0_, sigma2]
                    end
                else
                    tmp_mps = cat(mps[i][1], mps[i][2], dims=3)
                    tmp_mps_dag = cat(mps[i][1]', mps[i][2]', dims=3)
                    tmp_O = O[i]
                    a_len1 = size(mps[i-1][1])[2]
                    a_len2 = size(mps[i][1])[2]
                    b_len = size(O[i][1, 1, :, :])[1]
                    arr = zeros(ComplexF64, a_len1, a_len2, b_len, b_len, a_len1, a_len2)
                    @tensor begin
                        arr[a0, a1, b0, b1, a0_, a1_] = tmp_mps_dag[a1, a0, sigma1] * tmp_O[sigma1, sigma2, b0, b1] * tmp_mps[a0_, a1_, sigma2]
                    end
                end
                
            elseif i==N_site
                # last step
                tmp_mps = cat(mps[i][1][:], mps[i][2][:], dims=2)
                tmp_mps_dag = cat(mps[i][1]'[:], mps[i][2]'[:], dims=2)
                tmp_O = O[i]
                tmp_site = contracted_sites[i-t-1]
                a_len0 = size(tmp_site)[1]
                a_len1 = size(mps[i-1][1])[2]
                a_len2 = size(mps[i][1])[2]
                b_len = size(O[i][1, 1, :])[1]
                arr_0 = zeros(ComplexF64, n_phys, a_len0, a_len1, b_len, b_len, a_len0)
                arr_1 = zeros(ComplexF64, n_phys, n_phys, a_len0, b_len, b_len, a_len0)
                arr = zeros(ComplexF64, a_len0, b_len, a_len0)
                @tensor begin
                    arr_0[sigma2, a0, a1, b0, b1, a0_] = tmp_site[a0, a1, b0, b1, a0_, a1_] * tmp_mps[a1_, sigma2]
                    arr_1[sigma1, sigma2, a0, b0, b1, a0_] = tmp_mps_dag[a1, sigma1] * arr_0[sigma2, a0, a1, b0, b1, a0_]
                    arr[a0, b0, a0_] = arr_1[sigma1, sigma2, a0, b0, b1, a0_] * tmp_O[sigma1, sigma2, b1]
                end        
            else
                tmp_mps = cat(mps[i][1], mps[i][2], dims=3)
                tmp_mps_dag = cat(mps[i][1]', mps[i][2]', dims=3)
                tmp_O = O[i]
                tmp_site = contracted_sites[i-t-1]
                a_len0 = size(tmp_site)[1]
                a_len1 = size(mps[i-1][1])[2]
                a_len2 = size(mps[i][1])[2]
                b_len = size(O[i][1, 1, :, :])[1]
                arr_0 = zeros(ComplexF64, n_phys, a_len0, a_len1, b_len, b_len, a_len0, a_len2)
                arr_1 = zeros(ComplexF64, n_phys, n_phys, a_len0, a_len2, b_len, b_len, a_len0, a_len2)
                arr = zeros(ComplexF64, a_len0, a_len2, b_len, b_len, a_len0, a_len2)
                @tensor begin
                    arr_0[sigma2, a0, a1, b0, b1, a0_, a2_] = tmp_site[a0, a1, b0, b1, a0_, a1_] * tmp_mps[a1_, a2_, sigma2]
                    arr_1[sigma1, sigma2, a0, a2, b0, b1, a0_, a2_] = tmp_mps_dag[a2, a1, sigma1] * arr_0[sigma2, a0, a1, b0, b1, a0_, a2_]
                    arr[a0, a2, b0, b2, a0_, a2_] = arr_1[sigma1, sigma2, a0, a2, b0, b1, a0_, a2_] * tmp_O[sigma1, sigma2, b1, b2]
                end
            end
            
            push!(contracted_sites, arr)
        end
        return contracted_sites[N_site - t]
    end

    function L_expression(mps::Array{Any,1}, O::Array{Any,1}, t::Int64)
        # t: target site, sites 1 to t-1 are merged to L expression
        N_site = size(mps)[1]
        n_phys = 2
        contracted_sites = []
        for i = 1:t-1
            if i==1
                tmp_mps = cat(mps[i][1][:], mps[i][2][:], dims=2)
                tmp_mps_dag = cat(mps[i][1]'[:], mps[i][2]'[:], dims=2)
                tmp_O = O[i]
                a_len = size(mps[i][1])[2]
                b_len = size(O[i][1, 1, :, :])[1]
                arr = zeros(ComplexF64, a_len, b_len, a_len)
                @tensor begin
                    arr[a0, b0, a0_] = tmp_mps_dag[a0, sigma1] * tmp_O[sigma1, sigma2, b0] * tmp_mps[a0_, sigma2]
                end
            else
                tmp_mps = cat(mps[i][1], mps[i][2], dims=3)
                tmp_mps_dag = cat(mps[i][1]', mps[i][2]', dims=3)
                tmp_O = O[i]
                tmp_site = contracted_sites[i-1]
                a_len0 = size(tmp_site)[1]
                a_len1 = size(mps[i-1][1])[2]
                a_len2 = size(mps[i][1])[2]
                b_len = size(O[i][1, 1, :, :])[1]
                arr_0 = zeros(ComplexF64, n_phys, a_len1, b_len, a_len2)
                arr_1 = zeros(ComplexF64, n_phys, n_phys, a_len2, b_len, a_len2)
                arr = zeros(ComplexF64, a_len2, b_len, a_len2)
                @tensor begin
                    arr_0[sigma2, a1, b1, a2_] = tmp_site[a1, b1, a1_] * tmp_mps[a1_, a2_, sigma2]
                    arr_1[sigma1, sigma2, a2, b1, a2_] = tmp_mps_dag[a2, a1, sigma1] * arr_0[sigma2, a1, b1, a2_]
                    arr[a2, b2, a2_] = arr_1[sigma1, sigma2, a2, b1, a2_] * tmp_O[sigma1, sigma2, b1, b2]
                end       
            end
            
            push!(contracted_sites, arr)
        end
        return contracted_sites[t-1]
    end

    function left_norm_for_2_sites(mps::Array{Any,1}, t::Int64, D::Int64)
        # t: target, site to be left normalized
        mps_ = copy(mps)
        site_1 = cat(mps[t][1], mps[t][2], dims=1)
        site_2 = cat(mps[t+1][1], mps[t+1][2], dims=2)
        mixed_site = site_1 * site_2
        A, M = SVD_L(mixed_site, D)
        col = convert(Int64, size(A, 1) / 2)
        mps_[t] = [A[1:col, :], A[(col+1):col*2, :]]
        col2 = convert(Int64, size(M, 2) / 2)
        mps_[t+1] = [M[:, 1:col2], M[:, (col2+1):col2*2]]
        return mps_
    end

    function right_norm_for_2_sites(mps::Array{Any,1}, t::Int64, D::Int64)
        # t: target, site to be right normalized
        mps_ = copy(mps)
        site_1 = cat(mps[t-1][1], mps[t-1][2], dims=1)
        site_2 = cat(mps[t][1], mps[t][2], dims=2)
        mixed_site = site_1 * site_2
        M, B = SVD_R(mixed_site, D)
        col = convert(Int64, size(M, 1) / 2)
        mps_[t-1] = [M[1:col, :], M[(col+1):col*2, :]]
        col2 = convert(Int64, size(B, 2) / 2)
        mps_[t] = [B[:, 1:col2], B[:, (col2+1):col2*2]]
        return mps_
    end

    function left_most_site_update(mps_::Array{Any,1}, O::Array{Any,1}, t::Int64)
        R = R_expression(mps_, O, t)
        tmp_O = O[t]
        H = zeros(ComplexF64, size(R, 1), size(R, 3), size(tmp_O, 1), size(tmp_O, 2))
        @tensor begin
            H[sigma1, a0, sigma2, a0_] = tmp_O[sigma1, sigma2, b0] * R[a0, b0, a0_]
        end
        H_ = zeros(ComplexF64, size(tmp_O, 1) * size(R, 1), size(tmp_O, 2) * size(R, 3))
        for i=1:size(H, 1), j=1:size(H, 2), k=1:size(H, 3), l=1:size(H, 4)
            H_[(i-1)*size(H, 2) + j, (k-1)*size(H, 4) + l] = H[i,j,k,l]
        end
        v = eigvecs(H_)[:, 1]
        d = convert(Int64, size(v, 1) / 2)
        mps_[t][1][:] = v[1:d]
        mps_[t][2][:] = v[d+1:2*d]
        mps_ = left_norm_for_2_sites(mps_, t, D)
        return mps_
    end
    
    function mid_site_update(mps_::Array{Any,1}, O::Array{Any,1}, t::Int64)
        R = R_expression(mps_, O, t)
        L = L_expression(mps_, O, t)
        tmp_O = O[t]
        H = zeros(ComplexF64, size(tmp_O, 1), size(L, 1), size(R, 1), size(tmp_O, 2), size(L, 3), size(R, 3))
        @tensor begin
            H[sigma1, a0, a1, sigma2, a0_, a1_] = L[a0, b0, a0_] * tmp_O[sigma1, sigma2, b0, b1] * R[a1, b1, a1_]
        end
        H_ = zeros(ComplexF64, size(tmp_O, 1) * size(L, 1) * size(R, 1), size(tmp_O, 2) * size(L, 3) * size(R, 3))
        for i=1:size(H, 1), j=1:size(H, 2), k=1:size(H, 3), l=1:size(H, 4), m=1:size(H, 5), n=1:size(H, 6)
            H_[(i-1)*size(H, 2)*size(H, 3) + (j-1)*size(H, 3) + k, (l-1)*size(H, 5)*size(H, 6) + (m-1)*size(H, 6) + n] = H[i,j,k,l,m,n]
        end
        v = eigvecs(H_)[:, 1]
        d = convert(Int64, size(v, 1) / 2)
        M_1 = transpose(reshape(v[1:d], size(transpose(mps_[t][1]))))
        M_2 = transpose(reshape(v[d+1:2*d], size(transpose(mps_[t][1]))))
        mps_[t][1][:, :] = M_1
        mps_[t][2][:, :] = M_2
        return mps_
    end
        
    function right_most_site_update(mps_::Array{Any,1}, O::Array{Any,1}, t::Int64)
        L = L_expression(mps_, O, t)
        tmp_O = O[t]
        H = zeros(ComplexF64, size(L, 1), size(L, 3), size(tmp_O, 1), size(tmp_O, 2))
        @tensor begin
            H[sigma1, a0, sigma2, a0_] = tmp_O[sigma1, sigma2, b0] * L[a0, b0, a0_]
        end
        H_ = zeros(ComplexF64, size(tmp_O, 1) * size(L, 1), size(tmp_O, 2) * size(L, 3))
        for i=1:size(H, 1), j=1:size(H, 2), k=1:size(H, 3), l=1:size(H, 4)
            H_[(i-1)*size(H, 2) + j, (k-1)*size(H, 4) + l] = H[i,j,k,l]
        end
        v = eigvecs(H_)[:, 1]
        d = convert(Int64, size(v, 1) / 2)
        mps_[t][1][:] = v[1:d]
        mps_[t][2][:] = v[d+1:2*d]
        return mps_
    end

    function iterative_ground_state_search(mps::Array{Any,1}, O::Array{Any,1}, D::Int64)
        hist = []
        N = size(mps, 1)
        mps_ = copy(mps)
        push!(hist, expectation(mps_, O))
        for t = 1:N
            if t == 1
                mps_ = left_most_site_update(mps_, O, t)
                mps_ = left_norm_for_2_sites(mps_, t, D)  
            elseif t == N
                mps_ = right_most_site_update(mps_, O, t)
            else
                mps_ = mid_site_update(mps_, O, t)
                mps_ = left_norm_for_2_sites(mps_, t, D)
            end
            push!(hist, expectation(mps_, O))
        end
        
        for t = N-1:-1:1
            mps_ = right_norm_for_2_sites(mps_, t+1, D)
            if t == 1
                mps_ = left_most_site_update(mps_, O, t)
            else
                mps_ = mid_site_update(mps_, O, t)
            end  
            push!(hist, expectation(mps_, O))
        end
        return (mps_, hist)
    end


    ## ---------- 2 site DMRG ---------- ##
    function left_norm_for_2_sites_2DMGR(mps::Array{Any,1}, t::Int64, D::Int64)
        # t: target, site to be left normalized
        mps_ = copy(mps)
        print("mps_ :")
        println(mps_)
        site_1 = cat(mps[t][1], mps[t][2], dims=1)
        site_2 = cat(mps[t+1][1], mps[t+1][2], dims=2)
        mixed_site = site_1 * site_2
        print("mixed_site :")
        println(mixed_site)
        A, M = SVD_L(mixed_site, D)
        print("A :")
        println(A)
        print("M :")
        println(M)
        col = convert(Int64, size(A, 1) / 2)
        print("size(A, 1) / 2 :")
        println(size(A, 1) / 2)
        mps_[t] = [A[1:col, :], A[(col+1):col*2, :]]
        print("mps_[t]")
        println(mps_[t])
        col2 = convert(Int64, size(M, 2) / 2)
        mps_[t+1] = [M[:, 1:col2], M[:, (col2+1):col2*2]]
        print("mps_[t+1]")
        println(mps_[t+1])
        return mps_
    end

    function right_sweep_first_2site(mps_::Array{Any,1}, O::Array{Any,1}, t::Int64)
        R = R_expression(mps_, O, t+1)
        tmp_O = O[t]
        tmp_O2 = O[t+1]
        tmp_mps = cat(mps_[t][1][:], mps_[t][2][:], dims=2)
        tmp_mps2 = cat(mps_[t][1], mps_[t][2], dims=3)
        H = zeros(ComplexF64, size(tmp_O, 1), size(tmp_O2, 1),　size(R, 1), size(tmp_O, 2), size(tmp_O2, 2), size(R, 3))
        HB = zeros(ComplexF64, size(tmp_O, 2), size(tmp_O2, 2), size(R, 3))
        @tensor begin
            H[sigma1_1, sigma1_2, a0, sigma2_1, sigma2_2, a0_] = tmp_O[sigma1_1, sigma2_1, b1] * tmp_O2[sigma1_2, sigma2_2, b1, b0] * R[a0, b0, a0_]
            # HB[sigma2_1, sigma2_2, a0_] = tmp_mps[a1, sigma1_1] * tmp_O[sigma1_1, sigma2_1, b1] * tmp_mps2[a0, a1, sigma1_2] * tmp_O2[sigma1_2, sigma2_2, b0, b1] * R[a0, b0, a0_]
        end        
        H_ = zeros(ComplexF64, size(tmp_O, 1) * size(tmp_O2, 1) * size(R, 1), size(tmp_O, 2) * size(tmp_O2, 2) * size(R, 3))
        # HB_ = zeros(ComplexF64, size(tmp_O, 2) * size(tmp_O2, 2) * size(R, 3))
        for i=1:size(H, 1), j=1:size(H, 2), k=1:size(H, 3), l=1:size(H, 4), m=1:size(H, 5), n=1:size(H, 6)
            H_[(i-1)*size(H, 2)*size(H, 3) + (j-1)*size(H, 3) + k, (l-1)*size(H, 5)*size(H, 6) + (m-1)*size(H, 6) + n] = H[i,j,k,l,m,n]
            # HB_[(l-1)*size(H, 5)*size(H, 6) + (m-1)*size(H, 6) + n] = HB[l,m,n]
        end
        v = eigvecs(H_)[:, 1]
        print("H_")
        println(H_)
        print("v")
        println(v)
        println("E0__")
        println(eigvals(H_))
        U= MPS(v,1e-5)
        mps_[t] = U[1]
        mps_[t+1] = U[2]
        return mps_
    end

    function right_sweep_2site(mps_::Array{Any,1}, O::Array{Any,1}, t::Int64)
        R = R_expression(mps_, O, t+1)
        L = L_expression(mps_, O, t)
        tmp_O = O[t]
        tmp_O2 = O[t+1]
        tmp_mps = cat(mps_[t][1], mps_[t][2], dims=3)
        tmp_mps2 = cat(mps_[t+1][1], mps_[t+1][2], dims=3)
        H = zeros(ComplexF64, size(tmp_O, 1), size(tmp_O2, 1), size(L, 1), size(R, 1), size(tmp_O, 2), size(tmp_O2, 2), size(L, 3), size(R, 3))
        @tensor begin
            H[sigma1_1, sigma1_2, a0, a1, sigma2_1, sigma2_2, a0_, a1_] = L[a0, b3, a0_] * tmp_O[sigma1_1, sigma2_1, b3, b2] * tmp_O[sigma1_2, sigma2_2, b2, b1] * R[a1, b1, a1_]
        end
        H_ = zeros(ComplexF64, size(tmp_O, 1) * size(tmp_O2, 1) * size(L, 1) * size(R, 1), size(tmp_O, 2) * size(tmp_O2, 2) * size(L, 3) * size(R, 3))
        for i=1:size(H, 1), j=1:size(H, 2), k=1:size(H, 3), l=1:size(H, 4), m=1:size(H, 5), n=1:size(H, 6), o=1:size(H, 7), p=1:size(H, 8)
            H_[(i-1)*size(H, 2)*size(H, 3)*size(H, 4) + (j-1)*size(H, 3)*size(H, 4) + (k-1)*size(H, 4) + l, (m-1)*size(H, 6)*size(H, 7)*size(H, 8) + (n-1)*size(H, 7)*size(H, 8) + (o-1)*size(H, 8) + p] = H[i,j,k,l,m,n,o,p]
        end
        v = eigvecs(H_)[:, 1]
        print("H_")
        println(H_)
        print("v")
        println(v)
        println("E0__")
        println(eigvals(H_))
        U = MPS(v,1e-5)
        mps_[t] = U[1]
        mps_[t+1] = U[2]
        return mps_
    end

    function right_sweep_final_2site(mps_::Array{Any,1}, O::Array{Any,1}, t::Int64)
        L = L_expression(mps_, O, t)
        tmp_O = O[t]
        tmp_O2 = O[t+1]
        H = zeros(ComplexF64, size(tmp_O, 1), size(tmp_O2, 1),　size(L, 1), size(tmp_O, 2), size(tmp_O2, 2), size(L, 3))
        @tensor begin
            H[sigma1_1, sigma1_2, a0, sigma2_1, sigma2_2, a0_] =L[a0, b2, a0_] * tmp_O[sigma1_1, sigma2_1, b2, b1] * tmp_O2[sigma1_2, sigma2_2, b1]
        end
        H_ = zeros(ComplexF64, size(tmp_O, 1) * size(tmp_O2, 1) * size(L, 1), size(tmp_O, 2) * size(tmp_O2, 2) * size(L, 3))
        for i=1:size(H, 1), j=1:size(H, 2), k=1:size(H, 3), l=1:size(H, 4), m=1:size(H, 5), n=1:size(H, 6)
            H_[(i-1)*size(H, 2)*size(H, 3) + (j-1)*size(H, 3) + k, (l-1)*size(H, 5)*size(H, 6) + (m-1)*size(H, 6) + n] = H[i,j,k,l,m,n]
        end
        v = eigvecs(H_)[:, 1]
        print("H_")
        println(H_)
        print("v")
        println(v)
        println("E0__")
        println(eigvals(H_))
        U = MPS(v,1e-5)
        mps_[t] = U[1]
        mps_[t+1] = U[2]
        return mps_
    end

    function left_sweep_first_2site(mps_::Array{Any,1}, O::Array{Any,1}, t::Int64)
        L = L_expression(mps_, O, t-1)
        tmp_O = O[t]
        tmp_O2 = O[t-1]
        H = zeros(ComplexF64, size(tmp_O, 1), size(tmp_O2, 1),　size(L, 1), size(tmp_O, 2), size(tmp_O2, 2), size(L, 3))
        @tensor begin
            H[sigma1_1, sigma1_2, a0, sigma2_1, sigma2_2, a0_] = tmp_O[sigma1_1, sigma2_1, b2] * tmp_O2[sigma1_2, sigma2_2, b2, b1] * L[a0, b1, a0_]
        end
        H_ = zeros(ComplexF64, size(tmp_O, 1) * size(tmp_O2, 1) * size(L, 1), size(tmp_O, 2) * size(tmp_O2, 2) * size(L, 3))
        for i=1:size(H, 1), j=1:size(H, 2), k=1:size(H, 3), l=1:size(H, 4), m=1:size(H, 5), n=1:size(H, 6)
            H_[(i-1)*size(H, 2)*size(H, 3) + (j-1)*size(H, 3) + k, (l-1)*size(H, 5)*size(H, 6) + (m-1)*size(H, 6) + n] = H[i,j,k,l,m,n]
        end
        v = eigvecs(H_)[:, 1]
        print("H_")
        println(H_)
        print("v")
        println(v)
        println("E0__")
        println(eigvals(H_))
        U= MPS(v,1e-5)
        mps_[t] = U[1]
        mps_[t-1] = U[2]
        return mps_
    end
    
    function left_sweep_2site(mps_::Array{Any,1}, O::Array{Any,1}, t::Int64)
        R = R_expression(mps_, O, t)
        L = L_expression(mps_, O, t-1)
        tmp_O = O[t]
        tmp_O2 = O[t-1]
        H = zeros(ComplexF64, size(tmp_O, 1), size(tmp_O2, 1), size(L, 1), size(R, 1), size(tmp_O, 2), size(tmp_O2, 2), size(L, 3), size(R, 3))
        @tensor begin
            H[sigma1_1, sigma1_2, a0, a1, sigma2_1, sigma2_2, a0_, a1_] = L[a0, b3, a0_] * tmp_O2[sigma1_1, sigma2_1, b3, b2] * tmp_O[sigma1_2, sigma2_2, b2, b1] * R[a1, b1, a1_]
        end
        H_ = zeros(ComplexF64, size(tmp_O, 1) * size(tmp_O2, 1) * size(L, 1) * size(R, 1), size(tmp_O, 2) * size(tmp_O2, 2) * size(L, 3) * size(R, 3))
        for i=1:size(H, 1), j=1:size(H, 2), k=1:size(H, 3), l=1:size(H, 4), m=1:size(H, 5), n=1:size(H, 6), o=1:size(H, 7), p=1:size(H, 8)
            H_[(i-1)*size(H, 2)*size(H, 3)*size(H, 4) + (j-1)*size(H, 3)*size(H, 4) + (k-1)*size(H, 4) + l, (m-1)*size(H, 6)*size(H, 7)*size(H, 8) + (n-1)*size(H, 7)*size(H, 8) + (o-1)*size(H, 8) + p] = H[i,j,k,l,m,n,o,p]
        end
        v = eigvecs(H_)[:, 1]
        print("H_")
        println(H_)
        print("v")
        println(v)
        println("E0__")
        println(eigvals(H_))
        U = MPS(v,1e-5)
        # HB_ = HB_ / norm(HB_)
        # U= MPS(HB_,1e-5)
        mps_[t] = U[1]
        mps_[t-1] = U[2]
        return mps_
    end
        
    function left_sweep_final_2site(mps_::Array{Any,1}, O::Array{Any,1}, t::Int64)
        R = R_expression(mps_, O, t)
        tmp_O = O[t]
        tmp_O2 = O[t-1]
        H = zeros(ComplexF64, size(tmp_O, 1), size(tmp_O2, 1),　size(R, 1), size(tmp_O, 2), size(tmp_O2, 2), size(R, 3))
        @tensor begin
            H[sigma1_1, sigma1_2, a0, sigma2_1, sigma2_2, a0_] = tmp_O2[sigma1_1, sigma2_1, b0] * tmp_O[sigma1_2, sigma2_2, b0, b1] * R[a0, b1, a0_]
        end
        H_ = zeros(ComplexF64, size(tmp_O, 1) * size(tmp_O2, 1) * size(R, 1), size(tmp_O, 2) * size(tmp_O2, 2) * size(R, 3))
        for i=1:size(H, 1), j=1:size(H, 2), k=1:size(H, 3), l=1:size(H, 4), m=1:size(H, 5), n=1:size(H, 6)
            H_[(i-1)*size(H, 2)*size(H, 3) + (j-1)*size(H, 3) + k, (l-1)*size(H, 5)*size(H, 6) + (m-1)*size(H, 6) + n] = H[i,j,k,l,m,n]
        end
        v = eigvecs(H_)[:, 1]
        print("H_")
        println(H_)
        print("v")
        println(v)
        println("E0__")
        println(eigvals(H_))
        U = MPS(v,1e-5)
        mps_[t] = U[1]
        mps_[t-1] = U[2]
        return mps_
    end

    function iterative_ground_state_search_2site(mps::Array{Any,1}, O::Array{Any,1}, D::Int64)
        hist = []
        N = size(mps, 1)
        mps_ = copy(mps)
        push!(hist, expectation(mps_, O))

        for ittr=1:3
            for t = 1:N-1
                println(t)
                # println(mps)
                if t == 1
                    mps_ = right_sweep_first_2site(mps_, O, t)
                    # mps_ = left_norm_for_2_sites_2DMGR(mps_, t, D)  
                elseif t == N-1
                    mps_ = right_sweep_final_2site(mps_, O, t)
                else
                    mps_ = right_sweep_2site(mps_, O, t)
                    # mps_ = left_norm_for_2_sites_2DMGR(mps_, t, D)
                end
                # println("after")
                # println(mps)
                println("mps")
                println(mps_)
                E = expectation(mps_, O)
                println("E0_lim")
                println(E)
                push!(hist, E)
            end
        
            for t = N:-1:2
                # mps_ = right_norm_for_2_sites(mps_, t+1, D)
                println(t)
                if t == N
                    mps_ = left_sweep_first_2site(mps_, O, t)
                elseif t == 2
                    mps_ = left_sweep_final_2site(mps_, O, t)
                else
                    mps_ = left_sweep_2site(mps_, O, t)
                end  
                println("mps")
                println(mps_)
                E = expectation(mps_, O)
                push!(hist, E)
            end
        end
        return (mps_, hist)
    end



    # ------------ init random MPS ------------ #
    function init_rand_MPS(N::Int64, D::Int64, normalize::Char='l')
        mps = []
        mps2 = []
        half_N = convert(Int64, floor(N/2))
        for i in 1:half_N
            l = 2^(i-1)
            mps_1 = []
            mps_2 = []
            for j in 1:2
                arr1 = rand(ComplexF64, (min(l, D), min(l*2, D))) / min(l*2, D)
                arr2 = rand(ComplexF64, (min(l*2, D), min(l, D))) / min(l, D)
                push!(mps_1, arr1)
                push!(mps_2, arr2)
            end
            push!(mps, mps_1)
            push!(mps2, mps_2)
        end
        
        if N%2 == 1
            mps_ = []
            for j in 1:2
                arr = rand(ComplexF64, (min(2^half_N, D), min(2^half_N, D))) / min(2^half_N, D)
                push!(mps_, arr)
            end
            push!(mps, mps_)
        end
        
        for j in 1:half_N
            push!(mps, mps2[half_N - (j-1)])
        end
        
        for i in 1:N-1
            mps = left_norm_for_2_sites(mps, i, D)
        end
        norm = mps[N][1]' * mps[N][1] + mps[N][2]' * mps[N][2]
        mps[N][1] = mps[N][1]/sqrt(norm)
        mps[N][2] = mps[N][2]/sqrt(norm)
        if normalize == 'r'
            for i in N:-1:2
                mps = right_norm_for_2_sites(mps, i, D)
            end
        end
        return mps
    end

end