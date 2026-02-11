using Base.Threads
using CSV
using DataFrames
using DelimitedFiles
using LinearAlgebra
using LinearMaps
using IterativeSolvers
include("./SimplicialLaplacians.jl");

function main(my_dataset)
    
    # Flag for if we are using the clique complex of the data or not
    CC = false
    
    prefix = "../Processed-Data/" * my_dataset * "/"
    suffix = ""
    
    if(CC)
        triangle_df = CSV.read(prefix * "clique_tlist" * suffix * ".csv", DataFrame);
    else
        triangle_df = CSV.read(prefix * "triangles" * suffix * ".csv", DataFrame);
    end
    edge_df = CSV.read(prefix * "edges" * suffix * ".csv", DataFrame);
    
    triangle_matrix = Matrix(triangle_df[:, 1:3]);
    edge_matrix = Matrix(edge_df[:, 1:2]);

    println("creating B1...")
    elist = sort(edge_matrix, dims=2)
    elist = sortslices(elist,dims=1)
    B1 = createNodeToEdgeIncidenceMatrix(elist)
    B1T = sparse(B1')

    println("creating B2...")        
    trianglelist = sort(triangle_matrix, dims=2)
    trianglelist = sortslices(trianglelist,dims=1)
    B2 = createEdgeToTriangleIncidenceMatrix(trianglelist, elist)
    B2T = sparse(B2')

    println("preprocessing...")
    nedges = size(elist,1)
    ntris = size(B2,2)
    
    d2 = max.(1, vec(sum(abs.(B2),dims=2)))
    D2 = Diagonal(d2)
    D2_inv = Diagonal(1.0 ./ d2)
    D2_sqrt = sqrt.(D2)
    D2_inv_sqrt = sqrt.(D2_inv)

    B1wT = D2 * B1T
    B2wT = B2

    d1 = vec(2 .* (abs.(B1) * d2))
    D1 = Diagonal(d1)
    D1_inv = Diagonal(1.0 ./ d1)

    d3 = ones(Float64, ntris) ./ 3.0
    D3 = Diagonal(d3)

    M1 = D2_inv_sqrt * D2 * B1T * D1_inv
    M2 = B1 * D2_sqrt
    M3 = D2_inv_sqrt * B2
    M4 = D3 * B2T * D2_inv * D2_sqrt

    beta = 2.5
    LM_L1_sym = LinearMap(M1) * LinearMap(M2) + LinearMap(M3) * LinearMap(M4)

    num_edge=size(elist,1)
    
    # Compute for different values of teleportation parameter, to check robustness
    for beta = [2.5, 2.01, 0.01]
        
        PR_op = LinearMap(beta * I, num_edge) + LM_L1_sym
        list_n1 = Int64[]
        list_n2 = Int64[]
        list_harmPR = Float64[]
        list_gradPR = Float64[]
        list_curlPR = Float64[]
        list_curlfreePR = Float64[]
        list_totPR = Float64[]

        for i in range(1,stop = num_edge)
            n1 = findnz(B1[:,i])[1][1]
            #println(n1)
            n2 = findnz(B1[:,i])[1][2]


            #Put one at the position of the new edge
            b = zeros(Float64, nedges)
            b[i] = 1.0
            c = D2_inv_sqrt * b

            # Solve system
            y = cg(PR_op, (beta-2) .* c, tol=1e-4)
            s = D2_sqrt * y

            sol_grad_w = B1wT*lsqr(B1wT, s, atol=1e-3, btol=1e-3)
            sol_curl_w = B2wT*lsqr(B2wT, s, atol=1e-3, btol=1e-3)
            sol_harm_w = s - sol_grad_w - sol_curl_w
            sol_curl_free_w = sol_grad_w + sol_harm_w

            harm_norm = sqrt.(sol_harm_w'*sol_harm_w)
            curl_norm = sqrt.(sol_curl_w'*sol_curl_w)
            grad_norm = sqrt.(sol_grad_w'*sol_grad_w)
            curl_free_norm = sqrt.(sol_curl_free_w'*sol_curl_free_w)
            tot_norm = sqrt.(s' * s)
            push!(list_n1,n1)
            push!(list_n2,n2)
            push!(list_harmPR,harm_norm)
            push!(list_gradPR,grad_norm)
            push!(list_curlPR,curl_norm)
            push!(list_curlfreePR, curl_free_norm)
            push!(list_totPR,tot_norm)
        end

        final_arr = hcat(list_n1,list_n2,list_harmPR,list_curlPR, list_gradPR, list_curlfreePR, list_totPR)
        final_df = DataFrame(final_arr, [:node_1, :node_2, :harmPR, :curlPR, :gradPR, :curlfreePR, :totPR])

        output_prefix = "../EdgePR/$(my_dataset)"
        if(!isdir(output_prefix))
            mkdir(output_prefix)
        end

        if(CC)
            CSV.write(output_prefix * "/cc_edgePR_unsym_" * string(beta) * ".csv", final_df)
        else
            CSV.write(output_prefix * "/edgePR_unsym_" * string(beta) * ".csv", final_df)
        end
    end
end


main(string(ARGS[1]))
