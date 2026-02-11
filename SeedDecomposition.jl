using Base.Threads
using CSV
using DataFrames
using DelimitedFiles
using LinearAlgebra
using LinearMaps
using IterativeSolvers
using ProgressMeter
include("./SimplicialLaplacians.jl");

function main(my_dataset, use_CC=false)
    prefix = "./Data/" * my_dataset * "/"
    
    if(use_CC)
        triangle_df = CSV.read(prefix * "clique_tlist.csv", DataFrame);
    else
        triangle_df = CSV.read(prefix * "triangles.csv", DataFrame);
    end
    edge_df = CSV.read(prefix * "edges.csv", DataFrame);


    output_prefix = "./Outputs/$(my_dataset)"
    if(!isdir(output_prefix))
        mkdir(output_prefix)
    end

    
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


    num_edge=size(elist,1)
    
        
    list_n1      = Vector{Int64}(undef, num_edge)
    list_n2      = Vector{Int64}(undef, num_edge)
    list_harm    = Vector{Float64}(undef, num_edge)
    list_grad    = Vector{Float64}(undef, num_edge)
    list_curl    = Vector{Float64}(undef, num_edge)

    @showprogress 1 "Computing Hodge Decomposition Measures..." for i in 1:num_edge
        n1 = findnz(B1[:,i])[1][1]
        #println(n1)
        n2 = findnz(B1[:,i])[1][2]


        #Put one at the position of the new edge
        b = zeros(Float64, num_edge)
        b[i] = 1.0

        # Solve system
        sol_grad = B1T*lsqr(B1T, b, atol=1e-3, btol=1e-3)
        sol_curl = B2*lsqr(B2, b, atol=1e-3, btol=1e-3)
        sol_harm = b - sol_grad - sol_curl

        list_n1[i]      = n1
        list_n2[i]      = n2
        list_harm[i]  = norm(sol_harm)
        list_grad[i]  = norm(sol_grad)
        list_curl[i]  = norm(sol_curl)
    end

    final_df = DataFrame(
        node_1 = list_n1, 
        node_2 = list_n2, 
        harm = list_harm, 
        curl = list_curl, 
        grad = list_grad, 
    )
    if(use_CC)
        CSV.write(output_prefix * "/cc_seed_decomp.csv", final_df)
    else
        CSV.write(output_prefix * "/seed_decomp.csv", final_df)
    end
end


main(string(ARGS[1]))
