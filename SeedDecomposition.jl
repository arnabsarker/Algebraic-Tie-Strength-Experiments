using Base.Threads
using CSV
using DataFrames
using DelimitedFiles
using LinearAlgebra
using LinearMaps
using IterativeSolvers
using ProgressMeter
using ArgParse

include("./SimplicialLaplacians.jl");

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "input_directory"
            help = "Directory containing input CSVs (edges.csv, triangles.csv, etc.)"
            required = true
        "output_directory"
            help = "Directory where results will be saved"
            required = true
        "--compute_clique_complex", "-c"
            help = "Flag to use clique complex files (clique_tlist.csv)"
            action = :store_true
    end

    return parse_args(s)
end

function main(input_dir, output_dir, use_CC=false)
    if use_CC
        triangle_file = joinpath(input_dir, "clique_tlist.csv")
    else
        triangle_file = joinpath(input_dir, "triangles.csv")
    end
    edge_file = joinpath(input_dir, "edges.csv")

    println("Scanning input directory: $input_dir")
    triangle_df = CSV.read(triangle_file, DataFrame)
    edge_df     = CSV.read(edge_file, DataFrame)

    if !isdir(output_dir)
        println("Creating output directory: $output_dir")
        mkpath(output_dir) 
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
    
    output_filename = use_CC ? "cc_seed_decomp.csv" : "seed_decomp.csv"
    save_file_path = joinpath(output_dir, output_filename)
    CSV.write(save_file_path, final_df)
    println("Success! Results saved to directory: $output_dir")
    println("Filename: $output_filename")
end

let
    args = parse_commandline()
    main(
        args["input_directory"], 
        args["output_directory"], 
        args["compute_clique_complex"]
    )
end