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

    println("Creating B1...")
    elist = sort(edge_matrix, dims=2)
    elist = sortslices(elist, dims=1)
    B1 = createNodeToEdgeIncidenceMatrix(elist)
    B1T = sparse(B1')

    println("Creating B2...")        
    trianglelist = sort(triangle_matrix, dims=2)
    trianglelist = sortslices(trianglelist, dims=1)
    B2 = createEdgeToTriangleIncidenceMatrix(trianglelist, elist)
    B2T = sparse(B2')

    println("Preprocessing data...")
    ntris = size(B2, 2)
    
    d2 = max.(1, vec(sum(abs.(B2), dims=2)))
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

    num_edge = size(elist, 1)
    PR_op = LinearMap(beta * I, num_edge) + LM_L1_sym
    
    list_n1      = Vector{Int64}(undef, num_edge)
    list_n2      = Vector{Int64}(undef, num_edge)
    list_harmPR  = Vector{Float64}(undef, num_edge)
    list_gradPR  = Vector{Float64}(undef, num_edge)
    list_curlPR  = Vector{Float64}(undef, num_edge)
    list_totPR   = Vector{Float64}(undef, num_edge)

    @showprogress 1 "Computing Edge PR..." for i in 1:num_edge
        nz_indices = findnz(B1[:, i])[1]
        n1 = nz_indices[1]
        n2 = nz_indices[2]

        b = zeros(Float64, num_edge)
        b[i] = 1.0
        c = D2_inv_sqrt * b

        y = cg(PR_op, (beta - 2) .* c, reltol=1e-4)
        s = D2_sqrt * y

        sol_grad_w = B1wT * lsqr(B1wT, s, atol=1e-3, btol=1e-3)
        sol_curl_w = B2wT * lsqr(B2wT, s, atol=1e-3, btol=1e-3)
        sol_harm_w = s - sol_grad_w - sol_curl_w

        list_n1[i]      = n1
        list_n2[i]      = n2
        list_harmPR[i]  = norm(sol_harm_w)
        list_gradPR[i]  = norm(sol_grad_w)
        list_curlPR[i]  = norm(sol_curl_w)
        list_totPR[i]   = norm(s)
    end

    final_df = DataFrame(
        node_1 = list_n1, 
        node_2 = list_n2, 
        harmPR = list_harmPR, 
        curlPR = list_curlPR, 
        gradPR = list_gradPR, 
        totPR  = list_totPR
    )

    output_filename = use_CC ? "cc_edgePR.csv" : "edgePR.csv"
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