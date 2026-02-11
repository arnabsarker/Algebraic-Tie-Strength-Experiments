using CSV
using DataFrames
using ArgParse

function parse_commandline()
    s = ArgParseSettings()
    s.description = "Relabel nodes in simplicial complex CSV files."

    @add_arg_table s begin
        "input_dir"
            help = "Path to the directory containing CSV files"
            required = true
        "output_dir"
            help = "Path to save processed files (defaults to 'processed_data')"
            required = false
        "--include_clique_complex", "-c"
            help = "If set, look for and process clique_tlist.csv"
            action = :store_true  # This makes it a boolean flag
    end

    return parse_args(s)
end

function process_files(input_dir::String, output_dir::String, include_clique::Bool=false)
    
    file_configs = [
        ("edges.csv",          [:node_1, :node_2]),
        ("triangles.csv",      [:node_1, :node_2, :node_3])
    ]

    if include_clique
        push!(file_configs, ("clique_tlist.csv", [:node_1, :node_2, :node_3]))
    end

    println("Reading files from: $input_dir")
    loaded_data = [] # Will store tuples of (filename, dataframe, node_cols)

    for (fname, node_cols) in file_configs
        fpath = joinpath(input_dir, fname)
        
        if !isfile(fpath)
            error("Required file not found: $fpath")
        end

        # Read and strictly select columns
        df = CSV.read(fpath, DataFrame)
        select!(df, node_cols)
        
        push!(loaded_data, (filename=fname, data=df, cols=node_cols))
    end

    println("Building global node mapping...")
    all_nodes = Any[] # Use Any to support strings or mixed types initially

    # Iterate over loaded dataframes to collect all nodes
    for item in loaded_data
        for col in item.cols
            append!(all_nodes, item.data[!, col])
        end
    end

    # Create mapping: Unique Node Value -> Integer Index (1 to N) for Julia processing
    unique_nodes = sort(unique(all_nodes))
    node_map = Dict(val => i for (i, val) in enumerate(unique_nodes))
    
    println("Total unique nodes found: $(length(unique_nodes))")

    if !isdir(output_dir)
        mkpath(output_dir)
    end
    println("Processing and writing outputs to: $output_dir")

    for item in loaded_data
        for col in item.cols
            item.data[!, col] = [node_map[x] for x in item.data[!, col]]
        end

        out_path = joinpath(output_dir, "julia_" * item.filename)
        CSV.write(out_path, item.data)
    end

    df_labels = DataFrame(
        original_label = unique_nodes, 
        julia_index = 1:length(unique_nodes)
    )
    CSV.write(joinpath(output_dir, "julia_labels.csv"), df_labels)

    println("Done.")
end

function main()
    parsed_args = parse_commandline()
    
    input_dir = parsed_args["input_dir"]
    
    output_dir = parsed_args["output_dir"]
    if isnothing(output_dir)
        output_dir = input_dir
    end

    include_clique = parsed_args["include_clique_complex"]

    process_files(input_dir, output_dir, include_clique)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end