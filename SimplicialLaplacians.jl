using LinearAlgebra
using SparseArrays
"""Function to create an edge list from a given adjacency matrix.

The 'edge list' consists of two vectors 'from', 'to' marking the source and target of
each node. The position within the edge-list defines the edge id.
The numbering scheme is lexicographical, i.e., edges starting at node 1 will be among
the first, e.g., (1,1) if existent is the edge with ID 1.
"""
function createEdgeListFromAdjMatrix(A, weighted= false)
    # >> we have an undirected graph, we remove the upper triangular part for consistency
    # >> the ordering as findnz operates column first!
    A = tril(A)
    to, from, value = findnz(A)
    elist = hcat(from,to)
    return elist
end


"""Function to build node-to-edge-incidence matrix from an edgelist.

Given an E x 2 edgelist in the form [from, to], create the N x E edge incidence matrix.
Here N is the number of nodes in the graph. The tail (from) of the edge will have a -1
entry in the position (from, edge_id) position; the head (to) of the edge will have a +1
entry in the position (to, edge_id).
Make sure that edges are ordered consistently and node ids are contiguous.

Output: node to edge incidence matrix B ∈ R^{N x E}
"""
function createNodeToEdgeIncidenceMatrix(elist)
    numNodes = maximum(elist)
    numEdges = size(elist,1)
    B =  sparse(elist[:,1],1:numEdges,-ones(numEdges),numNodes,numEdges) 
    B += sparse(elist[:,2],1:numEdges,+ones(numEdges),numNodes,numEdges)
    return B 
end 


"""Build the edge-to-triangle-incidence matrix from a list of (filled) triangles, and a
corresponding edge-list.

Given an T x 3 list in the form [n1, n2, n3], create the incidence matrix from edges to
triangles. The following conventions are used: an edges is always assumed to be oriented
from the endpoint with lower id to higher id (consistent with functions in this module).
A triangle is always enumerated in the form [n1,n2,n3], where ni are node ids, such that
n1<n2<n3. The orientation of the triangle is assumed to be aligned with this orientation, 
i.e., according to the first edge defining it.
The edgelist [from, to] provided is assumed to be ordered in such a way that from < to 
for all edges and that rows are lexicographically ordered
"""
function createEdgeToTriangleIncidenceMatrix(trianglelist, elist)
    # check consistency of elist
    @assert(all(elist[:,1] .< elist[:,2]))
    @assert(all(elist == sortslices(elist,dims=1)))

    # we are more lenient with the triangle-list -- if this is not ordered, we order it.
    if !all(trianglelist[:,1] .< trianglelist[:,2] .< trianglelist[:,3])
        println("Triangle list not sorted -- this will be adjusted")
        trianglelist = sort(trianglelist,2)
    end
    tlist = sortslices(trianglelist,dims=1)
    if tlist != trianglelist
        println("Triangle list not sorted -- this has been accounted for")
    end
        

    numEdges = size(elist,1)
    numTriangles = size(tlist,1)

    # initialize edge to triangle incidence
    edge_indices = Dict()
    edges = Matrix(elist')
    for ed in 1:numEdges
        edge = (edges[1, ed], edges[2, ed])
        edge_indices[edge] = ed
    end

    B2_I = Int64[]
    B2_J = Int64[]
    B2_V = Int64[]
    tris = Matrix(trianglelist')
    for t = 1:numTriangles
        i, j, k = tris[:, t]
        # (i, j) --> 1
        # (j, k) --> 1
        # (i, k) --> -1
        push!(B2_I, edge_indices[(i, j)], edge_indices[(j, k)], edge_indices[(i, k)])
        push!(B2_J, t, t, t)
        push!(B2_V, 1, 1, -1)
    end

    #=
    #B2 = spzeros(numEdges,numTriangles)
    for ed in 1:numEdges
        edge = elist[ed,:]'

        # find edges in triangle list
        # due to our ordering convention of the edges and triangles above, the first two
        # edges will always be aligned with the triangular face, while the third won't be
        index1, ~,~ = findnz(sparse(all(tlist[:,[1,2]] .== edge,dims=2))) 
        num1 = ones(Int64,length(index1))
        #B2 += sparse(ed*num1,index1,num1,numEdges,numTriangles)
        append!(B2_I, ed * num1)
        append!(B2_J, index1)
        append!(B2_V, num1)

        # index2 = LinearIndices(findall(all(trianglelist[:,[2,3]] .== edge,dims=2)))
        index2, ~,~ = findnz(sparse(all(tlist[:,[2,3]] .== edge,dims=2))) 
        num2 = ones(Int64,length(index2))
        #B2 += sparse(ed*num2,index2,num2,numEdges,numTriangles)
        append!(B2_I, ed * num2)
        append!(B2_J, index2)
        append!(B2_V, num2)

        # index3 = LinearIndices(findall(all(trianglelist[:,[1,3]] .== edge,dims=2)))
        index3, ~,~ = findnz(sparse(all(tlist[:,[1,3]] .== edge,dims=2))) 
        num3 = ones(Int64,length(index3))
        #B2 += sparse(ed*num3,index3,-num3,numEdges,numTriangles)
        append!(B2_I, ed * num3)
        append!(B2_J, index3)
        append!(B2_V, num3)
    end
    =#
    
    return sparse(B2_I, B2_J, B2_V, numEdges, numTriangles)
end

""" Find all triangles in a graph and create a triangle-list.

Given a graph with adjacency matrix A find all triangles and produce a list of triangles
in the form [n1, n2, n3], where ni is a node index"""
function createTriangleListFromAjdacencyMatrix(A)
    # get all edges (2-tuples) of nodes present in the graph
    elist = createEdgeListFromAdjMatrix(A)
    numEdges = size(elist,1)
    numNodes = size(A,1)
    Triangles = Set()
    for ed in 1:numEdges
        # consider the elist to be in form i, j
        i, j = elist[ed,:]
        # neigbors of i are all nodes k that appears in the list either as 
        # k: (i,k) <-> elist[elist[:,1].==i,2] 
        # or k: (k,i) <-> elist[elist[:,2].==i,1]
        neighborsOfFirstNode = vcat(elist[elist[:,1].==i,2], elist[elist[:,2].==i,1])
        # same for node j
        neighborsOfSecondNode = vcat(elist[elist[:,1].==j,2], elist[elist[:,2].==j,1])

        # find intersection between those neighbors => triangle
        jointNeighbors = intersect(neighborsOfFirstNode,neighborsOfSecondNode)
        for t in jointNeighbors
            sorted_triangle = sort([i,j,t])
            Triangles = union(Triangles,Set([sorted_triangle]))
        end
    end
    numTriangles = length(Triangles) 
    tlistTemp = [t for t in Triangles]
    tlist = zeros(Int64,numTriangles,3)
    for i in 1:numTriangles
        tlist[i,:] = tlistTemp[i]
    end
    tlist = sortslices(tlist,dims=1)
    return tlist
end


""" Find all triangles in a graph and create a triangle-list.

Given a graph with adjacency matrix A find all triangles and produce a list of triangles
in the form [n1, n2, n3], where ni is a node index"""
function createTriangleListFromEdgeList(elist)
    # get all edges (2-tuples) of nodes present in the graph
    numEdges = size(elist,1)
    Triangles = Set()
    for ed in 1:numEdges
        # consider the elist to be in form i, j
        i, j = elist[ed,:]
        # neigbors of i are all nodes k that appears in the list either as 
        # k: (i,k) <-> elist[elist[:,1].==i,2] 
        # or k: (k,i) <-> elist[elist[:,2].==i,1]
        neighborsOfFirstNode = vcat(elist[elist[:,1].==i,2], elist[elist[:,2].==i,1])
        # same for node j
        neighborsOfSecondNode = vcat(elist[elist[:,1].==j,2], elist[elist[:,2].==j,1])

        # find intersection between those neighbors => triangle
        jointNeighbors = intersect(neighborsOfFirstNode,neighborsOfSecondNode)
        for t in jointNeighbors
            sorted_triangle = sort([i,j,t])
            Triangles = union(Triangles,Set([sorted_triangle]))
        end
    end
    numTriangles = length(Triangles) 
    tlistTemp = [t for t in Triangles]
    tlist = zeros(Int64,numTriangles,3)
    for i in 1:numTriangles
        tlist[i,:] = tlistTemp[i]
    end
    tlist = sortslices(tlist,dims=1)
    return tlist
end

"""Function to create an L1 Laplacian based on a graph, where triangles are automatically
filled"""
function createL1Laplacian(A)
    elist = createEdgeListFromAdjMatrix(A)
    tlist = createTriangleListFromAjdacencyMatrix(A)

    B1 = createNodeToEdgeIncidenceMatrix(elist)
    B2 = createEdgeToTriangleIncidenceMatrix(tlist,elist)
    L1 = B1'*B1 + B2*B2'
    return L1
end

"""Function to create a weighted L1 Laplacian based on a graph, where triangles are 
automatically filled"""
function createWeightedL1Laplacian(A; mode="RW")
    elist = createEdgeListFromAdjMatrix(A)
    tlist = createTriangleListFromAjdacencyMatrix(A)

    B1 = createNodeToEdgeIncidenceMatrix(elist)
    B2 = createEdgeToTriangleIncidenceMatrix(tlist,elist)

    # d1 == number of upper adjacent faces; if no upper adjacent face keep weight at 1
    d1 = sum(abs.(B2),dims=2)
    D1 = Diagonal(max.(1,d1[:]))
    D1inv = Diagonal(1 ./ max.(1,d1[:]))

    # d0 weight is the weighted degree of the nodes 
    # (edge weight is the number of upper adjacent faces)
    d0weighted = sum(abs.(B1*D1),dims=2)
    D0weighted= Diagonal(d0weighted[:])
    D0weightedinv = Diagonal(1 ./ d0weighted[:])

    # assemble normalized Laplacian
    L1 = D1*B1'*1/2*D0weightedinv*B1 + B2*1/3*B2'*D1inv
    if mode == "sym"
        L1 = D1inv.^(1/2) * L1 * D1.^(1/2)
    end
    return L1, elist
end

"""Function to create a weighted L1 Laplacian based on the corresponding incidence 
matrices"""
function createWeightedL1Laplacian(B1, B2; mode="RW")

    # d1 == number of upper adjacent faces; if no upper adjacent face keep weight at 1
    d1 = sum(abs.(B2),dims=2)
    D1 = Diagonal(max.(1,d1[:]))
    D1inv = Diagonal(1 ./ max.(1,d1[:]))

    # d0 weight is the weighted degree of the nodes 
    # (edge weight is the number of upper adjacent faces)
    d0weighted = sum(abs.(B1*D1),dims=2)
    D0weighted= Diagonal(d0weighted[:])
    D0weightedinv = Diagonal(1 ./ d0weighted[:])

    # assemble normalized Laplacian
    L1 = D1*B1'*1/2*D0weightedinv*B1 + B2*1/3*B2'*D1inv
    if mode == "sym"
        L1 = D1inv.^(1/2) * L1 * D1.^(1/2)
    end
    return L1
end

"""Function to implement inverse simplicial page-rank operator for edges
Compute the (sparse matrix) operator:
        beta*eye + L_k
The simplicial page-rank vector can the be computed from the inverse of this matrix.
For instance, if we assume a personalization vector e, then the personalized SC PageRank
is simply ( [beta*eye+L_k]^T )^-1 e"""
function inverseSimplicialPageRankOperator(beta, A)
    L1, _ = createWeightedL1Laplacian(A)
    num_edge=size(L1,1)
    IPR = beta*sparse(I,num_edge,num_edge) + L1
    return IPR
end

"""
Compute the projection of a vector onto a particular subspace, spanned by a set
of vectors
   
   Input: VecToProj --- the (set of) vectors we want to project (n x k array)
          SpanVec ---   the set of vectors spanning the subspace to project onto
"""
function projectVectorToSubspace(VecToProj,SpanVec)
    Proj = SpanVec*pinv(Matrix(SpanVec))* VecToProj
    return Proj
end
