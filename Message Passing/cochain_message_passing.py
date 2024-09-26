import torch 
from Raise import RaisedGraph



class CochainMessagePassing(torch.nn.Module):

    '''This class implements a cochain message passing layer. It takes in an n-dimensional simplicial complex and performs message passing on the 
    adjacency matrix. 
    Parameters:
        1. '''
    def __init__(self, 
                 adj_matrix, 
                 node_dimension, 
                 simplex_dimension, 
                 dimension, 

                 dropout,
                 alpha,
                 num_heads,

                 boundary_adjacency = True, 
                 co_boundary_adjacency = True, 
                 lower_adjacencies = True, 
                 upper_adjacencies = True
                 
                 ) -> None:
        super(CochainMessagePassing, self).__init__()
        
        self.node_dimension = node_dimension
        self.simplex_dimension = simplex_dimension
        self.dimension = dimension
        self.adj_matrix = adj_matrix

        self.raised_graph = RaisedGraph(self.adj_matrix)

        self.dropout = dropout # dropout rate
        self.alpha = alpha  # parameter for leaky relu

        self.boundary_adjacency = boundary_adjacency
        self.co_boundary_adjacency = co_boundary_adjacency
        self.lower_adjacencies = lower_adjacencies
        self.upper_adjacencies = upper_adjacencies

        self.adjacencies = self.raised_graph.get_adjacencies() # dimension: 4, number_of_simplices, number_of_simplices

        self.transforms = {} # these are the functions to transform

        ## we can represent the adjacency matrices in one matrx of dimension (allowed_adjacencies, total_simplices, total_simplices)
        self.weights = torch.rand(4, len(self.raised_graph.boundary), len(self.raised_graph.boundary))
        self.num_complexes = len(self.raised_graph.boundary) # length of one of the adjacency matrices

        self.queries = torch.rand(self.num_heads, 4, self.num_complexes, self.num_complexes)
        
        for dimension in range(len(self.dimension)): # produce a unified represnetaion of a simplex
            self.transforms[dimension] = torch.nn.Linear((dimension + 1) * self.node_dimension, self.simplex_dimension)

        self.num_heads = num_heads # number of attention heads
        self.num_neighborhood = 4 

        self.lower = self.raised_graph.lower
        self.upper = self.raised_graph.upper
        self.co_boundary = self.raised_graph.co_boundary
        self.boundary = self.raised_graph.boundary 


        self.attention_weights = torch.nn.Parameter(torch.rand(self.batch_size, self.num_neighborhood, self.num_heads, self.num_complexes, 2 * self.node_dimension // self.num_heads ))

    def forward(self, x):
        # x . the function assumes the dimension to be: batch_size, number_of_neighborhood_functions, number_of_nodes, node_dimension
        # reshapre the tensor to account for number of heads and the number of nodes. 
        batch_size, n_neighborhood, n_complexes, dimension = x.shape
        x_transformed = torch.unsqueeze(x, 0).reshape(batch_size, n_neighborhood, n_complexes, self.num_heads, dimension / self.num_heads).transpose(2, 3)
        x_transformed_repeat = x_transformed.repeat(1, 1, 1, 1, self.num_complexes)
        x_transformed_repeat_inteleaved = x_transformed.repeat_interleave(self.num_complexes, dim = -1)
    
        x_concatenated = torch.cat([x_transformed_repeat, x_transformed_repeat_inteleaved], dim = -1) # this mergees the two tensors along the last dimension

        attention = torch.nn.functional.softmax(x_concatenated  *self.attention_weights)
        attention.transpose(2, 3).reshape(batch_size, n_neighborhood, n_complexes, self.num_complexes)

        