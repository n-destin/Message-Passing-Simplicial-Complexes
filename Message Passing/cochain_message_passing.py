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
        self.weights = torch.rand(self.num_heads, 4, len(self.raised_graph.boundary), len(self.raised_graph.boundary))
        
        for dimension in range(len(self.imensions)): # produce a unified represnetaion of a simplex
            self.transforms[dimension] = torch.nn.Linear((dimension + 1) * self.node_dimension, self.simplex_dimension)

        self.num_heads = num_heads # number of attention heads

        self.lower = self.raised_graph.lower
        self.upper = self.raised_graph.upper
        self.co_boundary = self.raised_graph.co_boundary
        self.bounadry = self.raised_graph.boundary

        def forward(self, x):
            # x is the feature matrix of the nodes 

            for dimension in range(self.dimension):
                if dimension == 0:
                    x_transformed = torch.unsqueeze(torch.unqueeze(x, 0).repeat(4, 1, 1), 1).repeat(self.num_heads, 1, 1) # expanding the feature matrix of the nodes to the number of adjacencies and number of heads


                else:
                    pass         
