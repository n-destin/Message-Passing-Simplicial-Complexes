import torch 


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
                 boundary_adjacency = True, 
                 co_boundary_adjacency = True, 
                 lower_adjacencies = True, 
                 upper_adjacencies = True) -> None:
        super(CochainMessagePassing, self).__init__()
        
        self.node_dimension = node_dimension
        self.simplex_dimension = simplex_dimension
        self.dimension = dimension
        self.adj_matrix = adj_matrix
        self.boundary_adjacency = boundary_adjacency
        self.co_boundary_adjacency = co_boundary_adjacency
        self.lower_adjacencies = lower_adjacencies
        self.upper_adjacencies = upper_adjacencies
        self.transform = {}
        self.construct_simplices()
        for dim in range(self.dimension):
            self.transform[dim] = torch.nn.Linear(dim * self.node_dimension, self.simplex_dimension)