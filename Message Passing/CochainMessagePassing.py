import torch 
from CellInspector import CellularInspector
from Raise import RaisedGraph
from typing import Optional
from torch import Tensor, SparseTensor



class CochainMessagePassing(torch.nn.Module):

    '''This class implements a cochain message passing layer. It takes in an n-dimensional simplicial complex and performs message passing on the 
    adjacency matrix. 
    Parameters:
        1. '''
    def __init__(self, 
                up_message_dimension, 
                down_message_dimension, 
                boundary_dimension,

                adj_matrix, 
                node_dimension, 
                simplex_dimension, 
                dimension, 

                dropout,
                alpha,
                num_heads,

                aggregate_up: Optional[str] = "add",
                aggregate_down : Optional[str] = "add", 

                use_boundary_message = True, 
                use_down_message = True,   

                flow = "sourcet_to_target"            
                 
                 ) -> None:
        super(CochainMessagePassing, self).__init__()

        self.aggregate_up = aggregate_up
        self.aggregate_down = aggregate_down

        self.use_boundary_message = use_boundary_message
        self.use_down_message = use_down_message

        self.up_message_dimension = up_message_dimension
        self.down_message_dimension = down_message_dimension
        self.boundary_dimension = boundary_dimension

        self.inspector = CellularInspector(self)
        self.node_dimension = node_dimension

        self.flow = flow

        assert self.aggregate_boundary in ["add", "mean", "max"]
        assert self.aggregate_co_boundary in ["add", "mean", "max"]
        assert self.flow in ["source_to_target", "target_to_source"]

        self.simplex_dimension = simplex_dimension

        self.adj_matrix = adj_matrix
        self.inspector.inspect(self.message_up)
        self.inspector.inspect(self.message_down)
        self.inspector.inspect(self.message_boundary)
        self.inspector.inspect(self.aggregate_up, n_items=1)
        self.inspector.inspect(self.aggregate_down, n_items=1)
        self.inspector.inspect(self.aggregate_boundary, n_items=1)
        self.inspector.inspect(self.message_and_aggregate_up, n_items=1)
        self.inspector.inspect(self.message_and_aggregate_down, n_items=1)
        self.inspector.inspect(self.message_and_aggregate_boundary, n_items=1)
        self.inspector.inspect(self.update, n_items=3)

        self.fuse_up = self.inspector.__implements___("message_and_aggregate_up")
        self.fuse_down = self.inspector.__implements___("message_and_aggregate_down")
        self.fuse_boundary = self.inspector.__implements___("message_and_aggregate_boundary")


        def check_input_together(self, index_up, index_down, size_up, size_down):
            # checks if the sizes of teh up and down adjancecnies aggree 
            if (index_up is not None and index_down is not None and size_up is not None and index_down is not None):
                assert size_up[0] == size_down[0]
                assert size_up[1] == size_up[1]


        def __check_input_separately(self, index, size):
            return_size = [None, None]
            if isinstance(index, Tensor):
                assert index.dtype == torch.long
                assert index.dim() == 2
                assert index.size(0) == 2

                if size is not None:
                    return_size[0] = size[0]
                    return_size[1] = size[1]
                
                return return_size
        
            elif isinstance(index, SparseTensor):
                if self.flow == "target_to_source":
                    raise ValueError(('Flow adjacency targe to source is invalid for message passing via sparsetensor.'))

                return_size[0] = index.sparse_size(1)
                return_size[1]= index.sparse_size(0)

                return return_size
            
            elif index is None:
                return return_size
            

            raise ValueError("MessagePassing.propagate only sipport torch.LongTensor of shape [2, num_messages] or torch_sparse.SparseTensor for argemnt edge_index")
        
            
        





        self.dropout = dropout # dropout rate
        self.alpha = alpha  # parameter for leaky relu

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


        # self.attention_weights = torch.nn.Parameter(torch.rand(self.batch_size, self.num_neighborhood, self.num_heads, self.num_complexes, 2 * self.node_dimension // self.num_heads ))




    # def forward(self, x):
    #     # x . the function assumes the dimension to be: batch_size, number_of_neighborhood_functions, number_of_nodes, node_dimension
    #     # reshapre the tensor to account for number of heads and the number of nodes. 
    #     batch_size, n_neighborhood, n_complexes, dimension = x.shape
    #     x_transformed = torch.unsqueeze(x, 0).reshape(batch_size, n_neighborhood, n_complexes, self.num_heads, dimension / self.num_heads).transpose(2, 3)
    #     x_transformed_repeat = x_transformed.repeat(1, 1, 1, 1, self.num_complexes)
    #     x_transformed_repeat_inteleaved = x_transformed.repeat_interleave(self.num_complexes, dim = -1)
    
    #     x_concatenated = torch.cat([x_transformed_repeat, x_transformed_repeat_inteleaved], dim = -1) # this mergees the two tensors along the last dimension

    #     attention = torch.nn.functional.softmax(x_concatenated  *self.attention_weights)
    #     attention.transpose(2, 3).reshape(batch_size, n_neighborhood, n_complexes, self.num_complexes)

        