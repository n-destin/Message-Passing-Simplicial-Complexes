import torch 
import combinations
import numpy as np


class Node():
    def __init__(self, index, feature) -> None:
        self.index = index
        self.feature = feature

    def get_feature(self,):
        return self.feature

    def set_feature(self, feature):
        self.feature = feature

class Simplex():
    def __init__(self, nodes, feature = None) -> None:
         self.nodes = nodes
         self.feature = feature
    


class RaiseGraph():
    '''
        This class takes in an adjacency matrix of a graph and returns a n-dimensional simplicial complex.
        The class assumes that the edges don't have their featrures. Their features are consutructed from the nodes.
    '''
    def __init__(self,
                 adj_matrix,
                 simplex_dimension,
                 node_dimension, 
                 dimension, 
                 boundary_adjacency = True,
                 co_boundary_adjacency = True,
                 lower_adjacecnies = True,
                 upper_adjacencies = True) -> None:
            
            self.simplex_dimension = simplex_dimension
            self.node_dimension = node_dimension
            self.transform = {}
            self.adj_matrix = adj_matrix
            self.dimension = dimension
            self.boundary_adjacency = boundary_adjacency
            self.co_boundary_adjacency = co_boundary_adjacency
            self.lower_adjacecnies = lower_adjacecnies
            self.upper_adjacencies = upper_adjacencies


            for dimension in range(len(self.dimension)):
                self.transform[dimension] = torch.nn.linear(dimension * self.node_dimension, self.simplex_dimension)

            # dimension and at least one type of adjacency
            assert self.dimension >= 0 and self.dimension < 4 and (self.lower_adjacecnies or self.upper_adjacencies or self.co_boundary_adjacency or self.boundary_adjacency)
            self.construct_simplices()

    def construct_simplices(self):
        simplices = {}
        total = len(self.adj_matrix)
        for dimension in range(self.dimension):
            simplices[dimension] = []
            if dimension == 0:
                for node in range(len(self.adj_matrix)):
                    simplices[dimension].append(Node(node))
            else:
                for simplex in combinations.combinations(range(len(self.adj_matrix)), dimension + 1):
                    if self.verify_cycle(simplex):
                        simplices[dimension].append(Simplex(simplex), self.transform[dimension](torch.cat([simplices[0][node].get_feature() for node in simplex], dim = 0)))
                        total += 1
            


        # order the simplices
        
        return simplices

    def verify_cycle(self, nodes):
        '''
        Checks if a node can 
        '''
        subgraph =  np.ix_(nodes, nodes)
        degrees = np.sum(subgraph, axis = 0)
        if not np.all(degrees == len(nodes) - 1): # fully connected structures
            return False

        # peform a dfs 
        visited = set()
        def dfs(node):
            if node in visited:
                return
            visited.add(node)
            for neighbor, connected in enumerate(subgraph[node]):
                if connected and neighbor not in visited:
                    dfs(neighbor)
        
        dfs(0)
        return len(visited) == len(nodes)
    
    def verify_adjacency(self, simplex1, simplex2):
        '''
        Verifies if two simplices are adjacent
        '''
        if len(simplex1) != len(simplex2):
            return False
        else:
            # verify the adjacency
            pass 

    
    def get_adjacency(self):
        '''
        Returns the adjacency matrix of the simplicial complex
        '''
        
