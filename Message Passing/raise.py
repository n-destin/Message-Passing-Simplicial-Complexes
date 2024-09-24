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

    def __eq__(self, node: object) -> bool:
        return node.index == self.index

class Simplex():
    def __init__(self, nodes, feature = None) -> None:
         self.nodes = nodes
         self.feature = feature
         self.dimension = len(nodes) - 1
    

class RaisedGraph():
    '''
        This class takes in an adjacency matrix of a graph and returns a n-dimensional simplicial complex.
        The class assumes that the edges don't have their featrures. Their features are consutructed from the nodes.
    '''
    def __init__(self,
                 adj_matrix,
                 simplex_dimension,
                 node_dimension, 
                 dimension) -> None:
            
            self.simplex_dimension = simplex_dimension
            self.node_dimension = node_dimension
            self.transform = {}
            self.adj_matrix = adj_matrix
            self.dimension = dimension
            self.simplices = {}
            self.ordering = {}


            self.number_of_simplices = 0

            self.boundary_functions = [self.check_boundary, self.check_co_boundary, self.check_lower, self.check_upper]

            self.lower = torch.rand(len(self.simplices[0]), len(self.simplices[0]))
            self.upper = torch.rand(len(self.simplices[0]), len(self.simplices[0]))
            self.co_boundary = torch.rand(len(self.simplices[0]), len(self.simplices[0]))
            self.boundary = torch.rand(len(self.simplices[0]), len(self.simplices[0]))

            ## self.transoforms = torch.nn.Parameter(torch.rand(self.node_dimension, self.simplex_dimension)) ## Use this if aggregating the nodes fo a simplex using mean, sum, max, min
            
            for dimension in range(len(self.dimension)):
                self.transform[dimension] = torch.nn.linear(dimension * self.node_dimension, self.simplex_dimension)

            # dimension and at least one type of adjacency
            assert self.dimension >= 0 and self.dimension < 4 and (self.lower_adjacecnies or self.upper_adjacencies or self.co_boundary_adjacency or self.boundary_adjacency)
            self.construct_simplices()
            self.adjacencies = torch.zeros(4, self.number_of_simplices, len(self.simplices[0])) ## adjacency matrices for the simplicial complex

    def construct_simplices(self):
        total = len(self.adj_matrix)
        for dimension in range(self.dimension):
            self.simplices[dimension] = []
            if dimension == 0:
                for node in range(len(self.adj_matrix)):
                    self.simplices[dimension].append(Node(node))
                    self.number_of_simplices += 1
            else:
                for simplex in combinations.combinations(range(len(self.adj_matrix)), dimension + 1):
                    if self.verify_connectedness(simplex):
                        self.simplices[dimension].append(Simplex(simplex), self.transform[dimension](torch.cat([self.simplices[0][node].get_feature() for node in simplex], dim = 0)))
                        total += 1
                        self.number_of_simplices += 1
        # order the simplices
        for index in range(total):
            for simplex in self.simplices[dimension]:
                self.ordering[index] = simplex
        

    def verify_connectedness(self, nodes):
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
    
    def check_boundary(self, simplex1, simplex2):
        return len(set(simplex1.nodes).difference(set(simplex2.nodes))) == 1 and len(simplex1) > len(simplex2)

    def check_co_boundary(self, simplex1, simplex2):
        return len(set(simplex2.nodes).difference(set(simplex1.nodes))) == 1 and len(simplex2) > len(simplex1)

    def check_lower(self, simplex1, simplex2):
        if len(simplex1) != len(simplex2):
            return False
        else:
            for simplex in self.simplices[simplex1.dimension + 1]:
                if self.boundary(simplex, simplex1) and self.boundary(simplex, simplex2):
                    return True
        return False

    def check_upper(self, simplex1, simplex2):
        if len(simplex1) != len(simplex2):
            return False
        else:
            for simplex in self.simplices[simplex1.dimension - 1]:
                if self.co_boundary(simplex1, simplex) and self.co_boundary(simplex2, simplex):
                    return True

        return False
    
    
    def get_adjacencies(self):
        '''
        Returns the adjacency matrix of the simplicial complex
        '''
        for index1, simplex1 in enumerate(self.simplices[0]):
            for index2, simplex2 in enumerate(self.simplices[0]):
                for index, boundary_function in self.enumerate(self.boundary_functions):
                    if boundary_function(simplex1, simplex2):
                        self.adjacencies[index][index1][index2] = 1
        
        return self.adjacencies

        