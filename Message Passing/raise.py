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
    


class RaiseGraph():
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

            for dimension in range(len(self.dimension)):
                self.transform[dimension] = torch.nn.linear(dimension * self.node_dimension, self.simplex_dimension)

            # dimension and at least one type of adjacency
            assert self.dimension >= 0 and self.dimension < 4 and (self.lower_adjacecnies or self.upper_adjacencies or self.co_boundary_adjacency or self.boundary_adjacency)
            self.construct_simplices()

    def construct_simplices(self):
        total = len(self.adj_matrix)
        for dimension in range(self.dimension):
            self.simplices[dimension] = []
            if dimension == 0:
                for node in range(len(self.adj_matrix)):
                    self.simplices[dimension].append(Node(node))
            else:
                for simplex in combinations.combinations(range(len(self.adj_matrix)), dimension + 1):
                    if self.verify_cycle(simplex):
                        self.simplices[dimension].append(Simplex(simplex), self.transform[dimension](torch.cat([self.simplices[0][node].get_feature() for node in simplex], dim = 0)))
                        total += 1
        # order the simplices
        for index in range(total):
            for simplex in self.simplices[dimension]:
                self.ordering[index] = simplex
        

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

        to_return = []
        if len(simplex1) != len(simplex2):
            return False
        else:
            if len(simplex1) == 1:
            
        return []

    
    def get_adjacencies(self):
        '''
        Returns the adjacency matrix of the simplicial complex
        '''
        lower = torch.rand(len(self.simplices[0]), len(self.simplices[0]))
        upper = torch.rand(len(self.simplices[0]), len(self.simplices[0]))
        co_boundary = torch.rand(len(self.simplices[0]), len(self.simplices[0]))
        boundary = torch.rand(len(self.simplices[0]), len(self.simplices[0]))

        for index1, simplex1 in enumerate(self.simplices[0]):
            for index2, simplex2 in enumerate(self.simplices[0]):
                if self.verify_adjacency(simplex1, simplex2).contains("lower"):
                    lower[index1][index2] = 1
                if self.verify_adjacency(simplex2, simplex1).contains("upper"):
                    upper[index1][index2] = 1
                if self.verify_adjacency(simplex1, simplex2).contains("co_boundary"):
                    co_boundary[index1][index2] = 1
                if self.verify_adjacency(simplex2, simplex1).contiains("boundary"):
                    boundary[index1][index2] = 1

        return lower, upper, co_boundary, boundary