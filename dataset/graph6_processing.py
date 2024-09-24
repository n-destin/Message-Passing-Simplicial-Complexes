# @ Author: Destin Niyomufasha
# Message Passing Simplicial Complexes

def encode_graph(adj):
    
    
    binary_string = ""
    graph6 = ""
    for row in range(len(adj)):
        if row != 0:
            binary_string += "".join(str(adj[row][col]) for col in range(row))
    binary_string = binary_string + "0"*(6 - len(binary_string) % 6) if len(binary_string) % 6 != 0 else binary_string
    chunks = [binary_string[index : index + 6] for index in range(0, len(binary_string), 6)]
    graph6 = chr(len(adj) + 63) + "".join(chr(int(chunk, 2) + 63) for chunk in chunks) 
    
    return graph6

def sum(n):
    return n * (n + 1) / 2

def decode_graph(string):

    n_nodes = ord(string[0]) - 63 # number of vertices in the graph
    adj_matrix = [[0 for _ in range(n_nodes)] for _ in range(n_nodes)]

    values = [ord(char) - 63 for char in string[1:len(string)]]
    bits = [f"{value:b}" for value in values]
    processed_bits = "".join(["0"*(6 - len(bit) % 6) + bit if len(bit) % 6 != 0 else bit for bit in bits])

    left = 0
    right = 1
    
    adj_matrix[1][0] = 1
    while left < right and right <= sum(n_nodes - 1):
        row = right - left
        for index in range(row):
            adj_matrix[row][index] = int(processed_bits[left : right][index])
            adj_matrix[index][row] = int(processed_bits[left : right][index])
        temp = left
        left = right 
        right += (right - temp) + 1

    return adj_matrix