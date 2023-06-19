import streamlit as st
import math
from graphviz import Digraph

class Nodes:
    def __init__(self, prob, symbol, left=None, right=None):
        self.prob=prob
        self.symbol=symbol
        self.left=left
        self.right=right
        self.code=''

def calProb(the_data):   #calculates frequency of letters
    the_symbols=dict()      
    for i in the_data:
        if the_symbols.get(i)==None:
            the_symbols[i]=1
        else:
            the_symbols[i]+=1
    return the_symbols

codes = dict()   #store codes for each letter

def calCodes(node, val=''):
    newval=val+str(node.code)
    if node.left:
        calCodes(node.left, newval)
    if node.right:
        calCodes(node.right, newval)
    else:
        codes[node.symbol]=newval
    return codes

def encodedOutput(the_data, coding):
    l=[]
    for i in the_data:
        l.append(coding[i])
    
    ans=''.join([str(i) for i in l])
    return ans

def TotalGain(the_data, coding):
    befComp = len(the_data) * 8
    afComp = 0
    the_symbols = coding.keys()
    for symbol in the_symbols:
        the_count = the_data.count(symbol)
        afComp+= the_count * len(coding[symbol])
    return befComp, afComp

def calculateEntropy(probabilities):
    entropy = 0
    sum=0
    for probability in probabilities:
        sum+=probability
    for probability in probabilities:
        entropy += (probability/sum) * math.log2(sum/probability)
    return entropy

def calculateAverageLength(coding, probabilities):
    averageLength = 0
    sumq = 0
    for probability in probabilities.values():
        sumq += float(probability)
    for symbol, probability in probabilities.items():
        averageLength += (float(probability)/sumq) * len(coding[symbol])
    return averageLength


def HuffmanEncoding(the_data):
    symbolWithProbs = calProb(the_data)
    the_symbols = symbolWithProbs.keys()
    the_prob = symbolWithProbs.values()

    the_nodes = []

    for symbol in the_symbols:
        the_nodes.append(Nodes(symbolWithProbs.get(symbol), symbol))

    while len(the_nodes) > 1:
        the_nodes = sorted(the_nodes, key=lambda x: x.prob)
        right = the_nodes[0]
        left = the_nodes[1]

        left.code = 0
        right.code = 1

        newNode = Nodes(left.prob + right.prob, left.symbol + right.symbol, left, right)

        the_nodes.remove(left)
        the_nodes.remove(right)
        the_nodes.append(newNode)

    huffmanEncoding = calCodes(the_nodes[0])
    befComp, afComp = TotalGain(the_data, huffmanEncoding)
    entropy = calculateEntropy(the_prob)
    averageLength = calculateAverageLength(huffmanEncoding, symbolWithProbs)
    output = encodedOutput(the_data, huffmanEncoding)
    return output, the_nodes[0], befComp, afComp, entropy, averageLength

def HuffmanDecoding(encoded_string, huffman_tree):
    decoded_string = ""
    current_node = huffman_tree
    
    for bit in encoded_string:
        if bit == '0':
            current_node = current_node.left
        else:
            current_node = current_node.right
        
        if current_node.left is None and current_node.right is None:
            decoded_string += current_node.symbol
            current_node = huffman_tree
    
    return decoded_string

def print_symbol_frequencies(symbol_frequencies):
    table_data = [["Symbol", "Frequency", "Code"]]
    for symbol, frequency in symbol_frequencies.items():
        code = codes.get(symbol, "")
        table_data.append([symbol, frequency, code])

    st.table(table_data)

def print_huffman_tree(node):
    dot = Digraph()
    dot.node('root', label=f"{node.symbol}\n{node.prob}", shape='circle', style='filled', color='white', fillcolor='red')
    create_graph(node, dot, 'root')
    
    # Set graph attributes
    dot.graph_attr.update(bgcolor='None', size='10,10')
    dot.node_attr.update(shape='circle', style='filled', color='white')
    dot.edge_attr.update(color='white', fontcolor='white')

    # Render the graph
    dot.format = 'png'
    dot.render('huffman_tree', view=False)

    st.image('huffman_tree.png')

def create_graph(node, dot, parent=None):
    if node.left:
        dot.node(str(node.left), label=f"{node.left.symbol}\n{node.left.prob}", style='filled', fillcolor='#00CCFF')
        if parent:
            dot.edge(str(parent), str(node.left), label='0', color='#FF0000')
        create_graph(node.left, dot, node.left)

    if node.right:
        dot.node(str(node.right), label=f"{node.right.symbol}\n{node.right.prob}", style='filled', fillcolor='#00CCFF')
        if parent:
            dot.edge(str(parent), str(node.right), label='1', color='#00FF00')
        create_graph(node.right, dot, node.right)
        
def main():
    st.markdown("<h1 style='text-align: center; color: #457B9D;'>Huffman Coding</h1>", unsafe_allow_html=True)
    the_data = st.text_input("Enter the data:", "sustainibilitylab")
    encoding, the_tree, befComp, afComp, entropy, averageLength = HuffmanEncoding(the_data)
    st.write("Encoded Output: ", encoding)
    st.markdown("<h2 style='font-size: 24px;text-align: center; color: #457B9D;'>Huffman Tree:</h2>", unsafe_allow_html=True)
    print_huffman_tree(the_tree)
    

    st.markdown("<h2 style='font-size: 24px;text-align: center; color: #457B9D;'>Encoding Details:</h2>", unsafe_allow_html=True)
    
    st.write("Before Compression (no. of bits): ", befComp)
    st.write("After Compression (no. of bits): ", afComp)
    st.write("Entropy: ", entropy)
    st.write("Average Length: ", averageLength)
    
    st.markdown("<h2 style='font-size: 24px;text-align: center; color: #457B9D;'>Huffman Decoding:</h2>", unsafe_allow_html=True)
    encoded_input = st.text_input("Enter the encoded string:")
    decoded_output = HuffmanDecoding(encoded_input, the_tree)
    st.write("Decoded Output:", decoded_output)

    st.markdown("<h2 style='font-size: 24px; text-align: center;color: #457B9D;'>Symbols, Frequencies and Codes:</h2>", unsafe_allow_html=True)
    symbol_frequencies = calProb(the_data)
    print_symbol_frequencies(symbol_frequencies)



if __name__ == '__main__':
    main()
