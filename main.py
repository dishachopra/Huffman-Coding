import streamlit as st
import math
import numpy as np
from streamlit import components
from graphviz import Digraph
import pandas as pd

class Nodes:
    def __init__(self, prob, symbol, left=None, right=None):
        self.prob = prob
        self.symbol = symbol
        self.left = left
        self.right = right
        self.code = ''

def calFreq(the_data):
    the_symbols = dict()
    for i in the_data:
        if the_symbols.get(i) is None:
            the_symbols[i] = 1
        else:
            the_symbols[i] += 1
    return the_symbols

codes = dict()

def calCodes(node, val=''):
    newval = val + str(node.code)
    if node.left:
        calCodes(node.left, newval)
    if node.right:
        calCodes(node.right, newval)
    else:
        codes[node.symbol] = newval
    return codes

def encodedOutput(the_data, coding):
    l = []
    for i in the_data:
        l.append(coding[i])
    ans = ''.join([str(i) for i in l])
    return ans

def TotalGain(the_data, coding, symbol_frequencies):
    n = len(symbol_frequencies)
    a = np.log2(n)
    bits = int(np.ceil(a))
    befComp = len(the_data) * bits
    afComp = 0
    the_symbols = coding.keys()
    for symbol in the_symbols:
        the_count = the_data.count(symbol)
        afComp += the_count * len(coding[symbol])
    return befComp, afComp

def calculateEntropy(probabilities):
    entropy = 0
    sum = 0
    for probability in probabilities:
        sum += probability
    for probability in probabilities:
        entropy += (probability / sum) * math.log2(sum / probability)
    entropy = round(entropy, 2)
    return entropy

def calculateAverageLength(coding, probabilities):
    averageLength = 0
    sumq = 0
    for probability in probabilities.values():
        sumq += float(probability)
    for symbol, probability in probabilities.items():
        averageLength += (float(probability) / sumq) * len(coding[symbol])
    averageLength = round(averageLength, 2)
    return averageLength

def HuffmanEncoding(the_data):
    symbolWithProbs = calFreq(the_data)
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
    befComp, afComp = TotalGain(the_data, huffmanEncoding, symbolWithProbs)
    entropy = calculateEntropy(the_prob)
    averageLength = calculateAverageLength(huffmanEncoding, symbolWithProbs)
    output = encodedOutput(the_data, huffmanEncoding)

    # Generate the modified encoded output with horizontal curly brackets below each symbol
    modified_output = ""
    for symbol in the_data:
        bits = huffmanEncoding[symbol]
        modified_output += f"\\underbrace{{\\Large {bits}}}_{{{symbol}}} \\quad "

    return modified_output, the_nodes[0], befComp, afComp, entropy, averageLength

def HuffmanDecoding(encoded_string, huffman_tree):
    decoded_string = ""
    all_uniq_chars_in_encoded_string = set(encoded_string)
    # check set has any char other than 0 and 1
    diff = all_uniq_chars_in_encoded_string - {'0', '1'}
    if diff:
        raise ValueError(f"Invalid Huffman code. Characters other than 0 and 1 found\n {diff}")
    current_node = huffman_tree
    for bit in encoded_string:
        if bit == '0':
            current_node = current_node.left
        else:
            current_node = current_node.right

        if current_node.left is None and current_node.right is None:
            decoded_string += current_node.symbol
            current_node = huffman_tree

    if current_node != huffman_tree:
        # during raise value error, it will show the bit index where the error occurred
        bit_index = len(encoded_string) - len(bit)
        raise ValueError(f"Incorrect Huffman code. Decoding stopped at bit {bit_index}")

    return decoded_string

uniform = dict()

def print_symbol_frequencies(symbol_frequencies):
    n = len(symbol_frequencies)
    a = np.log2(n)
    bits = int(np.ceil(a))

    for index, symbol in enumerate(symbol_frequencies):
        indInt = int(index)
        co = bin(indInt)[2:].zfill(bits)
        uniform[symbol] = co

    table_data = [["Symbol", "Frequency", "Uniform Code Length", "Huffman Code"]]
    for symbol, frequency in symbol_frequencies.items():
        code = codes.get(symbol, "")
        table_data.append([symbol, frequency, uniform.get(symbol), code])

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

def check_huffman_codes(codes_input):
    # Check if all codes are binary and unique
    valid_codes = all(code_input.isdigit() and set(code_input) <= {'0', '1'} for code_input in codes_input.values())
    unique_codes = len(set(codes_input.values())) == len(codes_input)

    if valid_codes and unique_codes:
        return True
    else:
        return False

def main():
    st.markdown(
        """
        <style>
        body {
            background-color: black;
        }
        h1 { 
            color: #457B9D;
            text-align: center;
            font-size: 36px;
            font-family: 'Helvetica Neue', sans-serif;
        }
        h2, h3, h4 {
            color: #457B9D;
            font-size: 24px;
            font-family: 'Helvetica Neue', sans-serif;
        }
        p {
            font-size: 18px;
            font-family: 'Helvetica Neue', sans-serif;
        }
        code {
            font-size: 16px;
            font-family: 'Courier New', monospace;
        }
        table {
            font-size: 16px;
            font-family: 'Helvetica Neue', sans-serif;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

 

    st.markdown("<h1 style='text-align: center; color: #FFFF66;'>Huffman Coding</h1>", unsafe_allow_html=True)

   


# Text and options
    question_text = "If you need to transfer a message which is a string 'aaabbcdd', what code will you choose to represent each character?"
    options_data = {
        'Option A': 'a="00", b="01", c="10", d="11"',
        'Option B': 'a="1", b="011", c="010", d="00"'
    }

# Display the question and options in a table
    st.markdown("<p style='text-align: justify;'>Welcome to the world of data compression! In this blog, we will explore the art of reducing the size of data without sacrificing any vital information. Let me start with a question for you all.</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: justify;'>{question_text}</p>", unsafe_allow_html=True)

# Create a DataFrame to display the options in a table
    options_df = pd.DataFrame(list(options_data.items()), columns=['Options', 'Codes'])
    st.table(options_df)

# Get user's selection
    selected_option = st.radio("Select the correct answer:", ['', *options_data.keys()])

# Display prompt based on the user's selection
    if selected_option == 'Option A':
        st.info("You are close to the correct answer, but there can be another assignment for characters that can transmit this string into fewer bits. Let's dive into it.")
    elif selected_option == 'Option B':
        st.success("That was an intelligent guess. Let's understand the logic behind this assignment, Huffman encoding.")

    
    st.markdown("<p style='font-size: 20px; font-weight: bold; color: light blue; margin-bottom: 0;'>Enter the string:</p>", unsafe_allow_html=True)
    the_data = st.text_input("", "huffman")
    
            
   
    # Section 2: Explanation of Huffman Coding
    st.markdown("<h2 style='text-align: center;color: #FFFF66;'>The Problem:</h2>", unsafe_allow_html=True)

    st.markdown("In the digital world, we deal with lots of data every day, from text messages to images and videos. "
                "Transmitting or storing this data requires space and time. Imagine if we could somehow make this data "
                "smaller without losing any important information. That's where data compression comes in!")
    st.markdown("<h2 style='text-align: center;color: #FFFF66;'>The Solution - Huffman Coding:</h2>", unsafe_allow_html=True)
    st.markdown("Huffman Coding is a smart way to compress data. It was invented by David A. Huffman, and it works like magic! "
                "The basic idea is to give shorter codes to more common characters or pieces of data and longer codes to the less common ones. "
                "This makes the data take up less space. It is a lossless method for compressing and encoding text based on the frequency of the characters in the text. In information theory and computer science studies, Huffman code is a special type of optimal prefix code that is generally utilized for lossless data compression.")
    



    # Huffman Coding Visualization
    st.markdown("<p style='font-size: 20px; font-weight: bold; color: light blue; margin-bottom: 0;'>Encoded Output:</p>", unsafe_allow_html=True)
    encoding, the_tree, befComp, afComp, entropy, averageLength = HuffmanEncoding(the_data)
    st.latex(encoding)

    st.markdown("<h2 style='text-align: center;color: #FFFF66;'>Huffman Tree</h2>", unsafe_allow_html=True)
    st.markdown("To start with Huffman Coding, we first analyze the data to see how often each character or piece of data appears. We can use greedy algorithm to construct the tree. "
                "We give each one a 'weight' based on its frequency. With the weights assigned, we create a special tree called the 'Huffman Tree'. In this tree, the characters with higher "
                "weights (more frequent) are closer to the top, and those with lower weights (less frequent) are closer to the bottom.")
    
    st.markdown("<h2 style='text-align: center;color: #FFFF66;'>How is a Huffman Tree Created?</h2>", unsafe_allow_html=True)
    st.markdown("""
        - <h2 style="font-size: 20px;">Frequency Analysis:</h2> <span style="font-size: 20px;">Count the frequency of each character in the input data.</span>
        - <h2 style="font-size: 20px;">Create Nodes:</h2> <span style="font-size: 20px;">Each character becomes a leaf node, and its frequency becomes the weight.
        - <h2 style="font-size: 20px;">Build Priority Queue:</h2> <span style="font-size: 20px;">Place all the leaf nodes in a priority queue, ordered by their frequencies.
        - <h2 style="font-size: 20px;">Merge Nodes:</h2> <span style="font-size: 20px;">Repeatedly remove the two nodes with the lowest frequencies from the priority queue. Create a new internal node with a frequency equal to the sum of the two nodes' frequencies. Connect the two nodes as left and right children of the new node and add it back to the priority queue.
        - <h2 style="font-size: 20px;">Repeat:</h2> <span style="font-size: 20px;">Continue merging nodes until only one node remains in the priority queue. This node becomes the root of the Huffman tree.
        - <h2 style="font-size: 20px;">Assign Binary Codes:</h2> <span style="font-size: 20px;">Traverse the Huffman tree from the root to each leaf node. Assign a binary code to each character based on the path taken (left or right) during the traversal. The path to the left child is assigned a 0, and the path to the right child is assigned a 1.
    """, unsafe_allow_html=True)
    

  
    print_huffman_tree(the_tree)

    # Symbol Frequencies and Codes
    
    st.markdown("<h2 style='text-align: center;color: #FFFF66;'>Symbols, Frequencies and Codes</h2>", unsafe_allow_html=True)
    symbol_frequencies = calFreq(the_data)
    print_symbol_frequencies(symbol_frequencies)    
    st.markdown("<h3 style='text-align: center; color: #FFFF66;'>Saving Space and Time:</h3>", unsafe_allow_html=True)

    st.markdown("Since more frequent characters have shorter codes, when we compress the data using Huffman Coding, the more common characters "
                "take up less space. This leads to an overall reduction in the size of the data, making it smaller and more manageable. Since the more common characters have shorter codes, they take less time to transmit or store. This makes the data transmission "
                "faster and more efficient.")


    st.markdown("<h3 style='text-align: center; color: #FFFF66;'>Entropy- Measuring Efiiciency:</h3>", unsafe_allow_html=True)

    #what is entropy and average length of the code
    st.markdown("Entropy is a measure of the randomness of the data. The higher the entropy, the more random the data is. "
                    "Entropy is calculated using the probability of each symbol in the data. "
                    "The more random the data is, the more bits we need to represent it. "
                    "So, the higher the entropy, the longer the average length of the code.")
    

    st.latex(r'''H(X) = -\sum_{i=1}^{n} p(x_i) \log_2 p(x_i)''')
    st.markdown("Where: " + r"$p(x_i)$" + " is the probability of the symbol")
    # Inside main() function, after getting the frequency table symbol_frequencies.
    # Inside main() function, after getting the frequency table symbol_frequencies.
    # Inside main() function, after getting the frequency table symbol_frequencies.
    symbol_frequencies = calFreq(the_data)

# Calculate probabilities
    total_symbols = sum(symbol_frequencies.values())
    probabilities = [frequency / total_symbols for frequency in symbol_frequencies.values()]




# Print the probabilities and terms of the entropy equation
    st.write("Let's calculate entropy for the string provided earlier:")
    for symbol, frequency in symbol_frequencies.items():
        probability = frequency / total_symbols
        st.write(f"p({symbol}) = {probability:.2f}")

# Display the fully written entropy equation
    entropy_eq_fully_written = f"Entropy = -(" + " - ".join([f"{probability:.2f} \cdot \log_2(1 / {probability:.2f})" for probability in probabilities]) + ")"
    with st.expander("Entropy Equation"):
        st.latex(entropy_eq_fully_written)

    # Calculate and print entropy
    entropy_text = f"<span style='color: white;'>Entropy of the input data: </span><span style='color: yellow;'>{entropy:.2f}</span>"
    st.markdown(entropy_text, unsafe_allow_html=True)




    


    
    

    

    st.markdown("<h3 style='text-align: center; color: #FFFF66;'>Average Length:</h3>", unsafe_allow_html=True)

    st.markdown("Average length is the average number of bits required to represent each symbol in the data. "
                "It's calculated using the probability of each symbol and the length of the code assigned to it. "
                "The more common the symbol, the shorter the code, and the less common the symbol, the longer the code. "
                "So, the more common the symbol, the shorter the average length of the code.")

    st.latex(r'''L(X) = \sum_{i=1}^{n} p(x_i) l(x_i)''')
    st.markdown("Where: " + r"$l(x_i)$" + " is the length of the symbol")
    # Inside main() function, after getting the frequency table symbol_frequencies.
    symbol_frequencies = calFreq(the_data)

# Calculate probabilities
    total_symbols = sum(symbol_frequencies.values())
    probabilities = [frequency / total_symbols for frequency in symbol_frequencies.values()]

# Calculate and print average length
    average_length = calculateAverageLength(codes, symbol_frequencies)

# Format the average length equation using LaTeX
    avg_length_eq_latex = "L(X) = \sum_{i=1}^{n} p(x_i) l(x_i)"

# Print the probabilities and lengths of each symbol
    st.write("Let's calculate the average length for the string provided earlier:")
    for symbol, frequency in symbol_frequencies.items():
        probability = frequency / total_symbols
        st.write(f" l({symbol}) = {len(codes[symbol])}")

# Display the value of average_length
    
    average_length = calculateAverageLength(codes, symbol_frequencies)

# Calculate the lengths of each code in the average length equation
    lengths = [len(code) for code in codes.values()]

# Format the average length equation using LaTeX with actual code lengths
    avg_eq_fully_written = f"L(X) = " + " + ".join([f"{probability:.2f} \cdot {length:.2f}" for probability, length in zip(probabilities, lengths)])

# Display the value of average_length and the fully written average length equation
    with st.expander("Average Length Equation"):
        st.latex(avg_eq_fully_written)





    average_text = f"<span style='color: white;'>Average Length: </span><span style='color: yellow;'>{average_length:.2f}</span>"
    st.markdown(average_text, unsafe_allow_html=True)






 
    st.markdown("<h3 style='text-align: center; color: #FFFF66;'>Compression Metrics:</h3>", unsafe_allow_html=True)

    st.markdown("The compression ratio tells us how much smaller the compressed data is compared to the original data. "
                "The higher the compression ratio, the better the compression!")
    st.latex(r'''\mathrm{Compression Ratio} = \frac{\mathrm{Before\ Compression}}{\mathrm{After\ Compression}}''')
    
    st.markdown("Let's see how well our compression works for the data you input earlier:")

    # Compression Metrics

    st.write("Before Compression with uniform code length (no. of bits): ", f"<span style='color: #FFFF66;'>{befComp}</span>", unsafe_allow_html=True)
    st.write("After Compression with Huffman code (no. of bits): ", f"<span style='color: #FFFF66;'>{afComp}</span>", unsafe_allow_html=True)
    st.write("Compression Ratio: ", f"<span style='color: #FFFF66;'>{round(befComp / afComp, 2)}</span>", unsafe_allow_html=True)
    

    # Huffman Decoding
    st.markdown("<h2 style='text-align: center; color: #FFFF66;'>Huffman Decoding</h2>", unsafe_allow_html=True)
    st.markdown("Now that we've seen how to compress data using Huffman Coding, let's see how to decompress it. "
                "To decompress the data, we need to use the same Huffman Tree we used to compress it. "
                "We start at the root node and follow the path to the leaf node based on the binary code. "
                "When we reach the leaf node, we get the character assigned to that code. "
                "We repeat this process until we reach the end of the encoded data.")
    
    st.markdown("Let's try it out using the above Huffman tree!")
    st.markdown("<p style='font-size: 20px; font-weight: bold; color: light blue; margin-bottom: 0;'>Enter the encoded string:</p>", unsafe_allow_html=True)
    encoded_input = st.text_input(" ")
    decoded_output = HuffmanDecoding(encoded_input, the_tree)
    st.markdown("**Decoded Output:** " + decoded_output)


    # Conclusion
    st.markdown("<h2 style='text-align: center;color: #FFFF66;'>Conclusion</h2>", unsafe_allow_html=True)

    st.markdown("Huffman Coding is a clever technique that helps us reduce data size while maintaining all the important information. "
                "It's widely used in computer algorithms, file compression, and data transmission, making our digital lives more efficient and magical!")
    st.markdown("With the kingdom's enchanting story of Huffman Coding complete, the magical land of Cyberspace embraced this new era of data compression. "
                "The once-burdensome data now flowed effortlessly through the kingdom's communication channels, thanks to the wisdom of the young wizard, Alex.")

hide_st_style = """
            <style>
            body {
            background-color: black;
            color: white;
        }
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
