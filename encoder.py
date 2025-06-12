import numpy as np

text_sentence = "This is the encoder architecture in pure python and numpy"
word_dimensions = 24

# Encoder function: simulates the Transformer Encoder block
def encoder(sentence, dimension, first_layer = False, previous_output = None):

    # Tokenization: splits sentence into lowercase tokens
    def tokenize(text):

        text = text.lower()

        return text.split()
    
    # Creates a random embedding vector for each token
    def get_embeddings(text):

        embeddings = {word: np.random.randn(dimension) for word in text}

        return embeddings
    
    # Generates sinusoidal positional encodings (as used in original Transformer)
    def get_positional_encodings(text):

        positional_encoding = {}

        for position, word in enumerate(text, 1):

            pos_encoding = []

            for i in range(dimension//2):

                pos_encoding.append(np.sin(position/(10000 ** (2 * (i/4)))))
                pos_encoding.append(np.cos(position/(10000 ** (2 * (i/4)))))

            positional_encoding[word] = pos_encoding

        return positional_encoding
    
    tokens = tokenize(sentence)

    # Create embeddings + positional encodings
    word_embeddings = get_embeddings(tokens)
    positions = get_positional_encodings(tokens)


    # Add positional encodings to word embeddings
    contexed_words = {}

    for token in tokens:
        contexed_words[token] = word_embeddings[token] + positions[token]

    contexed_word_embbedings = np.array(list(contexed_words.values()))


    # Create Q, K, V matrices via learned linear projections
    def get_Q_K_V_matrices(X):
        w_q = np.random.randn(dimension, 8)
        w_k = np.random.randn(dimension, 8)
        w_v = np.random.randn(dimension, 8)

        Q = X @ w_q
        K = X @ w_k
        V = X @ w_v

        return (Q, K, V)

    Query, Key, Value = get_Q_K_V_matrices(contexed_word_embbedings if first_layer else previous_output)


    def softmax(x, axis=None):
        x_max = np.max(x, axis=axis, keepdims=True)
        e_x = np.exp(x - x_max)
        return e_x / np.sum(e_x, axis=axis, keepdims=True)

    # Computes scaled dot-product attention
    def get_attension_scores(Q,K,V):

        d_k = Q.shape[1]

        Q_dot_K_Transpose = (Q @ K.T) / np.sqrt(d_k)

        Q_dot_K_Transpose = softmax(Q_dot_K_Transpose, axis= 1)

        attention_scores = Q_dot_K_Transpose @ V

        return attention_scores


    # Implements multi-head self-attention
    def get_multi_head_attention_scores(X, heads = 8):

        multi_headed_attention_scores = []

        for head in range(heads):

            Q,K,V = get_Q_K_V_matrices(X)

            attention_score = get_attension_scores(Q,K,V)

            multi_headed_attention_scores.append(attention_score)

        return np.stack(multi_headed_attention_scores, axis = 1)
        
    multi_head_scores = get_multi_head_attention_scores(contexed_word_embbedings)


    # Concatenate heads and project to original dimension
    def projection(multi_headed_attention_scores):

        token_dim, no_of_heads, each_head_output_dim = multi_headed_attention_scores.shape

        concatenated_multi_head = multi_headed_attention_scores.reshape(len(tokens), -1)

        W_O = np.random.randn((no_of_heads * each_head_output_dim), dimension)

        projected_matrix = concatenated_multi_head @ W_O

        return projected_matrix

    final_projected_vector = projection(multi_headed_attention_scores= multi_head_scores)


    # Residual connection + layer normalization
    def residual_connection(X, projected_vector):

        updated_X = X + projected_vector

        def layer_norm(array):
            mean = np.mean(array)
            std = np.std(array)
            e = 0.000001

            return [(x - mean) / (std + e) for x in array]
        
        return np.array([layer_norm(item) for item in updated_X])

    output = residual_connection(contexed_word_embbedings, final_projected_vector)


    # Position-wise Feedforward Network with ReLU
    def forward_pass(input, no_of_neurons_1):

        def feed_forward(input, no_of_neurons, activation = None):

            no_of_weights = input.shape[1]

            weights = np.random.randn(no_of_weights, no_of_neurons)
            biases = np.random.randn(no_of_neurons)
            
            outputs = input @ weights + biases
            outputs = np.maximum(0, outputs) if activation else outputs
            return outputs 

        first_output = feed_forward(input, no_of_neurons_1, activation= True)

        final_output = feed_forward(first_output, input.shape[1])

        return final_output

    output = residual_connection(output, forward_pass(output, 512))

    return output

# Runs stacked encoder layers (like the Transformer encoder stack)
def run_architecture(text_sentence, word_dimensions, encoder_layers = 8):

    first_layer_output = encoder(sentence= text_sentence, dimension= word_dimensions, first_layer= True)

    layer_output = first_layer_output

    for layer in range(encoder_layers-1):
        layer_output = encoder(sentence= text_sentence, dimension = word_dimensions, first_layer= False, previous_output= layer_output)
    
    return layer_output

print(run_architecture(text_sentence, word_dimensions, encoder_layers= 16))
