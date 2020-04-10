from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


class EncoderRNN(nn.Module):
    """
    Encodes a variable length input sequence to a fixed-length context vector. In theory, this context vector
    (the final hidden layer of the RNN) will contain semantic information about the query sentence that is input
    to the bot.
    """

    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        # Init GRU - the input_size and hidden_size are both set to hidden_size because
        # our input size is a word embedding with number of features == hidden size
        self.gru = nn.GRU(input_size=hidden_size,
                          hidden_size=hidden_size,
                          num_layers=n_layers,
                          dropout=(0 if n_layers == 1 else dropout),
                          bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)

        # Pack padded batch of sequences for RNN module
        packed = pack_padded_sequence(embedded, input_lengths)

        # Forward pass through GRU
        outputs, hidden = self.gru(packed, hidden)

        # Unpack padding
        outputs, _ = pad_packed_sequence(outputs)

        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]

        # Return output and final hidden state
        return outputs, hidden


class Attention(nn.Module):
    """
    Luong global attention layer
    https://arxiv.org/abs/1508.04025
    """

    def __init__(self, method, hidden_size):
        super(Attention, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    @staticmethod
    def dot_score(hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)
        else:
            raise ValueError(self.method, "is not an appropriate attention method.")

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)


class LuongAttentionDecoderRNN(nn.Module):
    """
    The decoder RNN generates the response sentence in a token-by-token fashion. It uses the encoder’s context vectors,
    and internal hidden states to generate the next word in the sequence. It continues generating words until it
    outputs an EOS_token, representing the end of the sentence.
    """

    def __init__(self, embedding, output_size, config):
        super(LuongAttentionDecoderRNN, self).__init__()

        # Keep for reference
        self.config = config
        self.attn_model = config["attn_model"]
        self.hidden_size = config["hidden_size"]
        self.output_size = output_size
        self.n_layers = config["decoder_n_layers"]
        self.dropout = config["dropout"]

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(self.dropout)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, self.n_layers,
                          dropout=(0 if self.n_layers == 1 else self.dropout))
        self.concat = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, output_size)
        self.attn = Attention(self.attn_model, self.hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        # Note: we run this on step (word) at a time
        # Get embedding of current input word
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)

        # Forward through un-directional GRU
        rnn_output, hidden = self.gru(embedded, last_hidden)

        # Calculate attention weights from the current GRU output
        attn_weights = self.attn(rnn_output, encoder_outputs)

        # Multiply attention weights to encoder outputs to get new "weighted" sum context vector
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))

        # Concatenate weighted context vector and GRU output using Luong eq. 5
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))

        # Predict next word using Luong eq. 6
        output = self.out(concat_output)

        # Make choice of word more stochastic
        if self.config["sampling"]:
            output = output / self.config["temperature"]

        output = F.softmax(output, dim=1)

        # Return output and final hidden state
        return output, hidden


class GreedySearchDecoder(nn.Module):
    """
    Greedy decoding is the decoding method that we use during training when we are NOT using teacher forcing.
    In other words, for each time step, we simply choose the word from decoder_output with the highest softmax value.
    This decoding method is optimal on a single time-step level.

    Steps:
    1. Forward input through encoder model.
    2. Prepare encoder’s final hidden layer to be first hidden input to the decoder.
    3. Initialize decoder’s first input as SOS_token.
    4. Initialize tensors to append decoded words to.
    5. Iteratively decode one word token at a time:
            Forward pass through decoder.
            Obtain most likely word token and its softmax score.
            Record token and score.
            Prepare current token to be next decoder input.
    6. Return collections of word tokens and scores.
    """

    def __init__(self, encoder, decoder, config):
        super(GreedySearchDecoder, self).__init__()
        self.config = config
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)

        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]

        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=self.config["device"], dtype=torch.long) * self.config["SOS_token"]

        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=self.config["device"], dtype=torch.long)
        all_scores = torch.zeros([0], device=self.config["device"])

        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)

            # Obtain most likely word token and its softmax score
            if self.config["sampling"]:
                decoder_input = Categorical(decoder_output).sample()
                decoder_scores, _ = torch.max(decoder_output, dim=1)
            else:
                decoder_scores, decoder_input = torch.max(decoder_output, dim=1)

            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)

            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)

        # Return collections of word tokens and scores
        return all_tokens, all_scores
