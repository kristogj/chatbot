# Chatbot

## Dataset
The [Cornell Movie-Dialog Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)
is a rich dataset of movie character dialog:
* 220,579 conversational exchanges between 10,292 pairs of movie characters
* 9,035 characters from 617 movies
* 304,713 total utterances

Download the ZIP file and put it in a *data/* directory under the current directory.

## Model
The model used in this chatbot is a sequence-to-sequence (seq2seq) model. 
Here I use two seperate recurrent neural nets together. One of them
acts as an **encoder**, which encodes a variable length input sequence to a fixed-length 
context vector. In theory, this context vector (the final hidden layer of the RNN) 
will contain semantic information about the query sentence that is input to 
the bot. The second RNN is a **decoder**, which takes an input word and context vector, and
 returns a guess for the next word in the sequence and a hidden state to use in the 
 next iteration.

![Encoder-Decoder](/images/encoder-decoder.png)  
[Image source](https://jeddy92.github.io/JEddy92.github.io/ts_seq2seq_intro/)

## References
* Oriol Vinyals and Quoc V. Le. A neural conversational model.CoRR, abs/1506.05869, 2015.
* Minh-Thang Luong, Hieu Pham, and Christopher D. Manning. Effective approaches to attention-basedneural machine 
translation.CoRR, abs/1508.04025, 2015.
