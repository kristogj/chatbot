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

## Results
![Loss](/images/loss_.png)

**Chat:** <br />
> Hello <br />

Bot: hello . . . . . <br />
> How are you doing? <br />

Bot: i m fine . . . . <br />
> Can you help me with something? <br />

Bot: i can t . . . . <br />
> Why not? <br />

Bot: because i don t know . . . <br />
> You don t know? <br />

Bot: no . . . . . <br />
> Well that s bad <br />

Bot: it s a surprise . . . <br />
> Do you have a surprise? <br />

Bot: no . . . . . <br />
> You sure? <br />

Bot: i think so . . . . <br />
> ok <br />

Bot: you can t do it . . . <br />
> Can t do what? <br />

Bot: i don t know . . . <br />
> What do you know? <br />

Bot: i don t know . . . <br />
> So you do not know much? <br />

Bot: i don t know . . . <br />
> Have you seen any movies? <br />

Bot: no . . . . . <br />
> You do not watch movies? <br />

Bot: i don t know . . . <br />
> Do you know who harry potter is? <br />

Bot: no . . . <br />
> ok <br />

Bot: you can t do it . . . <br />
> can you tell me something funny? <br />

Bot: what ? ? ? ? ? <br />
> You tell me <br />

Bot: i didn t know you . . <br />
> Do you want to know my name? <br />

Bot: i don t know . . . <br />
> ok, I think we are done talking <br />

Bot: you re not gonna change . . . <br />
> I do not need to change <br />

Bot: you don t know . . . <br />
> Sorry, I have to go. I need to deliver this. <br />

Bot: what ? . . . . <br />
> bye <br />

Bot: bye . . . . . <br />

## References
* Oriol Vinyals and Quoc V. Le. A neural conversational model.CoRR, abs/1506.05869, 2015.
* Minh-Thang Luong, Hieu Pham, and Christopher D. Manning. Effective approaches to attention-basedneural machine 
translation.CoRR, abs/1508.04025, 2015.
