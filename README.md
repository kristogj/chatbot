# Chatbot using Recurrent Neural Networks with Attention

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

Attention mechanism is also added to the decoder. There are several ways of doing this, but this chatbot is using
the one presented by Luong. It uses global attention, which means that instead of just considering the current hidden 
state from the encoder, it considers all of the encoders hidden states. A simple figure explaining how 
attention weights are applied to the encoder outputs to produce the attended decoder inputs is shown in the image 
under. 

![Attention](/images/attention.png)  

## Results
![Loss](/images/loss_.png)

**Chat:** <br />

>I am home <br />

Bot: ok . is there . . .  <br />
> you could be more happy about it <br />

Bot: i know . . . ! <br />
> so smile then <br />

Bot: i m trying to watch it up . <br />
> ok, could you help me with something? <br />

Bot: you re in the city . . . <br />
> we could talk over video <br />

Bot: oh . . . . . <br />
> you know how to do that? <br />

Bot: not a man . . . . <br />
> ok. <br />

Bot: ok . how s the phone soda ? <br />
> stop, could you help me with this task? <br />

Bot: hmm... i am not sure I can help you with that. <br />
> why not? <br />

Bot: because i have to ! ! ! <br />
> you do not have to <br />

Bot: why not ? ? ? ? <br />
> I can do it myself <br />

Bot: why ? ? ? . . <br />
> I need to deliver i soon <br />

Bot: i will . . . . <br />
> you will help? <br />

Bot: it s possible . . . . <br />
> let me know then <br />

Bot: no <br />
> are you stupid? <br />

Bot: no . . times . . <br />
> times what? <br />

Bot: your butter wine was a suspect . <br />
> now you are talking weird <br />

Bot: they re both they re dying . ! <br />
> did you kill them? <br />

Bot: doesn t matter . ? ? ? <br />
> YES! <br />

Bot: and you see that ? ? ? <br />
> i did not see it, i promise <br />

Bot: you did . . . ! <br />
> I have to go now <br />

Bot: not for you ? ? ? <br />
> bye <br />

Bot: wait ! . . .



## References
* Oriol Vinyals and Quoc V. Le. A neural conversational model.CoRR, abs/1506.05869, 2015.
* Minh-Thang Luong, Hieu Pham, and Christopher D. Manning. Effective approaches to attention-based neural machine 
translation.CoRR, abs/1508.04025, 2015.
