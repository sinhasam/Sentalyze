The project is an implementation of a Bidirectional LSTM with 2D Max Pooling according to: https://arxiv.org/abs/1611.06639.

The project is done primarily in PyTorch, and is IMDB review data is used for the training of the data. 

Note: the reason for using negative log likelihood instead of cross entropy is because the model itself has a softmax in the last layer (got asked about it once). 
