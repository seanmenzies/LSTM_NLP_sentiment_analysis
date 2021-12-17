# LSTM_NLP_sentiment_analysis

## read_data
- Amazon customer review dataset
- read csv data from file as df
- create histogram to visualise ratings
- create wordcloud of both positive and negative reviews
- create 'sentiments' column that assign -1 to negative (1, 2) reviews, +1 to positive (4, 5) reviews and remove neutral (3) reviews
- remove punctuation

## LSTM model
- use functions in previous py file to load and clean data
- split data into training and testing data and labels
- tokenise and pad data
- build neural network with an embedding, two LSTM and one dense output layers
- tanh activation function used as it spans -1 - +1 dimensions
- run for only one epoch due to large vocabulary size and dataset
- compile and fit model

## data_collection
- scrape online articles, isolate <p> bodies and write to txt files
- tokenise new data using the same parameters as in the model
- load model and predict using new data
