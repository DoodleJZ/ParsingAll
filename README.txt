Before starting, put the dataset, pre-trained embeddings and elmo options and weight in data/ directory and setup the paths for data and embeddings.
To train the model, simply run run.sh
And use test.sh to test each model after setting the model path.

Requirements:
python 3.6
allennlp
pytorch=0.4.x
cython
gensim
pytorch-pretrained-bert 