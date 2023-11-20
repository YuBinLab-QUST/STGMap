# STGMap
A universal framework for spatial transcriptomics data mining with interpretable unsupervised graph representation learning


STGMap is implemented with the help of Python 3.6 and PyTorch. For the structural information embedding module, the dropout is set to 0.1 to prevent overfitting of the data in the network. For the feature information embedding module, the weights of the structural positive embedding and the neighbor positive embedding are set to 0.8 and 0.2, the hidden dimension of the MLP is set to 512, the output pipeline is the number of genes in the corresponding data, and the number of neighbors used to generate the neighbor positive embedding is set to 4. For the multi-path loss module, the weighting ratio of the triple loss is set to 10:10:1. The STGMap was trained and optimized by the Adam optimizer, with the learning rate set to 0.0025 and the weight decay set to 0.0001. The number of times the model was trained was 500.
