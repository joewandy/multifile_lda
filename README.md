# multifile_lda

This model slightly extends the standard Latent Dirichlet Allocation (LDA), commonly used for the unsupervised discovery of topics in a document.

Our problem is we have multiple collections of documents that we think should share the same set of topics. In this multi-file LDA model, within each collection, we still find the topic-to-document assignments, but now the topics are also shared across collections (files).

TODO:
- Implement faster Gibbs sampling using Numba/Cython
- Implement online variational inference
- ~~Use spare matrices to store the counts~~
- Plug in PyLDAVis or some other visualisation module
