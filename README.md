# bernoulli-embeddings

This repository provides a Python implementation - relying on the [JAX library](https://github.com/google/jax)- of the Bernoulli Embeddings developed by [Rudolph et al. (2016)](https://arxiv.org/abs/1608.00778). Bernoulli Embeddings are a particular class of Exponential Family Embeddings where dense low-dimensional representations of items are constructued by relying on binary co-occurence patterns. This implementation underlies current work with Vasco M. Carvalho, Stephen Hansen, and Glenn Magerman that estimates these embeddings on 14 million firm-to-firm transactions for Belgium for the year 2014.


For a complete example on how to estimate these embeddings on a particular dataset check [this blog post](https://yabramuvdi.github.io/movies_embeddings_estimation/) which uses movie ratings data to construct embedded representations of movies.
