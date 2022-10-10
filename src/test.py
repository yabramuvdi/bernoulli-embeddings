#%%

import jax.numpy as jnp
import numpy as np
import jax
from jax import grad, jit, vmap, value_and_grad
from jax.random import PRNGKey as Key
from functools import partial

# %%

class BernoulliEmbeddings():
    """ Implementation of Exponential Family Embeddings (Rudolph et al., 2016) 
        with a binary outcome.

    More description...


    Attributes
    ----------
    seed : int
        Controls the randomness of all functions within the class
    
    num_items: int
        Number of unique items that will get an embedded representation

    embedded_dim: int, default=300
        Desired dimension for embedded representations

    init_mean: float, default=0.0
        Mean of the Normal distribution that will generate the initial embedded
        representations

    init_std: float, default=1.0
        Standard deviation of the Normal distribution that will generate 
        the initial embedded representations
    
    """

    def __init__(self, seed, num_items, embedded_dim=300, init_mean=0.0, init_std=1.0):
        self.seed = seed
        self.num_items = num_items
        self.embedded_dim = embedded_dim
        self.init_mean = init_mean
        self.init_std = init_std
        # use seed to initialize the generator
        self.generator = np.random.default_rng(self.seed)
    

    def init_params(self):
        """ Function to generate random initial values for the embedded
            representations. We will use a Normal distribution.
    
        
        Returns:
        ----------
        alpha : jnp array of shape (num_items+1, embedded_dim)
            Matrix containing the initial "context embedding"; Alpha

        rho : jnp array of shape (num_items+1, embedded_dim)
            Matrix containing the initial "XX embedding"; Rho
        
        The two matrices will have an additional row that corresponds to the 
        "embedding" of the index used for padding (index = 0)
        
        """

        rho = jnp.array(self.generator.normal(loc=self.init_mean, scale=self.init_std, size=(self.num_items, self.embedded_dim)))
        alpha = jnp.array(self.generator.normal(loc=self.init_mean, scale=self.init_std, size=(self.num_items, self.embedded_dim)))
        
        # add a vector of zeros to the start of both embeddings matrices
        # this row should representation the embedding of the padding item
        rho = jnp.append(jnp.array([[0]*self.embedded_dim]), rho, axis=0)
        alpha = jnp.append(jnp.array([[0]*self.embedded_dim]), alpha, axis=0)

        self.rho = rho
        self.alpha = alpha
    
    @partial(jit, static_argnums=(0,))
    def calculate_item_prob(self, params, item_idx, basket_idxs, nonzero):
        """ Function to calculate the probability of observing a particular 
            item given its context for an item that is part of the observed
            basket.

        Parameters:
        ----------
        params : list 
            Contains the two matrices of parameters.
        
        item_idx : int
            Identifier of the item of interest

        basket_idxs: jnp array of shape (max_items, )
            Contains the identifiers of the items in the basket plus padding.
            This basket includes the target item as well.

        nonzero : jnp array of shape (max_items, )
            Signals the positions of the array where the items of the basket
            are located.


        Returns:
        ----------
        prob : float
            Probability of observing the target item within the given basket 
            given the current embedded representations (rho, alpha).

        """
        
        # embeddings vector
        rho = params[0]
        # context vector
        alpha = params[1]

        # get the embedding for the item of interest and the 
        # context embeddings for the rest
        item_emb = jnp.take(rho, item_idx, axis=0)
        basket_context = jnp.take(alpha, basket_idxs, axis=0)

        # transform the padding context embeddings to zero
        basket_context = basket_context*nonzero
        
        # sum all the context embeddings and substract the contribution 
        # from the target item
        basket_sum = jnp.sum(basket_context, axis=0) - jnp.take(alpha, item_idx, axis=0)

        # normalize by the number of items in the basket
        basket_avg = basket_sum/(jnp.sum(nonzero)-1)
        
        # calculate the similarity (dot product) between the normalized 
        # sum of the context embeddings (i.e. average context embedding) 
        # and the embedding of the item of interest
        similarity = item_emb@basket_avg

        # transform similarity score into a probability with a sigmoid function
        item_prob = jax.nn.sigmoid(similarity)

        return item_prob

    # TODO: Where to put this? Does it need to be jitted?
    # define vectorized version of the function to calculate the probability 
    # of observing a given item given its context. we will iterate over all
    # the items in the context
    batched_probs = vmap(calculate_item_prob, in_axes=(None, None, 0, None, None))


# %%

#============
# Fake data
#============

# create our baskets of data with padding
baskets = jnp.array([[1,2,3,4], [2,4,0,0], [1,2,3,4]])
nonzero = jnp.array([jnp.array([1,1,1,1]).reshape(4,1), 
                     jnp.array([1,1,0,0]).reshape(4,1),
                     jnp.array([1,1,1,1]).reshape(4,1)
                    ])

print(baskets.shape, nonzero.shape)

# %%

# initialize object
my_emb = BernoulliEmbeddings(seed=92, 
                             num_items=4, 
                             embedded_dim=50, 
                             init_mean=0.0, 
                             init_std=1.0)

# %%

# initialize parameters
my_emb.init_params()
print(my_emb.rho.shape, my_emb.alpha.shape)

# %%

# calculate probability from one item in one basket
my_emb.calculate_item_prob([my_emb.rho, my_emb.alpha], 2, baskets[0], nonzero[0])

# %%

# calculate probabilities for all items in one basket
my_emb.batched_probs([my_emb.rho, my_emb.alpha], baskets[0], baskets[0], nonzero[0])

# %%
