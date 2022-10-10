#%%

import jax.numpy as jnp
import numpy as np
import jax
from jax import grad, jit, vmap, value_and_grad
from jax.random import PRNGKey as Key
from functools import partial

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

    @partial(jit, static_argnums=(0,))
    def calculate_neg_item_prob(self, params, item_idx, basket_idxs, nonzero):
        """ Function to calculate the probability of observing a particular 
            item given its context when the item is not part of the basket; i.e.
            the item is a negative sample

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
        
        # sum all the context embeddings
        basket_sum = jnp.sum(basket_context, axis=0)

        # normalize by the number of items in the basket
        basket_avg = basket_sum/jnp.sum(nonzero)
        
        # calculate the similarity (dot product) between the normalized 
        # sum of the context embeddings (i.e. average context embedding) 
        # and the embedding of the item of interest
        similarity = item_emb@basket_avg

        # transform similarity score into a probability with a sigmoid function
        item_prob = jax.nn.sigmoid(similarity)

        return item_prob

    # TODO: Where to put this? Does it need to be jitted?
    # define vectorized version of the function
    batched_probs_neg = vmap(calculate_neg_item_prob, in_axes=(None, None, 0, None, None))

    # TODO: check this calculation
    @partial(jit, static_argnums=(0,))
    def supplier_log_prior(self, params, item_idx, rho_var, alpha_var):
        """ Function to calculate the log prior for a single item. This term
            acts as a regulizer by penalizing embedded representations with
            large values.
            
        Parameters:
        ----------
        params : list 
            Contains the two matrices of parameters.
        
        item_idx : int
            Identifier of the item of interest

        rho_var: float
            Prior for the variance of the embeddings matrxi

        alpha_var: float
            Prior for the variance of the context matrix
        
        Returns:
        ----------
        log_prior: float
            value of log prior
        """
        
        # embeddings vector
        rho = params[0]
        # context vector
        alpha = params[1]

        # calculate the L2 norm of each vector
        rho_term = -rho[item_idx].T@rho[item_idx]/(2*rho_var)
        alpha_term = -alpha[item_idx].T@alpha[item_idx]/(2*alpha_var)
        
        log_prior = rho_term + alpha_term

        return log_prior

    # TODO: Where to put this
    # define vectorized version of the function
    batched_log_prior = vmap(supplier_log_prior, in_axes=(None, None, 0, None, None))

    @partial(jit, static_argnums=(0,))
    def per_basket_loss(self, params, items_idxs, nonzero, ns_idxs, rho_var, alpha_var):
        """ Function to calculate the main components of the loss for a single
            basket of items.
            
        Parameters:
        ----------
        params : list 
            Contains the two matrices of parameters.
        
        items_idx : int
            Identifier of the item of interest

        rho_var: float
            Prior for the variance of the embeddings matrxi

        alpha_var: float
            Prior for the variance of the context matrix
        
        Returns:
        ----------
        pos_loss: float
            loss associated with the positive items in the basket (probabilities
            of positive items should be as close as possible to 1)

        neg_loss: float
            loss associated with the negative samples (probabilities of negative
            items should be as close as possible to 0)
        
        regularization_loss: float
            loss associated with the log prior/regularization term

        """

        # 1. positive component
        pos_probs = self.batched_probs(params, items_idxs, items_idxs, nonzero)
        # add a small quantity to avoid small probabilities becoming zero
        # before taking the log
        pos_log_probs = jnp.log(pos_probs + 1e-5)
        # make log probability for padding supplier equal to zero
        pos_log_probs = pos_log_probs*nonzero.flatten()
        # sum all log probabilities
        pos_loss = -jnp.sum(pos_log_probs)

        # 2. negative component
        neg_probs = 1 - self.batched_probs_neg(params, ns_idxs, items_idxs, nonzero)
        # add a small quantity to avoid small probabilities becoming zero
        # before taking the log
        neg_loss = -jnp.sum(jnp.log(neg_probs + 1e-5))

        # 3. log prior/regularization component
        log_priors = self.batched_log_prior(params, items_idxs, rho_var, alpha_var)
        regularization_loss = -jnp.sum(log_priors)
        
        return pos_loss, neg_loss, regularization_loss

    # TODO: Where to put this
    # define vectorized version of the function
    batched_loss = vmap(per_basket_loss, in_axes=(None, None, 0, 0, 0, None, None))


    @partial(jit, static_argnums=(0,))
    def complete_loss(self, params, all_items_idxs, all_nonzero, all_ns_idxs, rho_var, alpha_var, zero_factor, total_N, total_S):
        """ Function to calculate the main components of the loss for a single
            basket of items.
            
        Parameters:
        ----------
        params : list 
            Contains the two matrices of parameters.
        
        items_idx : int
            Identifier of the item of interest

        rho_var: float
            Prior for the variance of the embeddings matrxi

        alpha_var: float
            Prior for the variance of the context matrix
        
        Returns:
        ----------
        pos_loss: float
            loss associated with the positive items in the basket (probabilities
            of positive items should be as close as possible to 1)

        neg_loss: float
            loss associated with the negative samples (probabilities of negative
            items should be as close as possible to 0)
        
        regularization_loss: float
            loss associated with the log prior/regularization term

        """
        
        # get the all the components of the loss
        pos_loss, neg_loss, regularization_loss = self.batched_loss(params, all_items_idxs, all_nonzero, all_ns_idxs, rho_var, alpha_var)
        
        # elements to calculate the scaling factors
        batch_size = all_items_idxs.shape[0]
        num_ns = all_ns_idxs.shape[1]
        # batch scaling ratio
        batch_scaling = total_N/batch_size
        
        # calculate the factor to scale positive observations
        pos_scaling_factor = batch_scaling
        
        # calculate the factor to scale negative observations
        # we scale by the batch size, by the ratio of negative samples
        # to total amount of possible suppliers and by a hyperparameters
        neg_scaling_factor = zero_factor*batch_scaling*((total_S - 1)/(num_ns))
        
        return pos_scaling_factor*jnp.sum(pos_loss) + neg_scaling_factor*jnp.sum(neg_loss) + regularization_loss

    # Loss with gradients. We use argnums to specify the position of the
    # parameters of the model
    grad_loss = jit(value_and_grad(complete_loss, argnums=1))
        