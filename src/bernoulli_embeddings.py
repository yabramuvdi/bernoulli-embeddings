#%%
import jax.numpy as jnp
import numpy as np
import jax
from jax import grad, jit, vmap, value_and_grad
from jax.random import PRNGKey as Key
from jax.example_libraries import optimizers
from functools import partial
from typing import NamedTuple
import optax

import math
import time
import matplotlib.pyplot as plt

# Implementation of Exponential Family Embeddings (Rudolph et al., 2016) 
# with a binary outcome.
# More description...

def single_neg_sample(pos_inds, n_items, n_samp, generator):
    """ Function to draw a negative sample
    
        Pre-verified with binary search `pos_inds` is assumed to be ordered
        Source: https://medium.com/@2j/negative-sampling-in-numpy-18a9ad810385
    """
    raw_samp = generator.integers(1, n_items - len(pos_inds) + 1, size=n_samp)
    pos_inds_adj = pos_inds - np.arange(len(pos_inds))
    ss = np.searchsorted(pos_inds_adj, raw_samp, side='right')
    neg_inds = raw_samp + ss
    return jnp.array(neg_inds)

class Params(NamedTuple):
    """Class for storing the two matrices of parameters for the model"""
    rho: jnp.ndarray
    alpha: jnp.ndarray

def gen_model_args(seed, num_items, num_baskets, 
                   embedded_dim=300, init_mean=0.0, init_std=1.0,
                   rho_var=1.0, alpha_var=1.0, zero_factor=0.1,
                   num_epochs=1000, batch_size=32, ns_multiplier=50, 
                   num_ns=10, print_loss_freq=100, save_params_freq=100):
    """ 
    Function to initialize all the arguments of the model

    Attributes
    ----------
    seed : int
        Controls the randomness of all functions within the class
    
    num_items: int
        Number of unique items that will get an embedded representation

    num_baskets: int
        Number of baskets in the data

    embedded_dim: int, default=300
        Desired dimension for embedded representations

    init_mean: float, default=0.0
        Mean of the Normal distribution that will generate the initial embedded
        representations

    init_std: float, default=1.0
        Standard deviation of the Normal distribution that will generate 
        the initial embedded representations

    rho_var: float
        Prior for the variance of the embeddings matrxi

    alpha_var: float
        Prior for the variance of the context matrix

    zero_factor: float
        Scaling factor for the portion of the loss related to negative sampling

    
    Returns
    ----------
    args_dict: dict
        Contains a named dictionary will all the arguments

    generator:

    """

    args_dict = {"seed": seed,
                 "num_items": num_items,
                 "num_baskets": num_baskets,
                 "embedded_dim": embedded_dim,
                 "init_mean": init_mean,
                 "init_std": init_std,
                 "rho_var": rho_var,
                 "alpha_var": alpha_var,
                 "zero_factor": zero_factor,
                 "num_epochs": num_epochs, 
                 "batch_size": batch_size, 
                 "ns_multiplier": ns_multiplier, 
                 "num_ns": num_ns, 
                 "print_loss_freq": print_loss_freq,
                 "save_params_freq": save_params_freq,
                 }

    return args_dict, np.random.default_rng(seed)

def init_params(model_args, generator):
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

        rho = jnp.array(generator.normal(loc=model_args["init_mean"], scale=model_args["init_std"], size=(model_args["num_items"], model_args["embedded_dim"])))
        alpha = jnp.array(generator.normal(loc=model_args["init_mean"], scale=model_args["init_std"], size=(model_args["num_items"], model_args["embedded_dim"])))
        
        # add a vector of zeros to the start of both embeddings matrices
        # this row should representation the embedding of the padding item
        rho = jnp.append(jnp.array([[0]*model_args["embedded_dim"]]), rho, axis=0)
        alpha = jnp.append(jnp.array([[0]*model_args["embedded_dim"]]), alpha, axis=0)

        return Params(rho=rho, alpha=alpha)


@jit
def calculate_item_prob(params, item_idx, basket_idxs, nonzero):
    """ Function to calculate the probability of observing a particular 
        item given its context for an item that is part of the observed
        basket.

    Parameters:
    ----------
    params : tuple 
        Contains the two arrays of parameters (rho, alpha).
    
    item_idx : int
        Identifier of the item of interest

    basket_idxs: jnp array of shape (max_items, )
        Contains the identifiers of the items in the basket plus padding.
        This basket includes the target item as well.

    nonzero : jnp array of shape (max_items, )
        Signals the positions of the array where the items of the basket
        are located; the rest of the array contains padding.


    Returns:
    ----------
    prob : float
        Probability of observing the target item within the given basket 
        given the current embedded representations (rho, alpha).

    """
    
    # embeddings vector
    rho = params.rho
    # context vector
    alpha = params.alpha

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

# define a vectorized version of the function that iterates through all items in the basket
batched_probs = jit(vmap(calculate_item_prob, in_axes=(None, 0, None, None)))

@jit
def calculate_neg_item_prob(params, item_idx, basket_idxs, nonzero):
    """ Function to calculate the probability of observing a particular 
        item given its context when the item is not part of the basket; i.e.
        the item is a negative sample

    Parameters:
    ----------
    params : tuple
        Contains the two arrays of parameters (rho, alpha).
    
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
    rho = params.rho
    # context vector
    alpha = params.alpha

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

# define vectorized version of the function
batched_probs_neg = jit(vmap(calculate_neg_item_prob, in_axes=(None, 0, None, None)))


# TODO: check this calculation
@jit
def supplier_log_prior(params, item_idx, model_args):
    """ Function to calculate the log prior for a single item. This term
        acts as a regulizer by penalizing embedded representations with
        large values.
        
    Parameters:
    ----------
    params : tuple
        Contains the two arrays of parameters.
    
    item_idx : int
        Identifier of the item of interest

    model_args: dict
        Contains all the arguments of the model
    
    Returns:
    ----------
    log_prior: float
        value of log prior
    """
    
    # embeddings vector
    rho = params.rho
    # context vector
    alpha = params.alpha

    # calculate the L2 norm of each vector
    rho_term = -rho[item_idx].T@rho[item_idx]/(2*model_args["rho_var"])
    alpha_term = -alpha[item_idx].T@alpha[item_idx]/(2*model_args["alpha_var"])
    log_prior = rho_term + alpha_term

    return log_prior

# define vectorized version of the function
batched_log_prior = jit(vmap(supplier_log_prior, in_axes=(None, 0, None)))

@jit
def per_basket_loss(params, items_idxs, nonzero, ns_idxs, model_args):
    """ Function to calculate the main components of the loss for a single
        basket of items.
        
    Parameters:
    ----------
    params : tuple
        Contains the two arrays of parameters.
    
    items_idx : jnp array of shape (max_items, )
        Identifier of all the items in the basket

    nonzero : jnp array of shape (max_items, 1)
        Signals the positions of the array where the items of the basket
        are located.

    ns_idxs: jnp array of shape (num_neg_samples, )
        Identifiers of negative samples
    
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

    # 1. Positive component
    pos_probs = batched_probs(params, items_idxs, items_idxs, nonzero)
    # add a small quantity to avoid small probabilities becoming zero
    # before taking the log
    pos_log_probs = jnp.log(pos_probs + 1e-5)
    # make log probability for padding supplier equal to zero
    pos_log_probs = pos_log_probs*nonzero.flatten()
    # sum all log probabilities
    pos_loss = -jnp.sum(pos_log_probs)

    # 2. Negative component
    neg_probs = 1 - batched_probs_neg(params, ns_idxs, items_idxs, nonzero)
    # add a small quantity to avoid small probabilities becoming zero
    # before taking the log
    neg_loss = -jnp.sum(jnp.log(neg_probs + 1e-5))

    # 3. log prior/regularization component
    log_priors = batched_log_prior(params, items_idxs, model_args)
    regularization_loss = -jnp.sum(log_priors)
    
    return pos_loss, neg_loss, regularization_loss

# define vectorized version of the function
batched_loss = jit(vmap(per_basket_loss, in_axes=(None, 0, 0, 0, None)))

@jit
def complete_loss(params, all_items_idxs, all_nonzero, all_ns_idxs, model_args):
    """ Function to calculate the loss for all the baskets provided
        
    Parameters:
    ----------
    params : tuple
        Contains the two arrays of parameters.
    
    all_items_idx : jnp array of shape (num_baskets, max_items)
        Identifier of all the items in the basket

    all_nonzero : jnp array of shape (num_baskets, max_items)
        Signals the positions of the array where the items of the basket
        are located.

    all_ns_idxs: jnp array of shape (num_baskets, num_neg_samples)
        Identifiers of negative samples
    
    model_args: dict
        Contains all the arguments of the model
    
    Returns:
    ----------
    loss: float
        complete loss for the data

    """
    
    # get the all the components of the loss for all baskets
    pos_loss, neg_loss, regularization_loss = batched_loss(params, all_items_idxs, all_nonzero, all_ns_idxs, model_args)

    # elements to calculate the scaling factors
    batch_size = all_items_idxs.shape[0]
    num_ns = all_ns_idxs.shape[1]
    # batch scaling ratio
    batch_scaling = model_args["num_baskets"]/batch_size
    
    # calculate the factor to scale positive observations
    pos_scaling_factor = batch_scaling
    
    # calculate the factor to scale negative observations
    # we scale by the batch size, by the ratio of negative samples
    # to total amount of possible suppliers and by a hyperparameters
    neg_scaling_factor = model_args["zero_factor"]*batch_scaling*((model_args["num_items"] - 1)/(num_ns))
    
    # sum the loss of all basket and scale its components
    loss = pos_scaling_factor*jnp.sum(pos_loss) + neg_scaling_factor*jnp.sum(neg_loss) + jnp.sum(regularization_loss)
    #loss = pos_scaling_factor*jnp.sum(pos_loss)
    # loss jnp.sum(pos_loss) + jnp.sum(neg_loss)
    
    return loss

# Loss with gradients. We use argnums to specify the position of the
# parameters of the model
# grad_loss = jit(value_and_grad(complete_loss, argnums=1), static_argnums=(0,))

# # TODO: Add better documentation
# def init_optimizer(params, algorithm="adagrad", step_size=0.1, momentum=0.9):
#     """ initialize an optimizer
#     """

#     if algorithm == "adam":
#         optimizer = optax.adam(step_size)
#     elif algorithm == "sgd":
#         optimizer = optax.sgd(step_size)
#     elif algorithm == "adagrad":
#         optimizer = optax.adagrad(step_size, momentum)
#     else:
#         raise TypeError("No valid optimization algorithm selected")

#     # initialize current optimum state with the current parameters
#     opt_state = optimizer.init(params)

#     optimizer_dict = {"optimizer": optimizer,       # initialized optimizer
#                       "opt_state": opt_state,       # current state of the optimizer
#                      }

#     return optimizer_dict 

@partial(jit, static_argnums=(1,))
def update(params, optimizer, opt_state, all_suppliers_idxs, all_nonzero, all_ns_idxs, model_args):
    """ compute the gradient for a batch of data and update parameters
         
        Resources: https://github.com/deepmind/optax
    """
    
    # evaluate the gradients and the value of the loss function
    loss_value, grads = value_and_grad(complete_loss, argnums=0)(params, all_suppliers_idxs, all_nonzero, all_ns_idxs, model_args)
    
    # calculate the optimal update to our current parameters
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    # apply update
    new_params = optax.apply_updates(params, updates)

    return loss_value, new_params, new_opt_state, grads


def train(params, optimizer, opt_state, 
          items_per_basket, all_baskets_idxs, 
          all_items_idxs, all_nonzero,
          model_args, generator, output_path):
              
    # infrastructure for monitoring the training process
    loss_epochs = []
    all_times = []

    print(f"Starting training -------------")
    num_updates = 0
    for epoch in range(model_args["num_epochs"]):
        start_time = time.time()

        # generate negative samples
        if epoch%model_args["ns_multiplier"] == 0:
            neg_time = time.time()
            # get a "bucket" of negative samples (for several epochs)
            bucket_neg_samples = [single_neg_sample(items_per_basket[basket.item()], model_args["num_items"], model_args["num_ns"]*model_args["ns_multiplier"], generator) for basket in all_baskets_idxs]
            bucket_neg_samples = jnp.array(bucket_neg_samples)
            duration = np.round(time.time() - neg_time, 2)
            ns_init_idx = 0
            ns_end_idx = model_args["num_ns"]
            print(f"Bucket of negative samples generated in: {duration/60} minutes")

        # take a random batch of data
        batch_idx = generator.choice(list(range(all_baskets_idxs.shape[0])), size=model_args["batch_size"], replace=False)
        batch_items = jnp.take(all_items_idxs, batch_idx, axis=0)
        batch_nonzero = jnp.take(all_nonzero, batch_idx, axis=0)

        batch_neg_samples = jnp.take(bucket_neg_samples, batch_idx, axis=0)    
        relevant_idx = jnp.array(list(range(ns_init_idx, ns_end_idx)))
        batch_neg_samples = jnp.take(batch_neg_samples, relevant_idx, axis=1)

        # update idxs for negative samples
        ns_init_idx = ns_end_idx
        ns_end_idx += model_args["num_ns"]

        # MAIN STEP: calculate gradients and update parameters
        loss, new_params, new_opt_state, grads = update(params, optimizer, opt_state,
                                                        batch_items, batch_nonzero, 
                                                        batch_neg_samples, model_args)
        # update values of variables
        params = new_params
        opt_state = new_opt_state
        num_updates += 1

        #----
        # trainining diagnostics
        #---
        loss_epochs.append(loss)
        epoch_time = (time.time() - start_time)/60
        all_times.append(epoch_time)

        epoch_message = "Epoch {} in {:0.2f} minutes".format(epoch, epoch_time)
        loss_message = "Loss value: {:,}".format(loss)

        # print progress
        if epoch%model_args["print_loss_freq"] == 0:
            print(epoch_message)
            print(loss_message)

            # plot the loss
            if epoch > 50:
                plt.plot(range(50, len(loss_epochs)), loss_epochs[50:])
                plt.show()
            else:
                plt.plot(range(len(loss_epochs)), loss_epochs[0:])
                plt.show()

        # save params
        # TODO: erase older params
        if epoch%model_args["save_params_freq"] == 0:
            jnp.save(output_path + f"embeddings_epoch_{epoch}.npy", params.rho)
            jnp.save(output_path + f"context_epoch_{epoch}.npy", params.alpha)
            
    return params
            