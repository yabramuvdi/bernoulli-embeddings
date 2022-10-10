#%%

import numpy as np
import pandas as pd
import jax.numpy as jnp

#%%


def gen_indexes(df):
    """ Function to generate indexes for all sellers and consumers.
        Requires a column named "consumer" and one named "supplier"
    """
     
    suppliers_cat = df["supplier"].astype('category')
    # we start categories on 1 so that 0 can be the padding index
    idx2supplier = dict(enumerate(suppliers_cat.cat.categories, start=1))
    supplier2idx = {v:k for k,v in idx2supplier.items()}
    df["supplier_idx"] = df["supplier"].apply(lambda x: supplier2idx[x])
    
    consumer_cat = df["consumer"].astype('category')
    idx2consumer = dict(enumerate(consumer_cat.cat.categories, start=1))
    consumer2idx = {v:k for k,v in idx2consumer.items()}
    df["consumer_idx"] = df["consumer"].apply(lambda x: consumer2idx[x])
    
    return df, supplier2idx, consumer2idx

def data_split(test_perc, val_perc, all_consumers, all_suppliers, all_nonzero):
    """ Function to split the jnp arrays into train, test and validation
        according to the given percentages
    """

    # 1. train-test split
    test_size = int(all_consumers.shape[0]*test_perc)
    test_idxs = np.random.choice(list(range(all_consumers.shape[0])), size=test_size, replace=False)

    all_consumers_test = jnp.take(all_consumers, test_idxs, axis=0)
    all_consumers_train = jnp.delete(all_consumers, test_idxs)

    all_suppliers_test = jnp.take(all_suppliers, test_idxs, axis=0)
    all_suppliers_train = jnp.delete(all_suppliers, test_idxs, axis=0)

    all_nonzero_test = jnp.take(all_nonzero, test_idxs, axis=0)
    all_nonzero_train = jnp.delete(all_nonzero, test_idxs, axis=0)

    # 2. test-validation split
    val_size = int(all_consumers_test.shape[0]*val_perc)
    val_idxs = np.random.choice(list(range(all_consumers_test.shape[0])), size=val_size, replace=False)

    all_consumers_val = jnp.take(all_consumers_test, val_idxs, axis=0)
    all_consumers_test = jnp.delete(all_consumers_test, val_idxs)
    print(all_consumers_train.shape, all_consumers_test.shape, all_consumers_val.shape)

    all_suppliers_val = jnp.take(all_suppliers_test, val_idxs, axis=0)
    all_suppliers_test = jnp.delete(all_suppliers_test, val_idxs, axis=0)
    print(all_suppliers_train.shape, all_suppliers_test.shape, all_suppliers_val.shape)

    all_nonzero_val = jnp.take(all_nonzero_test, val_idxs, axis=0)
    all_nonzero_test = jnp.delete(all_nonzero_test, val_idxs, axis=0)
    print(all_nonzero_train.shape, all_nonzero_test.shape, all_nonzero_val.shape)

    # prepare data for returning
    consumers_data = [all_consumers_train, all_consumers_test, all_consumers_val]
    suppliers_data = [all_suppliers_train, all_suppliers_test, all_suppliers_val]
    nonzero_data = [all_nonzero_train, all_nonzero_test, all_nonzero_val]

    return (consumers_data, suppliers_data, nonzero_data)

def gen_data(df, consumers_idxs, max_suppliers=None):
    """ Function to format the data appropriately for a set of
        given consumer indices
        
    Args
    ----------
        - df (pandas): containing the firm2firm transactions
        - consumers_idxs (list): containing the indices of the consumers
          of interest
        - max_suppliers (int): stablish a maximum number of supplier 
          per transaction
    
    Returns
    ----------
        - jnp array with the consumers
        - jnp array where each row contain the indices of the suppliers
          transacting with a particular consumer (plus 0 paddings)
        - jnp array with zeros and ones distinguishing suppliers from padding

    """
    
    # subset the dataset to focus only on the provided indices
    df = df.loc[df["consumer_idx"].isin(consumers_idxs)]
    
    # group all the transactions
    temp = dict(tuple(df[['consumer_idx', 'supplier_idx']].groupby('consumer_idx')))
        
    # simplify the dictionary by only storing the set of unique items involved
    # the transaction. Additionally, through this process, I will also capture
    # the number of items in each transaction
    sup_per_consumer = {}
    num_suppliers = []
    for key, value in temp.items():
        # get all the suppliers for a giver consumer
        suppliers_involved = list(set(value["supplier_idx"]))
        # order the suppliers (this will be important for the negative sampling)
        suppliers_involved.sort()
        sup_per_consumer[key] = suppliers_involved
        num_suppliers.append(len(sup_per_consumer[key]))
        
    # get the maximum number of suppliers that any consumer has
    real_max_sup = np.max(np.array(num_suppliers))
    suppliers_idxs = []
    nonzero_suppliers = []
    
    if max_suppliers is not None:
        # for the transactions with more items than the max number of items
        # allowed, we will randomly subsample a set of items. For the other
        # transactions, we will possibly need to add padding
        for suppliers, length in zip(sup_per_consumer.values(), num_suppliers):
            if length > max_suppliers:
                sampling_seed = np.random.randint(low=0, high=2**63, size=1)[0]
                suppliers_idxs.append(jax.random.choice(Key(sampling_seed), a=jnp.array(suppliers), shape=(max_suppliers,), replace=False))
                
                # create a vector signaling the position of all the nonzero elements (nowhere)
                nonzero = np.ones((max_suppliers,))
                nonzero_suppliers.append(jnp.array(nonzero).reshape((max_suppliers,1)))
           
            else:
                # add 0 padding until the max_items value
                len_padding = max_suppliers - length
                suppliers_idxs.append(jnp.array(suppliers + [0]*len_padding))
                
                # create a vector signaling the position of all the nonzero elements
                nonzero = np.zeros((max_suppliers,))
                # change the 0 for a 1 in the positions where there are items
                nonzero[list(range(length))] = 1
                nonzero_suppliers.append(jnp.array(nonzero).reshape((max_suppliers,1)))
    else:
        for suppliers, length in zip(sup_per_consumer.values(), num_suppliers):
            # add 0 padding up until the nmaximum umber of items in any given transaction
            len_padding = real_max_sup - length
            suppliers_idxs.append(jnp.array(suppliers + [0]*len_padding))
            
            # create a vector signaling the position of all the nonzero elements
            nonzero = np.zeros((real_max_sup,))
            # change the 0 for a 1 in the positions where there are items
            nonzero[list(range(length))] = 1
            nonzero_suppliers.append(jnp.array(nonzero).reshape((real_max_sup,1)))

    
    consumers_idxs = jnp.array(list(sup_per_consumer.keys()))
    
    return sup_per_consumer, consumers_idxs, jnp.array(suppliers_idxs), jnp.array(nonzero_suppliers)

def single_neg_sample(generator, pos_inds, n_items, n_samp):
    """ Pre-verified with binary search `pos_inds` is assumed to be ordered
    """
    raw_samp = generator.integers(1, n_items - len(pos_inds) + 1, size=n_samp)
    pos_inds_adj = pos_inds - np.arange(len(pos_inds))
    ss = np.searchsorted(pos_inds_adj, raw_samp, side='right')
    neg_inds = raw_samp + ss
    return jnp.array(neg_inds)