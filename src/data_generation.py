#%%

import numpy as np
import pandas as pd
import jax.numpy as jnp
from jax.random import PRNGKey as Key
import jax

def gen_indexes(df_original, basket_col, item_col):
    """ Function to generate indexes for all baskets and items.
        Generated indexes will go from 1 up until the number of baskets and
        the number of items in the data

    Parameters:
    ----------

    df : pandas dataframe
        Dataframe with two columns containing pairs of basket identifiers 
        and item identifiers
    
    basket_col : str
        Name of column containing the basket identifier
    
    item_col : str
        Name of column containing the item identifier


    Returns:
    ----------
    df_original : pandas dataframe
        Contains the same raw data plus two new columns with the generated
        identifiers for baskets and columns
    
    basket2idx : dict
        Maps original basket identifiers to the new identifiers
    
    item2idx : dict
        Maps original item identifiers to the new identifiers
    """
    
    # make a copy of the data
    df = df_original.copy()
    # transform the basket column into a new categorical variable
    basket_cat = df[basket_col].astype('category')
    # we start categories from index 1 so that 0 can be the padding index
    idx2basket = dict(enumerate(basket_cat.cat.categories, start=1))
    basket2idx = {v:k for k,v in idx2basket.items()}
    # apply transformation
    df["basket_idx"] = df[basket_col].apply(lambda x: basket2idx[x])
    # transform the item column into a new categorical variable
    item_cat = df[item_col].astype('category')
    # we start categories from index 1 so that 0 can be the padding index
    idx2item = dict(enumerate(item_cat.cat.categories, start=1))
    item2idx = {v:k for k,v in idx2item.items()}
    df["item_idx"] = df[item_col].apply(lambda x: item2idx[x])
    
    return df, basket2idx, item2idx

def drop_items(df, min_times):
    """ Function to drop items that appear less than a user-defined
        number of times. Generating good embedded representations for
        items that appear very few times is very difficult.

    Parameters
    ----------
    df : pandas dataframe 
        Containing the indexes of baskets and items after creating them using
        gen_indexes() function
    
    min_times : int
        Minimum number of times a number needs to appear in the data in order
        to be preserved

    Returns
    ----------
    df : pandas dataframe
        After removing items

    """

    temp = df.loc[:, ["basket", "item"]].groupby('item').count()
    temp.columns = ['item_count']
    df = df.merge(temp, left_on="item", right_index=True)
    df = df[(df["item_count"] >= min_times)]
    df.drop(columns=["item_count"], inplace=True)
    df.reset_index(inplace=True, drop=True)
    return df

def check_baskets(df, min_items=2):
    """ Function to drop baskets that have less than the user-define
        number of items.

    Parameters
    ----------
    df : pandas dataframe 
        Containing the indexes of baskets and items after creating them using
        gen_indexes() function
    
    min_items : int, default=2
        Minimum number of items in a basket for it to be kept in the data

    Returns
    ----------
    df : pandas dataframe
        After removing baskets
    """

    # check if there are baskets with less than min_items
    mask = df.groupby(['basket']).size() >= min_items
    idxs = mask.index[np.where(mask==True)[0]]
    # remove them
    df = df.loc[df["basket"].isin(idxs)]
    df.reset_index(inplace=True, drop=True)
    return df


def gen_data(df, basket_idxs=None, max_items=None):
    """ Function to format the data appropriately for a set of
        given basket indexes
        
    Parameters
    ----------
    df : pandas dataframe 
        Containing the indexes of baskets and items after creating them using
        gen_indexes() function
    
    basket_idxs : list, default = None
        Containing the indices of the baskets of interest

    max_items : int, default = None 
        Establish a maximum number of items per basket. Any basket that exceeds
        this number will be truncated by sampling 'max_items' from all items
        in the basket. If the parameter is set to 'None' all baskets will be
        padded using the maximum number of items that any single basket in the
        data has.
    
    Returns
    ----------
    
    items_per_basket : dict
        Where keys represent the index of a basket and the values are lists
        with the items that are part of the basket

    all_baskets_idxs : jnp array (num_baskets, )
        Contains all of indexes of the baskets in the data

    all_items_idxs : jnp array (num_baskets, max_items)
        Each row contains the indexes of all the items that are part of a
        particular basket. Padding is added (index 0) so that all rows have
        the same length

    all_nonzero : jnp array (num_baskets, max_items)
        Contains zeros and ones distinguishing items from padding
    """
    
    # subset the dataset to focus only on the provided indices
    if basket_idxs is not None:
        df = df.loc[df["basket_idx"].isin(basket_idxs)]
    
    # group all the items from the same basket together
    temp = dict(tuple(df[['basket_idx', 'item_idx']].groupby('basket_idx')))
        
    # simplify the dictionary by only storing the set of unique items involved
    # in the transaction. Additionally, through this process, we will also capture
    # the number of items in each basket
    items_per_basket = {}
    num_items = []
    for key, value in temp.items():
        # get all the items for a given basket
        items_involved = list(set(value["item_idx"]))
        # order the items (this will be important for the negative sampling)
        items_involved.sort()
        items_per_basket[key] = items_involved
        num_items.append(len(items_per_basket[key]))
        
    # get the maximum number of items among all baskets
    real_max_items = np.max(np.array(num_items))
    
    # create list for storing the final data
    items_idxs = []
    nonzero_items = []
    if max_items is not None:
        # for the baskets with more items than the max number of items
        # allowed, we will randomly subsample a set of items. For the other
        # baskets we will possibly need to add padding

        # iterate through all baskets
        for items, length in zip(items_per_basket.values(), num_items):
            if length > max_items:

                # TODO: control this source of randomness
                sampling_seed = np.random.randint(low=0, high=2**63, size=1)[0]
                items_idxs.append(jax.random.choice(Key(sampling_seed), a=jnp.array(items), shape=(max_items,), replace=False))
                
                # create a vector signaling the position of all the nonzero elements (nowhere)
                nonzero = np.ones((max_items,))
                nonzero_items.append(jnp.array(nonzero).reshape((max_items, 1)))
           
            else:
                # add 0 padding until the max_items value
                len_padding = max_items - length
                items_idxs.append(jnp.array(items + [0]*len_padding))
                
                # create a vector signaling the position of all the nonzero elements
                nonzero = np.zeros((max_items,))
                # change the 0 for a 1 in the positions where there are items
                nonzero[list(range(length))] = 1
                nonzero_items.append(jnp.array(nonzero).reshape((max_items,1)))
    else:
        for items, length in zip(items_per_basket.values(), num_items):
            # add 0 padding up until the nmaximum umber of items in any given basket
            len_padding = real_max_items - length
            items_idxs.append(jnp.array(items + [0]*len_padding))
            
            # create a vector signaling the position of all the nonzero elements
            nonzero = np.zeros((real_max_items,))
            # change the 0 for a 1 in the positions where there are items
            nonzero[list(range(length))] = 1
            nonzero_items.append(jnp.array(nonzero).reshape((real_max_items,1)))

    
    return items_per_basket, jnp.array(list(items_per_basket.keys())), jnp.array(items_idxs), jnp.array(nonzero_items)




# def data_split(test_perc, val_perc, all_consumers, all_suppliers, all_nonzero):
#     """ Function to split the jnp arrays into train, test and validation
#         according to the given percentages
#     """

#     # 1. train-test split
#     test_size = int(all_consumers.shape[0]*test_perc)
#     test_idxs = np.random.choice(list(range(all_consumers.shape[0])), size=test_size, replace=False)

#     all_consumers_test = jnp.take(all_consumers, test_idxs, axis=0)
#     all_consumers_train = jnp.delete(all_consumers, test_idxs)

#     all_suppliers_test = jnp.take(all_suppliers, test_idxs, axis=0)
#     all_suppliers_train = jnp.delete(all_suppliers, test_idxs, axis=0)

#     all_nonzero_test = jnp.take(all_nonzero, test_idxs, axis=0)
#     all_nonzero_train = jnp.delete(all_nonzero, test_idxs, axis=0)

#     # 2. test-validation split
#     val_size = int(all_consumers_test.shape[0]*val_perc)
#     val_idxs = np.random.choice(list(range(all_consumers_test.shape[0])), size=val_size, replace=False)

#     all_consumers_val = jnp.take(all_consumers_test, val_idxs, axis=0)
#     all_consumers_test = jnp.delete(all_consumers_test, val_idxs)
#     print(all_consumers_train.shape, all_consumers_test.shape, all_consumers_val.shape)

#     all_suppliers_val = jnp.take(all_suppliers_test, val_idxs, axis=0)
#     all_suppliers_test = jnp.delete(all_suppliers_test, val_idxs, axis=0)
#     print(all_suppliers_train.shape, all_suppliers_test.shape, all_suppliers_val.shape)

#     all_nonzero_val = jnp.take(all_nonzero_test, val_idxs, axis=0)
#     all_nonzero_test = jnp.delete(all_nonzero_test, val_idxs, axis=0)
#     print(all_nonzero_train.shape, all_nonzero_test.shape, all_nonzero_val.shape)

#     # prepare data for returning
#     consumers_data = [all_consumers_train, all_consumers_test, all_consumers_val]
#     suppliers_data = [all_suppliers_train, all_suppliers_test, all_suppliers_val]
#     nonzero_data = [all_nonzero_train, all_nonzero_test, all_nonzero_val]

#     return (consumers_data, suppliers_data, nonzero_data)