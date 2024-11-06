import numpy as np

from scmkl.train_model import train_model
from scmkl.test import find_selected_groups


def optimize_sparsity(adata, group_size, starting_alpha = 1.9, increment = 0.2, target = 1, n_iter = 10):
    '''
    Iteratively train a grouplasso model and update alpha to find the 
    parameter yielding the desired sparsity.
    
    Parameters
    ----------
    **adata** : *AnnData*
        > `AnnData` with `'Z_train'` and `'Z_test'` in 
        `adata.uns.keys()`.

    **group_size** : *int* 
        > Argument describing how the features are grouped. Should be
        `2 * D`. For more information see 
        [celer documentation](https://mathurinm.github.io/celer/generated/celer.GroupLasso.html).

    **starting_alpha** : *float*
        > The alpha value to start the search at.
    
    **increment** : *float* 
        > Amount to adjust alpha by between iterations.
    
    **target** : *int*
        > The desired number of groups selected by the model.

    **n_iter** : *int*
        > The maximum number of iterations to run.
            
    Returns
    -------
    **sparsity_dict** : *dict*
        > Tested alpha as keys and the number of selected pathways as 
        the values.
        
    **alpha** : *float*
        >The alpha value yielding the number of selected groups closest 
        to the target.

    Examples
    --------
    >>> sparcity_dict, alpha = scmkl.optimize_sparsity(adata, (2 * D), target = 1)
    >>>
    >>> alpha
    0.01
    '''
    assert increment > 0 and increment < starting_alpha, 'Choose a positive increment less than alpha'
    assert target > 0 and isinstance(target, int), 'Choose an integer target number of groups that is greater than 0'
    assert n_iter > 0 and isinstance(n_iter, int), 'Choose an integer number of iterations that is greater than 0'

    sparsity_dict = {}
    alpha = starting_alpha

    for _ in np.arange(n_iter):
        adata = train_model(adata, group_size, alpha)
        num_selected = len(find_selected_groups(adata))

        sparsity_dict[np.round(alpha,4)] = num_selected

        if num_selected < target:
            #Decreasing alpha will increase the number of selected pathways
            if alpha - increment in sparsity_dict.keys():
                # Make increment smaller so the model can't go back and forth between alpha values
                increment /= 2
            alpha = np.max([alpha - increment, 1e-1]) #Ensures that alpha will never be negative
        elif num_selected > target:
            if alpha + increment in sparsity_dict.keys():
                increment /= 2
            alpha += increment
        elif num_selected == target:
            break

    # Find the alpha that minimizes the difference between target and observed number of selected groups
    optimal_alpha = list(sparsity_dict.keys())[np.argmin([np.abs(selected - target) for selected in sparsity_dict.values()])]
    return sparsity_dict, optimal_alpha