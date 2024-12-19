################################                          
###           HVG            ###   
################################


import numpy as np
import pandas as pd
import scanpy as sc
import scipy.io as sio
import harmonypy as hm


print(sc.__version__)


# filter expression matrices to only include HVGs shared across all datasets
def hvg_selection_and_pooling(exp_paths, n_top_genes=1000):
    # input n expression matrices paths, output n expression matrices with only the union of the HVGs
    all_genes = None
    # read adata and find hvgs
    hvg_bools = []
    for d in exp_paths:
        adata = sio.mmread(d)
        adata = adata.toarray()
        print(adata.shape)
        adata = sc.AnnData(X=adata.T, dtype=adata.dtype)

        if all_genes is None:
            all_genes = adata.var_names
        else:
            # Include only genes that are present in all datasets
            all_genes = all_genes.intersection(adata.var_names)

        # Preprocess the data
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)

        # save hvgs
        hvg = adata.var['highly_variable']
        hvg_bools.append(hvg)

    # find union of hvgs
    hvg_union = hvg_bools[0]
    for i in range(1, len(hvg_bools)):
        print(sum(hvg_union), sum(hvg_bools[i]))
        hvg_union = hvg_union | hvg_bools[i]
    print("Number of HVGs: ", hvg_union.sum())

    # Convert hvg_union to Pandas Series with appropriate indexing
    hvg_union_series = pd.Series(hvg_union.to_numpy(), index=all_genes)
    
    # Get indices where hvg_union_series is True
    # true_indices = hvg_union_series[hvg_union_series].index
    true_indices = hvg_union_series[hvg_union_series].index.astype(int)
    # Create a new full index (1 to 36601 inclusive)
    full_index = range(1, 36602)
    hvg_union_full = pd.Series(False, index=full_index)
    
    # Set values to True for the recorded indices
    hvg_union_full[true_indices] = True
    
    print("Number of HVGs: ", hvg_union_full.sum())
    # Save the hvg_union to npy file
    np.save("../GSE240429/data/filtered_expression_matrices/hvg_union.npy", hvg_union.to_numpy())
    # filter expression matrices
    filtered_exp_mtxs = []
    for d in exp_paths:
        adata = sio.mmread(d)
        adata = adata.toarray()
        adata = sc.AnnData(X=adata.T, dtype=adata.dtype)

        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        filtered_exp_mtxs.append(adata[:, hvg_union].X)
    return filtered_exp_mtxs



exp_paths = ["../GSE240429/data/filtered_expression_matrices/1/matrix.mtx",
            "../GSE240429/data/filtered_expression_matrices/2/matrix.mtx",
            "../GSE240429/data/filtered_expression_matrices/3/matrix.mtx",
            "../GSE240429/data/filtered_expression_matrices/4/matrix.mtx"]
filtered_mtx = hvg_selection_and_pooling(exp_paths)
for i in range(len(filtered_mtx)):
    np.save("../GSE240429/data/filtered_expression_matrices/" + str(i + 1) + "/hvg_matrix.npy", filtered_mtx[i].T)


################################
#! batch correct using harmony
#! Other batch correction methods can be used here in place of harmony. Furthermore, model can be trained using the hvg matrix and achieve comparable results if the datasets used are similar enough

d = np.load("../GSE240429/data/filtered_expression_matrices/1/hvg_matrix.npy")
print(d.shape)

d2 = np.load("../GSE240429/data/filtered_expression_matrices/2/hvg_matrix.npy")
print(d2.shape)

d3 = np.load("../GSE240429/data/filtered_expression_matrices/3/hvg_matrix.npy")
print(d3.shape)

d4 = np.load("../GSE240429/data/filtered_expression_matrices/4/hvg_matrix.npy")
print(d4.shape)

d = np.concatenate((d.T, d2.T, d3.T, d4.T), axis = 0)
print(d.shape)
data_sizes = [2378, 2349, 2277, 2265]
batch_labels = np.concatenate((np.zeros(2378), np.ones(2349), np.ones(2277)*2, np.ones(2265)*3))
batch_labels = batch_labels.astype(str)
df = pd.DataFrame(batch_labels, columns=["dataset"])
print(df.shape)
# # Run the Harmony integration algorithm
harmony = hm.run_harmony(d, meta_data=df, vars_use=["dataset"])
harmony_corrected = harmony.Z_corr.T

#split back into datasets
d1 = harmony_corrected[:data_sizes[0]]
d2 = harmony_corrected[data_sizes[0]:data_sizes[0]+data_sizes[1]]
d3 = harmony_corrected[data_sizes[0]+data_sizes[1]:data_sizes[0]+data_sizes[1]+data_sizes[2]]
d4 = harmony_corrected[data_sizes[0]+data_sizes[1]+data_sizes[2]:]

print(d1.shape, d2.shape, d3.shape, d4.shape)

#save
np.save("../GSE240429/data/filtered_expression_matrices/1/harmony_hvg_matrix.npy", d1.T)
np.save("../GSE240429/data/filtered_expression_matrices/2/harmony_hvg_matrix.npy", d2.T)
np.save("../GSE240429/data/filtered_expression_matrices/3/harmony_hvg_matrix.npy", d3.T)
np.save("../GSE240429/data/filtered_expression_matrices/4/harmony_hvg_matrix.npy", d4.T)  #saving gene x cell to be consistent with hvg_matrix.npy





################################                          
###           HEG            ###   
################################


import numpy as np
import pandas as pd
import scanpy as sc
import scipy.io as sio

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.io as sio
import harmonypy as hm

def heg_selection_and_pooling(exp_paths, n_top_genes=3200):
    # input n expression matrices paths, output n expression matrices with only the union of the HEGs
    all_genes = None
    gene_set = None
    adata_list = []
    heg_bools = []
    
    # Store gene names as strings, not as DataFrame index
    all_gene_names = []

    # read adata and find hegs
    for d in exp_paths:
        adata = sio.mmread(d)
        adata = adata.toarray()
        print(adata.shape)
        adata = sc.AnnData(X=adata.T, dtype=adata.dtype)

        # Remove genes (columns) that are all zeros
        non_zero_genes = (adata.X.sum(axis=0) != 0)
        adata = adata[:, non_zero_genes]
        
        # Remove cells (rows) that are all zeros
        non_zero_cells = (adata.X.sum(axis=1) != 0)
        adata = adata[non_zero_cells, :]

        if all_genes is None:
            all_genes = set(adata.var_names)
        else:
            # Include only genes that are present in all datasets
            all_genes &= set(adata.var_names)

        adata_list.append(adata)
    
    # Convert all_genes back to list and sort
    all_genes = sorted(list(all_genes))

    # Now filter each adata to have the same set of genes and collect gene names
    filtered_adata_list = []
    for adata in adata_list:
        adata = adata[:, all_genes]
        all_gene_names.append(adata.var_names)
        filtered_adata_list.append(adata)
    adata_list = filtered_adata_list

    # Now align all genes among all datasets
    for adata in adata_list:
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        
        # Calculate average expression for each gene
        avg_exp = adata.X.mean(axis=0)
        
        # Get the indices of the top `n_top_genes` highly expressed genes
        heg_indices = np.argsort(avg_exp)[::-1][:n_top_genes]
        
        # Create a boolean mask for these top `n_top_genes` highly expressed genes
        heg_mask = np.zeros(len(all_genes), dtype=bool)
        heg_mask[heg_indices] = True
        
        # Save hegs
        heg_bools.append(heg_mask)

    # find union of hegs
    heg_union = heg_bools[0]
    for i in range(1, len(heg_bools)):
        print(sum(heg_union), sum(heg_bools[i]))
        heg_union = heg_union | heg_bools[i]
    print("Number of HEGs: ", heg_union.sum())

    # Convert heg_union to Pandas Series with appropriate indexing
    heg_union_series = pd.Series(heg_union, index=all_genes)

    # Get indices where heg_union_series is True
    true_indices = heg_union_series[heg_union_series].index

    print("Number of HEGs (after union): ", heg_union_series.sum())

    # Save the heg_union to npy file
    np.save("../GSE240429/data/filtered_expression_matrices/heg_union.npy", heg_union_series.to_numpy())

    # Filter expression matrices to include only the union of HEGs
    filtered_exp_mtxs = []
    for adata in adata_list:
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        filtered_expression = adata[:, heg_union_series].X
        
        # Remove rows that are all zeros again after HEG filtering
        non_zero_rows = (filtered_expression.sum(axis=1) != 0)
        filtered_expression = filtered_expression[non_zero_rows, :]
        
        filtered_exp_mtxs.append(filtered_expression)

    return filtered_exp_mtxs


exp_paths = ["../GSE240429/data/filtered_expression_matrices/1/matrix.mtx",
            "../GSE240429/data/filtered_expression_matrices/2/matrix.mtx",
            "../GSE240429/data/filtered_expression_matrices/3/matrix.mtx",
            "../GSE240429/data/filtered_expression_matrices/4/matrix.mtx"]
filtered_mtx = heg_selection_and_pooling(exp_paths)
for i in range(len(filtered_mtx)):
    np.save("../GSE240429/data/filtered_expression_matrices/" + str(i + 1) + "/heg_matrix.npy", filtered_mtx[i].T)


################################
#! batch correct using harmony
#! Other batch correction methods can be used here in place of harmony. Furthermore, model can be trained using the hvg matrix and achieve comparable results if the datasets used are similar enough

d = np.load("../GSE240429/data/filtered_expression_matrices/1/heg_matrix.npy")
print(d.shape)

d2 = np.load("../GSE240429/data/filtered_expression_matrices/2/heg_matrix.npy")
print(d2.shape)

d3 = np.load("../GSE240429/data/filtered_expression_matrices/3/heg_matrix.npy")
print(d3.shape)

d4 = np.load("../GSE240429/data/filtered_expression_matrices/4/heg_matrix.npy")
print(d4.shape)

d = np.concatenate((d.T, d2.T, d3.T, d4.T), axis = 0)
print(d.shape)
data_sizes = [2378, 2349, 2277, 2265]
batch_labels = np.concatenate((np.zeros(2378), np.ones(2349), np.ones(2277)*2, np.ones(2265)*3))
batch_labels = batch_labels.astype(str)
df = pd.DataFrame(batch_labels, columns=["dataset"])
print(df.shape)
# # Run the Harmony integration algorithm
harmony = hm.run_harmony(d, meta_data=df, vars_use=["dataset"])
harmony_corrected = harmony.Z_corr.T

#split back into datasets
d1 = harmony_corrected[:data_sizes[0]]
d2 = harmony_corrected[data_sizes[0]:data_sizes[0]+data_sizes[1]]
d3 = harmony_corrected[data_sizes[0]+data_sizes[1]:data_sizes[0]+data_sizes[1]+data_sizes[2]]
d4 = harmony_corrected[data_sizes[0]+data_sizes[1]+data_sizes[2]:]

print(d1.shape, d2.shape, d3.shape, d4.shape)

#save
np.save("../GSE240429/data/filtered_expression_matrices/1/harmony_heg_matrix.npy", d1.T)
np.save("../GSE240429/data/filtered_expression_matrices/2/harmony_heg_matrix.npy", d2.T)
np.save("../GSE240429/data/filtered_expression_matrices/3/harmony_heg_matrix.npy", d3.T)
np.save("../GSE240429/data/filtered_expression_matrices/4/harmony_heg_matrix.npy", d4.T)  #saving gene x cell to be consistent with hvg_matrix.npy

