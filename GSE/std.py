import h5py
import numpy as np
from scipy.io import mmwrite
from scipy.sparse import csc_matrix




def std_exp():
    # from h5 to mtx
    # Please change the file_path when select different dataset.
    file_path = './spatialLIBD_1/data/filtered_expression_matrices/3/151509_filtered_feature_bc_matrix.h5' # the h5 path   
    mtx_file_path = './spatialLIBD_1/data/filtered_expression_matrices/3/matrix.mtx'  # Matrix Market path
    barcodes_tsv_path = './spatialLIBD_1/data/filtered_expression_matrices/3/barcodes.tsv'  # barcodes TSV path
    features_tsv_path = './spatialLIBD_1/data/filtered_expression_matrices/3/features.tsv'  # features TSV path

    with h5py.File(file_path, 'r') as h5_file:
        # get 'matrix' 
        group = h5_file['matrix']

        data = group['data'][:]
        indices = group['indices'][:]
        indptr = group['indptr'][:]
        shape = tuple(group['shape'][:])

        # csc
        csc_matrix_data = csc_matrix((data, indices, indptr), shape=shape)

        # save
        mmwrite(mtx_file_path, csc_matrix_data)
        print(f"Matrix has been written to {mtx_file_path}")

        # save barcodes
        barcodes = group['barcodes'][:]
        barcodes = [barcode.decode('utf-8') for barcode in barcodes]
        with open(barcodes_tsv_path, 'w') as f:
            for barcode in barcodes:
                f.write("%s\n" % barcode)
        print(f"Barcodes have been written to {barcodes_tsv_path}")

        # features
        features_group = group['features']
        gene_ids = features_group['id'][:]
        gene_names = features_group['name'][:]

        # features DataFrame
        with open(features_tsv_path, 'w') as f:
            for gene_id, gene_name in zip(gene_ids, gene_names):
                f.write("%s\t%s\tGene Expression\n" % (gene_id.decode('utf-8'), gene_name.decode('utf-8')))
        print(f"Features have been written to {features_tsv_path}")

std_exp()
