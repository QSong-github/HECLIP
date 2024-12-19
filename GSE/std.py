import h5py
import numpy as np
from scipy.io import mmwrite
from scipy.sparse import csc_matrix




def std_exp():
    # from h5 to mtx
    # 文件路径
    file_path = '/blue/qsong1/wang.qing/bleep/BLEEP-main/GSE/spatialLIBD_1/data/filtered_expression_matrices/3/151509_filtered_feature_bc_matrix.h5'   # 替换为你的文件路径
    mtx_file_path = '/blue/qsong1/wang.qing/bleep/BLEEP-main/GSE/spatialLIBD_1/data/filtered_expression_matrices/3/matrix.mtx'  # 输出Matrix Market文件路径
    barcodes_tsv_path = '/blue/qsong1/wang.qing/bleep/BLEEP-main/GSE/spatialLIBD_1/data/filtered_expression_matrices/3/barcodes.tsv'  # 输出barcodes TSV文件路径
    features_tsv_path = '/blue/qsong1/wang.qing/bleep/BLEEP-main/GSE/spatialLIBD_2/data/filtered_expression_matrices/3/features.tsv'  # 输出features TSV文件路径

    with h5py.File(file_path, 'r') as h5_file:
        # 获取名为 'matrix' 的组
        group = h5_file['matrix']

        # 读取数据集
        data = group['data'][:]
        indices = group['indices'][:]
        indptr = group['indptr'][:]
        shape = tuple(group['shape'][:])

        # 创建CSC稀疏矩阵
        csc_matrix_data = csc_matrix((data, indices, indptr), shape=shape)

        # 保存为Matrix Market格式
        mmwrite(mtx_file_path, csc_matrix_data)
        print(f"Matrix has been written to {mtx_file_path}")

        # 读取并保存barcodes
        barcodes = group['barcodes'][:]
        barcodes = [barcode.decode('utf-8') for barcode in barcodes]
        with open(barcodes_tsv_path, 'w') as f:
            for barcode in barcodes:
                f.write("%s\n" % barcode)
        print(f"Barcodes have been written to {barcodes_tsv_path}")

        # 处理features数据
        features_group = group['features']
        gene_ids = features_group['id'][:]
        gene_names = features_group['name'][:]

        # 创建features DataFrame
        with open(features_tsv_path, 'w') as f:
            for gene_id, gene_name in zip(gene_ids, gene_names):
                f.write("%s\t%s\tGene Expression\n" % (gene_id.decode('utf-8'), gene_name.decode('utf-8')))
        print(f"Features have been written to {features_tsv_path}")

std_exp()