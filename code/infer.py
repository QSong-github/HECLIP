import torch
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
from dt_load import build_loaders, load4match
import os
import numpy as np
from GenesMetrics import *
import argparse
import umap
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='HECLIP')
parser.add_argument('--save_path', type=str, default='./save/', help='')
parser.add_argument('--dataset', type=str, default='GSE240429', help='[GSE240429,GSE245620,spatialLIBD_1,spatialLIBD_2]')
parser.add_argument('--type', type=str, default='heg', help='[hvg,heg]')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='')




def init(args):
    config={
            'projection_dim':3467,
            'temperature':1.0,
            'embedding_dim':2048,
            'model_name':'resnet50',
            'dropout':0.1,
            'datasize':[2378, 2349, 2277, 2265]
        }
    if args.dataset=='GSE240429' and args.type=='hvg':
        config['projection_dim']=3467
        config['datasize']=[2378, 2349, 2277, 2265]
    if args.dataset=='GSE240429' and args.type=='heg':
        config['projection_dim']=3511
        config['datasize']=[2378, 2349, 2277, 2265]

    if args.dataset=='GSE245620' and args.type=='hvg':
        config['projection_dim']=3508
        config['datasize']=[4992, 4992, 4992, 4991]
    if args.dataset=='GSE245620' and args.type=='heg':
        config['projection_dim']=3403
        config['datasize']=[4992, 4992, 4992, 4992]

    if args.dataset=='spatialLIBD_1' and args.type=='hvg':
        config['projection_dim']=3376
        config['datasize']=[3661, 4384, 4789, 4634]
    if args.dataset=='spatialLIBD_1' and args.type=='heg':
        config['projection_dim']=3468
        config['datasize']=[3661, 4384, 4789, 4634]

    if args.dataset=='spatialLIBD_2' and args.type=='hvg':
        config['projection_dim']=3405
        config['datasize']=[3661,3498,4110,4015,3639,3673,3592,3460]
    if args.dataset=='spatialLIBD_2' and args.type=='heg':
        config['projection_dim']=3615
        config['datasize']=[3661,3498,4110,4015,3639,3673,3592,3460]

    return config


def get_image_embeddings(model_path, model, args):
    test_loader = build_loaders(args, mode='eval')

    state_dict = torch.load(model_path)
    new_state_dict = {}
    for key in state_dict.keys():
        new_key = key.replace('module.', '')  # remove the prefix 'module.'
        new_key = new_key.replace('well', 'spot')  # for compatibility with prior naming
        new_state_dict[new_key] = state_dict[key]

    model.load_state_dict(new_state_dict)
    # print(model)
    model.eval()

    print("Finished loading model")

    test_image_embeddings = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            image_features = model.image_encoder(batch["image"].cuda())
            image_embeddings = model.image_projection(image_features)
            test_image_embeddings.append(image_embeddings)

    return torch.cat(test_image_embeddings)


# def get_spot_embeddings(model_path, model,args):
#     test_loader = build_loaders(args,mode='eval')
#     # model = CLIPModel().cuda()

#     state_dict = torch.load(model_path)
#     new_state_dict = {}
#     for key in state_dict.keys():
#         new_key = key.replace('module.', '')  # remove the prefix 'module.'
#         new_key = new_key.replace('well', 'spot')  # for compatibility with prior naming
#         new_state_dict[new_key] = state_dict[key]

#     model.load_state_dict(new_state_dict)
#     model.eval()

#     print("Finished loading model")

#     spot_embeddings = []
#     with torch.no_grad():
#         for batch in tqdm(test_loader):
#             spot_embeddings.append(model.spot_projection(batch["reduced_expression"].cuda()))
#     return torch.cat(spot_embeddings)






def find_matches(spot_embeddings, query_embeddings, top_k):
    # find the closest matches
    spot_embeddings = torch.tensor(spot_embeddings)
    query_embeddings = torch.tensor(query_embeddings)
    query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
    spot_embeddings = F.normalize(spot_embeddings, p=2, dim=-1)
    dot_similarity = query_embeddings @ spot_embeddings.T
    print(dot_similarity.shape)
    _, indices = torch.topk(dot_similarity.squeeze(0), k=top_k)
    print(indices.shape)

    return indices.cpu().numpy()   #for every sample retrieve top50 from reference set

def Hit_at_k(pred, true, k):
    # - pred: ndarray，(2277, 3467)
    # - true: ndarray，(2277, 3467)
    # - k: top k index

    num_samples = pred.shape[0]
    
    # find index
    pred_top_k_indices = np.argsort(pred, axis=1)[:, -k:]
    true_top_k_indices = np.argsort(true, axis=1)[:, -k:]

    # calculate Hit@K
    correct_predictions = 0
    
    for i in range(num_samples):
        # find intersections
        pred_set = set(pred_top_k_indices[i])
        true_set = set(true_top_k_indices[i])
        
        if pred_set & true_set:  # If there is an intersection, it is considered a correct prediction.
            correct_predictions += 1

    return correct_predictions / num_samples


def exp_plot(data1, data2, args):
    name = args.dataset + "_"+args.type
    # 合并数据
    data = np.vstack((data1, data2))

    # 创建标签，data1的标签为0，data2的标签为1
    labels = np.array([0] * len(data1) + [1] * len(data2))

    # 运行UMAP降维
    # 固定随机种子
    random_state = 42
    # 创建UMAP实例
    reducer = umap.UMAP(random_state=random_state)
    embedding = reducer.fit_transform(data)

    # 绘图
    plt.figure(figsize=(10, 8))
    # 设置字体大小
    plt.rcParams.update({'font.size': 16})
    # 绘制data1的数据
    plt.scatter(embedding[labels == 0, 0], embedding[labels == 0, 1], s=10, c='#DE582B', label='True')

    # 绘制data2的数据
    plt.scatter(embedding[labels == 1, 0], embedding[labels == 1, 1], s=10, c='#1868B2', label='Pred')

    # 添加标题和图例
    plt.title('EXP UMAP projection of '+ name, fontsize=20)
    plt.legend(loc='best', fontsize=16)
    plt.savefig('./figure/EXP/'+name+'.pdf', format='pdf', bbox_inches='tight')
    # 显示图像
    # plt.show()

def ebd_plot(data1, data2, args):
    name = args.dataset + "_"+args.type
    # 合并数据
    data = np.vstack((data1, data2))

    # 创建标签，data1的标签为0，data2的标签为1
    labels = np.array([0] * len(data1) + [1] * len(data2))

    # 运行UMAP降维
    # 固定随机种子
    random_state = 42
    # 创建UMAP实例
    reducer = umap.UMAP(random_state=random_state)
    embedding = reducer.fit_transform(data)

    # 绘图
    plt.figure(figsize=(10, 8))
    # 设置字体大小
    plt.rcParams.update({'font.size': 30})
    # 绘制data1的数据
    plt.scatter(embedding[labels == 0, 0], embedding[labels == 0, 1], s=10, c='#DE582B', label='Query')

    # 绘制data2的数据
    plt.scatter(embedding[labels == 1, 0], embedding[labels == 1, 1], s=10, c='#1868B2', label='Reference')

    # 添加标题和图例
    plt.title('Embedding of '+ name, fontsize=24)
    plt.legend(loc='best', fontsize=30, framealpha=0.5)  # 增大图例字体
    plt.savefig('./figure/EBD/'+name+'.pdf', format='pdf', bbox_inches='tight')
    # 显示图像
    # plt.show()


##################################################################################################################################################

def infer():
    args = parser.parse_args()
    dataset_name=args.dataset
    config = init(args)
    if args.type=='hvg':
        from models_hvg import HECLIPModel
    elif args.type=='heg':
        from models_heg import HECLIPModel
    model_path = args.save_path + args.dataset + '/' +args.type + '_best.pt'
    print(model_path)

    ebd_save_path = './ebd/'+dataset_name + "/"+args.type
    model = HECLIPModel(config).to(args.device)

    img_embeddings_all = get_image_embeddings(model_path, model,args)
    img_embeddings_all = img_embeddings_all.cpu().numpy()

    print('img_embeddings_all.shape:',img_embeddings_all.shape)

    if not os.path.exists(ebd_save_path):
        os.makedirs(ebd_save_path)

    l = 4
    if dataset_name=='spatialLIBD_2':
        l=8
    for i in range(l):
        index_start = sum(config['datasize'][:i])
        index_end = sum(config['datasize'][:i+1])
        image_embeddings = img_embeddings_all[index_start:index_end]
        print(image_embeddings.shape)
        np.save(ebd_save_path +"/img_embeddings_" + str(i+1) + ".npy", image_embeddings.T)


    image_query, expression_gt, reference, expression_key = load4match(args,ebd_save_path)
    ebd_plot(image_query.T, reference.T, args=args)

    print("###################### reshape ######################")           
    if image_query.shape[1] != config['projection_dim']:
        image_query = image_query.T
        print("image query shape: ", image_query.shape)
    if expression_gt.shape[0] != image_query.shape[0]:
        expression_gt = expression_gt.T
        print("expression_gt shape: ", expression_gt.shape)
    if reference.shape[1] != config['projection_dim']:
        reference = reference.T
        print("reference shape: ", reference.shape)
    if expression_key.shape[0] != reference.shape[0]:
        expression_key = expression_key.T
        print("expression_key shape: ", expression_key.shape)


    print("finding matches, using average of top K expressions")
    indices = find_matches(reference, image_query, top_k=50)
    matched_spot_embeddings_pred = np.zeros((indices.shape[0], reference.shape[1]))  # top embedding
    matched_spot_expression_pred = np.zeros((indices.shape[0], expression_key.shape[1]))  # top exp
    for i in range(indices.shape[0]):
        matched_spot_embeddings_pred[i, :] = np.average(reference[indices[i, :], :], axis=0)    # average
        matched_spot_expression_pred[i, :] = np.average(expression_key[indices[i, :], :], axis=0)

    print("matched spot embeddings pred shape: ", matched_spot_embeddings_pred.shape)
    print("matched spot expression pred shape: ", matched_spot_expression_pred.shape)



    true = expression_gt
    pred = matched_spot_expression_pred
    
    exp_plot(true, pred, args)


    print('pred.shape',pred.shape)   
    print('true.shape',true.shape)   



    # genemetrics
    ACC = count(true, pred)
    res =ACC.compute_all()
    print(res)



    # Hit@K
    hit_at_k = Hit_at_k(pred, true, 5)
    print(f"Hit@{5} : {hit_at_k:.4f}")
    hit_at_k = Hit_at_k(pred, true, 4)
    print(f"Hit@{4} : {hit_at_k:.4f}")
    hit_at_k = Hit_at_k(pred, true, 3)
    print(f"Hit@{3} : {hit_at_k:.4f}")
    hit_at_k = Hit_at_k(pred, true, 2)
    print(f"Hit@{2} : {hit_at_k:.4f}")
    hit_at_k = Hit_at_k(pred, true, 1)
    print(f"Hit@{1} : {hit_at_k:.4f}")


    corr = np.zeros(pred.shape[1])
    for i in range(pred.shape[1]):
        corr[i] = np.corrcoef(pred[:, i], true[:, i], )[0, 1]
    corr = corr[~np.isnan(corr)]


    ind = np.argsort(np.sum(true, axis=0))[-1:]
    print("HEG-1: ", np.mean(corr[ind]))
    ind = np.argsort(np.sum(true, axis=0))[-3:]
    print("HEG-3: ", np.mean(corr[ind]))
    ind = np.argsort(np.sum(true, axis=0))[-5:]
    print("HEG-5: ", np.mean(corr[ind]))

    ind = np.argsort(np.var(true, axis=0))[-1:]
    print("HVG-1: ", np.mean(corr[ind]))
    ind = np.argsort(np.var(true, axis=0))[-3:]
    print("HVG-3: ", np.mean(corr[ind]))
    ind = np.argsort(np.var(true, axis=0))[-5:]
    print("HVG-5: ", np.mean(corr[ind]))


    ind = np.argsort(np.sum(true, axis=0))[-10:]
    print("HEG-10: ", np.mean(corr[ind]))
    ind = np.argsort(np.sum(true, axis=0))[-50:]
    print("HEG-50: ", np.mean(corr[ind]))
    ind = np.argsort(np.sum(true, axis=0))[-100:]
    print("HEG-100: ", np.mean(corr[ind]))

    ind = np.argsort(np.var(true, axis=0))[-10:]
    print("HVG-10: ", np.mean(corr[ind]))
    ind = np.argsort(np.var(true, axis=0))[-50:]
    print("HVG-50: ", np.mean(corr[ind]))
    ind = np.argsort(np.var(true, axis=0))[-100:]
    print("HVG-100: ", np.mean(corr[ind]))


    # for saving
    hit_at_k_values = {}
    hit_at_k_values[f'Hit@{5}'] = round(Hit_at_k(pred, true, 5), 4)
    hit_at_k_values[f'Hit@{4}'] = round(Hit_at_k(pred, true, 4), 4)
    hit_at_k_values[f'Hit@{3}'] = round(Hit_at_k(pred, true, 3), 4)
    hit_at_k_values[f'Hit@{2}'] = round(Hit_at_k(pred, true, 2), 4)
    hit_at_k_values[f'Hit@{1}'] = round(Hit_at_k(pred, true, 1), 4)


    heg_dicts = {
        "HEG-1": round(np.mean(corr[np.argsort(np.sum(true, axis=0))[-1:]]), 4),
        "HEG-3": round(np.mean(corr[np.argsort(np.sum(true, axis=0))[-3:]]), 4),
        "HEG-5": round(np.mean(corr[np.argsort(np.sum(true, axis=0))[-5:]]), 4),
        "HEG-10": round(np.mean(corr[np.argsort(np.sum(true, axis=0))[-10:]]), 4),
        "HEG-50": round(np.mean(corr[np.argsort(np.sum(true, axis=0))[-50:]]), 4),
        "HEG-100": round(np.mean(corr[np.argsort(np.sum(true, axis=0))[-100:]]), 4),
    }

    hvg_dicts = {
        "HVG-1": round(np.mean(corr[np.argsort(np.var(true, axis=0))[-1:]]), 4),
        "HVG-3": round(np.mean(corr[np.argsort(np.var(true, axis=0))[-3:]]), 4),
        "HVG-5": round(np.mean(corr[np.argsort(np.var(true, axis=0))[-5:]]), 4),
        "HVG-10": round(np.mean(corr[np.argsort(np.var(true, axis=0))[-10:]]), 4),
        "HVG-50": round(np.mean(corr[np.argsort(np.var(true, axis=0))[-50:]]), 4),
        "HVG-100": round(np.mean(corr[np.argsort(np.var(true, axis=0))[-100:]]), 4)
    }



    # Combine all dictionaries into one
    combined_dict = {**hit_at_k_values, **heg_dicts, **hvg_dicts}

    # Convert to DataFrame and save as CSV
    df = pd.DataFrame([combined_dict])
    df.to_csv('./results.csv', index=False)


    if args.dataset=='GSE240429' and args.type=='hvg':
        # marker genes
        marker_gene_list = ["HAL", "CYP3A4", "VWF", "SOX9", "KRT7", "ANXA4", "ACTA2", "DCN"]
        # same features.tsv
        gene_names = pd.read_csv("../GSE240429/data/filtered_expression_matrices/3/features.tsv", header=None,
                                sep="\t").iloc[:, 1].values
        hvg_b = np.load("../GSE240429/data/filtered_expression_matrices/hvg_union.npy")
        marker_gene_ind = np.zeros(len(marker_gene_list))
        for i in range(len(marker_gene_list)):
            marker_gene_ind[i] = np.where(gene_names[hvg_b] == marker_gene_list[i])[0]
        print("mean correlation marker genes: ", np.mean(corr[marker_gene_ind.astype(int)]))
    else:
        print('marker genes unavilable!!!')






infer()