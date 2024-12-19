"""
The code for this work was developed based on the BLEEP model and we are grateful for their contribution.
Ref：
https://github.com/bowang-lab/BLEEP
https://proceedings.neurips.cc/paper_files/paper/2023/file/df656d6ed77b565e8dcdfbf568aead0a-Paper-Conference.pdf
"""
import cv2
import pandas as pd
import torch
import numpy as np
import torchvision.transforms.functional as TF
import random
from PIL import Image
from torch.utils.data import DataLoader

class HECLIPDataset(torch.utils.data.Dataset):
    def __init__(self, image_path, spatial_pos_path, barcode_path, reduced_mtx_path):
        # image_path is the path of an entire slice of visium h&e stained image (~2.5GB)

        # spatial_pos_csv
        # barcode name
        # detected tissue boolean
        # x spot index
        # y spot index
        # x spot position (px)
        # y spot position (px)

        # expression_mtx
        # feature x spot (alphabetical barcode order)

        # barcode_tsv
        # spot barcodes - alphabetical order

        self.whole_image = cv2.imread(image_path)
        self.spatial_pos_csv = pd.read_csv(spatial_pos_path, sep=",", header=None)
        # self.expression_mtx = csr_matrix(sio.mmread(expression_mtx_path)).toarray()
        self.barcode_tsv = pd.read_csv(barcode_path, sep="\t", header=None)
        self.reduced_matrix = np.load(reduced_mtx_path).T  # cell x features

        print("Finished loading all files")

    def transform(self, image):
        image = Image.fromarray(image)
        # Random flipping and rotations
        if random.random() > 0.5:
            image = TF.hflip(image)
        if random.random() > 0.5:
            image = TF.vflip(image)
        angle = random.choice([180, 90, 0, -90])
        image = TF.rotate(image, angle)
        return np.asarray(image)

    def __getitem__(self, idx):
        item = {}
        barcode = self.barcode_tsv.values[idx, 0]
        v1 = self.spatial_pos_csv.loc[self.spatial_pos_csv[0] == barcode, 4].values[0]
        v2 = self.spatial_pos_csv.loc[self.spatial_pos_csv[0] == barcode, 5].values[0]
        image = self.whole_image[(v1 - 128):(v1 + 128), (v2 - 128):(v2 + 128)]
        image = self.transform(image)

        item['image'] = torch.tensor(image).permute(2, 0, 1).float()  # color channel first, then XY
        item['reduced_expression'] = torch.tensor(self.reduced_matrix[idx, :]).float()  # cell x features (3467)


        return item

    def __len__(self):
        return len(self.barcode_tsv)
    




def build_loaders(args, mode):
    print("Building loaders")
    if args.type=='heg':
        f='harmony_heg_matrix.npy'
    elif args.type=='hvg':
        f='harmony_hvg_matrix.npy'

    if args.dataset=='GSE240429':
        dataset1 = HECLIPDataset(image_path = "../GSE240429/image/GEX_C73_A1_Merged.tiff",
                   spatial_pos_path = "../GSE240429/data/tissue_pos_matrices/tissue_positions_list_1.csv",
                   reduced_mtx_path = "../GSE240429/data/filtered_expression_matrices/1/"+f,
                   barcode_path = "../GSE240429/data/filtered_expression_matrices/1/barcodes.tsv")
        dataset2 = HECLIPDataset(image_path = "../GSE240429/image/GEX_C73_B1_Merged.tiff",
                    spatial_pos_path = "../GSE240429/data/tissue_pos_matrices/tissue_positions_list_2.csv",
                    reduced_mtx_path = "../GSE240429/data/filtered_expression_matrices/2/"+f,
                    barcode_path = "../GSE240429/data/filtered_expression_matrices/2/barcodes.tsv")
        dataset4 = HECLIPDataset(image_path = "../GSE240429/image/GEX_C73_D1_Merged.tiff",
                    spatial_pos_path = "../GSE240429/data/tissue_pos_matrices/tissue_positions_list_4.csv",
                    reduced_mtx_path = "../GSE240429/data/filtered_expression_matrices/4/"+f,
                    barcode_path = "../GSE240429/data/filtered_expression_matrices/4/barcodes.tsv")
        # repeat for data augmentation
        dataset5 = HECLIPDataset(image_path = "../GSE240429/image/GEX_C73_A1_Merged.tiff",
                   spatial_pos_path = "../GSE240429/data/tissue_pos_matrices/tissue_positions_list_1.csv",
                   reduced_mtx_path = "../GSE240429/data/filtered_expression_matrices/1/"+f,
                   barcode_path = "../GSE240429/data/filtered_expression_matrices/1/barcodes.tsv")
        dataset6 = HECLIPDataset(image_path = "../GSE240429/image/GEX_C73_B1_Merged.tiff",
                    spatial_pos_path = "../GSE240429/data/tissue_pos_matrices/tissue_positions_list_2.csv",
                    reduced_mtx_path = "../GSE240429/data/filtered_expression_matrices/2/"+f,
                    barcode_path = "../GSE240429/data/filtered_expression_matrices/2/barcodes.tsv")
        dataset7 = HECLIPDataset(image_path = "../GSE240429/image/GEX_C73_D1_Merged.tiff",
                    spatial_pos_path = "../GSE240429/data/tissue_pos_matrices/tissue_positions_list_4.csv",
                    reduced_mtx_path = "../GSE240429/data/filtered_expression_matrices/4/"+f,
                    barcode_path = "../GSE240429/data/filtered_expression_matrices/4/barcodes.tsv")
        if mode=='eval':
            # 3 for evaluation
            dataset3 = HECLIPDataset(image_path = "../GSE240429/image/GEX_C73_C1_Merged.tiff",
                        spatial_pos_path = "../GSE240429/data/tissue_pos_matrices/tissue_positions_list_3.csv",
                        reduced_mtx_path = "../GSE240429/data/filtered_expression_matrices/3/"+f,
                        barcode_path = "../GSE240429/data/filtered_expression_matrices/3/barcodes.tsv")
            dataset = torch.utils.data.ConcatDataset([dataset1, dataset2, dataset3, dataset4])
        else:
            dataset = torch.utils.data.ConcatDataset([dataset1, dataset2, dataset4, dataset5, dataset6, dataset7])


    elif args.dataset=='GSE245620':
        dataset1 = HECLIPDataset(image_path = "../GSE245620/image/GSM7845914_GEX_PSC011-4_A1_Merged.tif",
                   spatial_pos_path = "../GSE245620/data/tissue_pos_matrices/GSM7845914_PSC011_A1_VISIUM_tissue_positions_list.csv",
                   reduced_mtx_path = "../GSE245620/data/filtered_expression_matrices/1/"+f,
                   barcode_path = "../GSE245620/data/filtered_expression_matrices/1/GSM7845914_PSC011_A1_VISIUM_barcodes.tsv")
        dataset2 = HECLIPDataset(image_path = "../GSE245620/image/GSM7845915_GEX_PSC011-4_B1_Merged.tif",
                    spatial_pos_path = "../GSE245620/data/tissue_pos_matrices/GSM7845915_PSC011_B1_VISIUM_tissue_positions_list.csv",
                    reduced_mtx_path = "../GSE245620/data/filtered_expression_matrices/2/"+f,
                    barcode_path = "../GSE245620/data/filtered_expression_matrices/2/GSM7845915_PSC011_B1_VISIUM_barcodes.tsv")
        dataset4 = HECLIPDataset(image_path = "../GSE245620/image/GSM7845917_GEX_PSC011-4_D1_Merged.tif",
                    spatial_pos_path = "../GSE245620/data/tissue_pos_matrices/GSM7845917_PSC011_D1_VISIUM_tissue_positions_list.csv",
                    reduced_mtx_path = "../GSE245620/data/filtered_expression_matrices/4/"+f,
                    barcode_path = "../GSE245620/data/filtered_expression_matrices/4/GSM7845917_PSC011_D1_VISIUM_barcodes.tsv")
        # repeat for data augmentation
        dataset5 = HECLIPDataset(image_path = "../GSE245620/image/GSM7845914_GEX_PSC011-4_A1_Merged.tif",
                   spatial_pos_path = "../GSE245620/data/tissue_pos_matrices/GSM7845914_PSC011_A1_VISIUM_tissue_positions_list.csv",
                   reduced_mtx_path = "../GSE245620/data/filtered_expression_matrices/1/"+f,
                   barcode_path = "../GSE245620/data/filtered_expression_matrices/1/GSM7845914_PSC011_A1_VISIUM_barcodes.tsv")
        dataset6 = HECLIPDataset(image_path = "../GSE245620/image/GSM7845915_GEX_PSC011-4_B1_Merged.tif",
                    spatial_pos_path = "../GSE245620/data/tissue_pos_matrices/GSM7845915_PSC011_B1_VISIUM_tissue_positions_list.csv",
                    reduced_mtx_path = "../GSE245620/data/filtered_expression_matrices/2/"+f,
                    barcode_path = "../GSE245620/data/filtered_expression_matrices/2/GSM7845915_PSC011_B1_VISIUM_barcodes.tsv")
        dataset7 = HECLIPDataset(image_path = "../GSE245620/image/GSM7845917_GEX_PSC011-4_D1_Merged.tif",
                    spatial_pos_path = "../GSE245620/data/tissue_pos_matrices/GSM7845917_PSC011_D1_VISIUM_tissue_positions_list.csv",
                    reduced_mtx_path = "../GSE245620/data/filtered_expression_matrices/4/"+f,
                    barcode_path = "../GSE245620/data/filtered_expression_matrices/4/GSM7845917_PSC011_D1_VISIUM_barcodes.tsv")
        
        if mode=='eval':
            # 3 for evaluation
            dataset3 = HECLIPDataset(image_path = "../GSE245620/image/GSM7845916_GEX_PSC011-4_C1_Merged.tif",
                    spatial_pos_path = "../GSE245620/data/tissue_pos_matrices/GSM7845916_PSC011_C1_VISIUM_tissue_positions_list.csv",
                    reduced_mtx_path = "../GSE245620/data/filtered_expression_matrices/3/"+f,
                    barcode_path = "../GSE245620/data/filtered_expression_matrices/3/GSM7845916_PSC011_C1_VISIUM_barcodes.tsv")
            dataset = torch.utils.data.ConcatDataset([dataset1, dataset2, dataset3, dataset4])
        else:
            dataset = torch.utils.data.ConcatDataset([dataset1, dataset2, dataset4, dataset5, dataset6, dataset7])
        

    elif args.dataset=='spatialLIBD_1':
        dataset1 = HECLIPDataset(image_path = "../GSE/spatialLIBD_1/image/151507_full_image.tif",
                   spatial_pos_path = "../GSE/spatialLIBD_1/data/tissue_pos_matrices/tissue_positions_list_1.txt",
                   reduced_mtx_path = "../GSE/spatialLIBD_1/data/filtered_expression_matrices/1/"+f,
                   barcode_path = "../GSE/spatialLIBD_1/data/filtered_expression_matrices/1/barcodes.tsv")
        dataset2 = HECLIPDataset(image_path = "../GSE/spatialLIBD_1/image/151508_full_image.tif",
                    spatial_pos_path = "../GSE/spatialLIBD_1/data/tissue_pos_matrices/tissue_positions_list_2.txt",
                    reduced_mtx_path = "../GSE/spatialLIBD_1/data/filtered_expression_matrices/2/"+f,
                    barcode_path = "../GSE/spatialLIBD_1/data/filtered_expression_matrices/2/barcodes.tsv")
        # 3 for evaluation
        dataset4 = HECLIPDataset(image_path = "../GSE/spatialLIBD_1/image/151510_full_image.tif",
                    spatial_pos_path = "../GSE/spatialLIBD_1/data/tissue_pos_matrices/tissue_positions_list_4.txt",
                    reduced_mtx_path = "../GSE/spatialLIBD_1/data/filtered_expression_matrices/4/"+f,
                    barcode_path = "../GSE/spatialLIBD_1/data/filtered_expression_matrices/4/barcodes.tsv")
        # repeat for data augmentation
        dataset5 = HECLIPDataset(image_path = "../GSE/spatialLIBD_1/image/151507_full_image.tif",
                   spatial_pos_path = "../GSE/spatialLIBD_1/data/tissue_pos_matrices/tissue_positions_list_1.txt",
                   reduced_mtx_path = "../GSE/spatialLIBD_1/data/filtered_expression_matrices/1/"+f,
                   barcode_path = "../GSE/spatialLIBD_1/data/filtered_expression_matrices/1/barcodes.tsv")
        dataset6 = HECLIPDataset(image_path = "../GSE/spatialLIBD_1/image/151508_full_image.tif",
                    spatial_pos_path = "../GSE/spatialLIBD_1/data/tissue_pos_matrices/tissue_positions_list_2.txt",
                    reduced_mtx_path = "../GSE/spatialLIBD_1/data/filtered_expression_matrices/2/"+f,
                    barcode_path = "../GSE/spatialLIBD_1/data/filtered_expression_matrices/2/barcodes.tsv")
        dataset7 = HECLIPDataset(image_path = "../GSE/spatialLIBD_1/image/151510_full_image.tif",
                    spatial_pos_path = "../GSE/spatialLIBD_1/data/tissue_pos_matrices/tissue_positions_list_4.txt",
                    reduced_mtx_path = "../GSE/spatialLIBD_1/data/filtered_expression_matrices/4/"+f,
                    barcode_path = "../GSE/spatialLIBD_1/data/filtered_expression_matrices/4/barcodes.tsv")
        if mode=='eval':
            # 3 for evaluation
            dataset3 = HECLIPDataset(image_path = "../GSE/spatialLIBD_1/image/151509_full_image.tif",
                    spatial_pos_path = "../GSE/spatialLIBD_1/data/tissue_pos_matrices/tissue_positions_list_3.txt",
                    reduced_mtx_path = "../GSE/spatialLIBD_1/data/filtered_expression_matrices/3/"+f,
                    barcode_path = "../GSE/spatialLIBD_1/data/filtered_expression_matrices/3/barcodes.tsv")
            dataset = torch.utils.data.ConcatDataset([dataset1, dataset2, dataset3, dataset4])
        else:
            dataset = torch.utils.data.ConcatDataset([dataset1, dataset2, dataset4, dataset5, dataset6, dataset7])


    elif args.dataset=='spatialLIBD_2':
        dataset1 = HECLIPDataset(image_path = "../GSE/spatialLIBD_2/image/151669_full_image.tif",
                   spatial_pos_path = "../GSE/spatialLIBD_2/data/tissue_pos_matrices/tissue_positions_list_1.txt",
                   reduced_mtx_path = "../GSE/spatialLIBD_2/data/filtered_expression_matrices/1/"+f,
                   barcode_path = "../GSE/spatialLIBD_2/data/filtered_expression_matrices/1/barcodes.tsv")
        dataset2 = HECLIPDataset(image_path = "../GSE/spatialLIBD_2/image/151670_full_image.tif",
                    spatial_pos_path = "../GSE/spatialLIBD_2/data/tissue_pos_matrices/tissue_positions_list_2.txt",
                    reduced_mtx_path = "../GSE/spatialLIBD_2/data/filtered_expression_matrices/2/"+f,
                    barcode_path = "../GSE/spatialLIBD_2/data/filtered_expression_matrices/2/barcodes.tsv")
        dataset4 = HECLIPDataset(image_path = "../GSE/spatialLIBD_2/image/151672_full_image.tif",
                    spatial_pos_path = "../GSE/spatialLIBD_2/data/tissue_pos_matrices/tissue_positions_list_4.txt",
                    reduced_mtx_path = "../GSE/spatialLIBD_2/data/filtered_expression_matrices/4/"+f,
                    barcode_path = "../GSE/spatialLIBD_2/data/filtered_expression_matrices/4/barcodes.tsv")
        dataset5 = HECLIPDataset(image_path = "../GSE/spatialLIBD_2/image/151673_full_image.tif",
                    spatial_pos_path = "../GSE/spatialLIBD_2/data/tissue_pos_matrices/tissue_positions_list_5.txt",
                    reduced_mtx_path = "../GSE/spatialLIBD_2/data/filtered_expression_matrices/5/"+f,
                    barcode_path = "../GSE/spatialLIBD_2/data/filtered_expression_matrices/5/barcodes.tsv")
        dataset6 = HECLIPDataset(image_path = "../GSE/spatialLIBD_2/image/151674_full_image.tif",
                   spatial_pos_path = "../GSE/spatialLIBD_2/data/tissue_pos_matrices/tissue_positions_list_6.txt",
                   reduced_mtx_path = "../GSE/spatialLIBD_2/data/filtered_expression_matrices/6/"+f,
                   barcode_path = "../GSE/spatialLIBD_2/data/filtered_expression_matrices/6/barcodes.tsv")
        dataset7 = HECLIPDataset(image_path = "../GSE/spatialLIBD_2/image/151675_full_image.tif",
                    spatial_pos_path = "../GSE/spatialLIBD_2/data/tissue_pos_matrices/tissue_positions_list_7.txt",
                    reduced_mtx_path = "../GSE/spatialLIBD_2/data/filtered_expression_matrices/7/"+f,
                    barcode_path = "../GSE/spatialLIBD_2/data/filtered_expression_matrices/7/barcodes.tsv")
        dataset8 = HECLIPDataset(image_path = "../GSE/spatialLIBD_2/image/151676_full_image.tif",
                    spatial_pos_path = "../GSE/spatialLIBD_2/data/tissue_pos_matrices/tissue_positions_list_8.txt",
                    reduced_mtx_path = "../GSE/spatialLIBD_2/data/filtered_expression_matrices/8/"+f,
                    barcode_path = "../GSE/spatialLIBD_2/data/filtered_expression_matrices/8/barcodes.tsv")
        # repeat for data augmentation
        dataset9 = HECLIPDataset(image_path = "../GSE/spatialLIBD_2/image/151669_full_image.tif",
                   spatial_pos_path = "../GSE/spatialLIBD_2/data/tissue_pos_matrices/tissue_positions_list_1.txt",
                   reduced_mtx_path = "../GSE/spatialLIBD_2/data/filtered_expression_matrices/1/"+f,
                   barcode_path = "../GSE/spatialLIBD_2/data/filtered_expression_matrices/1/barcodes.tsv")
        dataset10 = HECLIPDataset(image_path = "../GSE/spatialLIBD_2/image/151670_full_image.tif",
                    spatial_pos_path = "../GSE/spatialLIBD_2/data/tissue_pos_matrices/tissue_positions_list_2.txt",
                    reduced_mtx_path = "../GSE/spatialLIBD_2/data/filtered_expression_matrices/2/"+f,
                    barcode_path = "../GSE/spatialLIBD_2/data/filtered_expression_matrices/2/barcodes.tsv")
        dataset11 = HECLIPDataset(image_path = "../GSE/spatialLIBD_2/image/151672_full_image.tif",
                    spatial_pos_path = "../GSE/spatialLIBD_2/data/tissue_pos_matrices/tissue_positions_list_4.txt",
                    reduced_mtx_path = "../GSE/spatialLIBD_2/data/filtered_expression_matrices/4/"+f,
                    barcode_path = "../GSE/spatialLIBD_2/data/filtered_expression_matrices/4/barcodes.tsv")
        dataset12 = HECLIPDataset(image_path = "../GSE/spatialLIBD_2/image/151673_full_image.tif",
                    spatial_pos_path = "../GSE/spatialLIBD_2/data/tissue_pos_matrices/tissue_positions_list_5.txt",
                    reduced_mtx_path = "../GSE/spatialLIBD_2/data/filtered_expression_matrices/5/"+f,
                    barcode_path = "../GSE/spatialLIBD_2/data/filtered_expression_matrices/5/barcodes.tsv")
        dataset13 = HECLIPDataset(image_path = "../GSE/spatialLIBD_2/image/151674_full_image.tif",
                   spatial_pos_path = "../GSE/spatialLIBD_2/data/tissue_pos_matrices/tissue_positions_list_6.txt",
                   reduced_mtx_path = "../GSE/spatialLIBD_2/data/filtered_expression_matrices/6/"+f,
                   barcode_path = "../GSE/spatialLIBD_2/data/filtered_expression_matrices/6/barcodes.tsv")
        dataset14 = HECLIPDataset(image_path = "../GSE/spatialLIBD_2/image/151675_full_image.tif",
                    spatial_pos_path = "../GSE/spatialLIBD_2/data/tissue_pos_matrices/tissue_positions_list_7.txt",
                    reduced_mtx_path = "../GSE/spatialLIBD_2/data/filtered_expression_matrices/7/"+f,
                    barcode_path = "../GSE/spatialLIBD_2/data/filtered_expression_matrices/7/barcodes.tsv")
        dataset15 = HECLIPDataset(image_path = "../GSE/spatialLIBD_2/image/151676_full_image.tif",
                    spatial_pos_path = "../GSE/spatialLIBD_2/data/tissue_pos_matrices/tissue_positions_list_8.txt",
                    reduced_mtx_path = "../GSE/spatialLIBD_2/data/filtered_expression_matrices/8/"+f,
                    barcode_path = "../GSE/spatialLIBD_2/data/filtered_expression_matrices/8/barcodes.tsv")
        if mode=='eval':
            # 3 for evaluation
            dataset3 = HECLIPDataset(image_path = "../GSE/spatialLIBD_2/image/151671_full_image.tif",
                        spatial_pos_path = "../GSE/spatialLIBD_2/data/tissue_pos_matrices/tissue_positions_list_3.txt",
                        reduced_mtx_path = "../GSE/spatialLIBD_2/data/filtered_expression_matrices/3/"+f,
                        barcode_path = "../GSE/spatialLIBD_2/data/filtered_expression_matrices/3/barcodes.tsv")
            dataset = torch.utils.data.ConcatDataset([dataset1, dataset2, dataset3, dataset4, dataset5, dataset6, dataset7, dataset8])
        else:
            dataset = torch.utils.data.ConcatDataset([dataset1, dataset2, dataset4, dataset5, dataset6, dataset7, dataset8, 
                                                    dataset9, dataset10, dataset11, dataset12, dataset13, dataset14, dataset15])
    
    
    
    

    if mode =='eval':
        test_loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)
        print("Finished building loaders")
        return test_loader
    elif mode=='train':
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size],
                                                                    generator=torch.Generator().manual_seed(42))
        print('train_size:', len(train_dataset), 'test_size:', len(test_dataset))
        print("train/test split completed")
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=True)

        print("Finished building loaders")
        return train_loader, test_loader



def load4match(args,ebd_save_path):
    if args.type=='heg':
        f='harmony_heg_matrix.npy'
    elif args.type=='hvg':
        f='harmony_hvg_matrix.npy'
    if args.dataset=='GSE240429':
        spot_expression1 = np.load("../GSE240429/data/filtered_expression_matrices/1/"+f)
        spot_expression2 = np.load("../GSE240429/data/filtered_expression_matrices/2/"+f)
        spot_expression3 = np.load("../GSE240429/data/filtered_expression_matrices/3/"+f)
        spot_expression4 = np.load("../GSE240429/data/filtered_expression_matrices/4/"+f)

    if args.dataset=='GSE245620':
        spot_expression1 = np.load("../GSE245620/data/filtered_expression_matrices/1/"+f)
        spot_expression2 = np.load("../GSE245620/data/filtered_expression_matrices/2/"+f)
        spot_expression3 = np.load("../GSE245620/data/filtered_expression_matrices/3/"+f)
        spot_expression4 = np.load("../GSE245620/data/filtered_expression_matrices/4/"+f)

    if args.dataset=='spatialLIBD_1':
        spot_expression1 = np.load("../GSE/spatialLIBD_1/data/filtered_expression_matrices/1/"+f)
        spot_expression2 = np.load("../GSE/spatialLIBD_1/data/filtered_expression_matrices/2/"+f)
        spot_expression3 = np.load("../GSE/spatialLIBD_1/data/filtered_expression_matrices/3/"+f)
        spot_expression4 = np.load("../GSE/spatialLIBD_1/data/filtered_expression_matrices/4/"+f)

    if args.dataset=='spatialLIBD_2':
        spot_expression1 = np.load("../GSE/spatialLIBD_2/data/filtered_expression_matrices/1/"+f)
        spot_expression2 = np.load("../GSE/spatialLIBD_2/data/filtered_expression_matrices/2/"+f)
        spot_expression3 = np.load("../GSE/spatialLIBD_2/data/filtered_expression_matrices/3/"+f)
        spot_expression4 = np.load("../GSE/spatialLIBD_2/data/filtered_expression_matrices/4/"+f)
        spot_expression5 = np.load("../GSE/spatialLIBD_2/data/filtered_expression_matrices/5/"+f)
        spot_expression6 = np.load("../GSE/spatialLIBD_2/data/filtered_expression_matrices/6/"+f)
        spot_expression7 = np.load("../GSE/spatialLIBD_2/data/filtered_expression_matrices/7/"+f)
        spot_expression8 = np.load("../GSE/spatialLIBD_2/data/filtered_expression_matrices/8/"+f)
        image_embeddings1 = np.load(ebd_save_path +"/img_embeddings_1.npy")
        image_embeddings2 = np.load(ebd_save_path +"/img_embeddings_2.npy")
        image_embeddings3 = np.load(ebd_save_path +"/img_embeddings_3.npy")
        image_embeddings4 = np.load(ebd_save_path +"/img_embeddings_4.npy")
        image_embeddings5 = np.load(ebd_save_path +"/img_embeddings_5.npy")
        image_embeddings6 = np.load(ebd_save_path +"/img_embeddings_6.npy")
        image_embeddings7 = np.load(ebd_save_path +"/img_embeddings_7.npy")
        image_embeddings8 = np.load(ebd_save_path +"/img_embeddings_8.npy")
        #query
        image_query = image_embeddings3
        print("image query shape: ", image_query.shape)
        expression_gt = spot_expression3
        print("expression_gt shape: ", expression_gt.shape)
        #reference
        image_reference = np.concatenate([image_embeddings1, image_embeddings2, image_embeddings4, image_embeddings5, 
                                    image_embeddings6, image_embeddings7, image_embeddings8], axis = 1)    # for retrival
        print("image reference shape: ", image_reference.shape)
        expression_key = np.concatenate([spot_expression1, spot_expression2, spot_expression4, spot_expression5, 
                                         spot_expression6, spot_expression7, spot_expression8], axis = 1)  # for exp generation
        print("expression_key shape: ", expression_key.shape)

        return image_query, expression_gt, image_reference, expression_key
    

    image_embeddings1 = np.load(ebd_save_path +"/img_embeddings_1.npy")
    image_embeddings2 = np.load(ebd_save_path +"/img_embeddings_2.npy")
    image_embeddings3 = np.load(ebd_save_path +"/img_embeddings_3.npy")
    image_embeddings4 = np.load(ebd_save_path +"/img_embeddings_4.npy")
    #query
    image_query = image_embeddings3
    print("image query shape: ", image_query.shape)
    expression_gt = spot_expression3
    print("expression_gt shape: ", expression_gt.shape)
    #reference
    image_reference = np.concatenate([image_embeddings1, image_embeddings2, image_embeddings4], axis = 1)    # for retrival
    print("image_reference shape: ", image_reference.shape)
    expression_key = np.concatenate([spot_expression1, spot_expression2, spot_expression4], axis = 1)  # for exp generation
    print("expression_key shape: ", expression_key.shape)

    return image_query, expression_gt, image_reference, expression_key

