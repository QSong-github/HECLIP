"""
The code for this work was developed based on the BLEEP.
Ref：
https://github.com/bowang-lab/BLEEP
https://proceedings.neurips.cc/paper_files/paper/2023/file/df656d6ed77b565e8dcdfbf568aead0a-Paper-Conference.pdf
"""

import os
from tqdm import tqdm
import torch
import torch.utils.data.distributed
from dt_load import build_loaders
from utils import AvgMeter
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='HECLIP')

parser.add_argument('--save_path', type=str, default='./save/', help='')
parser.add_argument('--batch_size', type=int, default=128, help='')
parser.add_argument('--max_epochs', type=int, default=15, help='')
parser.add_argument('--lr', type=float, default=0.001, help='')
parser.add_argument('--wd', type=float, default=0.001, help='')
parser.add_argument('--dataset', type=str, default='spatialLIBD_2', help='[GSE240429,GSE245620,spatialLIBD_1,spatialLIBD_2]')
parser.add_argument('--type', type=str, default='hvg', help='[hvg,heg]')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='')


def init(args):
    config={
            'projection_dim':3467,
            'temperature':1.0,
            'embedding_dim':2048,
            'model_name':'resnet50',
            'dropout':0.1
        }
    if args.dataset=='GSE240429' and args.type=='hvg':
        config['projection_dim']=3467
    if args.dataset=='GSE240429' and args.type=='heg':
        config['projection_dim']=3511

    if args.dataset=='GSE245620' and args.type=='hvg':
        config['projection_dim']=3508
    if args.dataset=='GSE245620' and args.type=='heg':
        config['projection_dim']=3403

    if args.dataset=='spatialLIBD_1' and args.type=='hvg':
        config['projection_dim']=3376
    if args.dataset=='spatialLIBD_1' and args.type=='heg':
        config['projection_dim']=3468

    if args.dataset=='spatialLIBD_2' and args.type=='hvg':
        config['projection_dim']=3405
    if args.dataset=='spatialLIBD_2' and args.type=='heg':
        config['projection_dim']=3615

    return config






def train_epoch(model, train_loader, optimizer):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        batch = {k: v.cuda() for k, v in batch.items() if k == "image" or k == "reduced_expression"}
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg)

    return loss_meter


def test_epoch(model, test_loader):
    loss_meter = AvgMeter()

    tqdm_object = tqdm(test_loader, total=len(test_loader))
    for batch in tqdm_object:
        batch = {k: v.cuda() for k, v in batch.items() if k == "image" or k == "reduced_expression"}
        loss = model(batch)

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter


def main():
    print("Starting...")
    
    args = parser.parse_args()
    print("dataset:",args.dataset)
    print("type:",args.type)
    save_path = args.save_path + args.dataset
    config = init(args)
    train_loader, test_loader = build_loaders(args,mode='train')
    if args.type=='hvg':
        from models_hvg import HECLIPModel
    elif args.type=='heg':
        from models_heg import HECLIPModel
    model = HECLIPModel(config).to(args.device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.wd
    )
    train_losses = []
    test_losses = []
    best_loss = float('inf')
    best_epoch = -1

    for epoch in range(args.max_epochs):
        print(f"Epoch: {epoch + 1}")
        
        # Train
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer)
        train_losses.append(train_loss.avg)  # Save train loss
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            test_loss = test_epoch(model, test_loader)
        test_losses.append(test_loss.avg)  # Save test loss
        
        # Check for the best model
        if test_loss.avg < best_loss:
            if not os.path.exists(str(save_path)):
                os.mkdir(str(save_path))
            best_loss = test_loss.avg
            best_epoch = epoch

            torch.save(model.state_dict(), os.path.join(str(save_path), f"{args.type}_best.pt"))
            print("Saved Best Model! Loss: {}".format(best_loss))

    # Save the train and test losses to a CSV file after all epochs are done
    results_df = pd.DataFrame({
        'epoch': list(range(1, args.max_epochs + 1)),
        'train_loss': train_losses,
        'test_loss': test_losses
    })
    results_df.to_csv('train_test_losses.csv', index=False)

    print("Done!, final loss: {}".format(best_loss))
    print("Best epoch: {}".format(best_epoch))


if __name__ == "__main__":
    main()
