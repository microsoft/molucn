import os
import os.path as osp
import sys
import time
import wandb

import dill
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import DataLoader
from molucn.dataset.pair import get_num_features
from molucn.gnn.model import GNN
from molucn.utils.parser_utils import overall_parser
from molucn.utils.train_utils import DEVICE, test_epoch, train_epoch
from molucn.utils.utils import set_seed

print(sys.path)


save_dir = "/cluster/work/zhang/kamara/molucn/"
data_dir = "/cluster/work/zhang/kamara/molucn/data"
log_dir = "/cluster/work/zhang/kamara/molucn/gridsearch"


def train_gnn(args, save_model=True, track_wandb=False):
    """Train the GNN model and save the logs (rmse, pcc scores) and the model"""

    # note that we define values from `wandb.config` instead 
    # of defining hard values
    #args.lr  =  wandb.config.lr
    #args.batch_size = wandb.config.batch_size
    #args.epoch = wandb.config.epoch

    set_seed(args.seed)
    train_params = (
        f"{args.conv}_{args.loss}_{args.pool}_{args.lambda1}_{args.explainer}"
    )


    # Check that data exists
    file_train = osp.join(
        args.data_path, f"{args.target}/{args.target}_seed_{args.seed}_train.pt"
    )
    file_test = osp.join(
        args.data_path, f"{args.target}/{args.target}_seed_{args.seed}_test.pt"
    )

    if not osp.exists(file_train) or not osp.exists(file_test):
        raise FileNotFoundError(
            "Data not found. Please try to - choose another protein target or - run code/pair.py with a new seed."
        )

    with open(file_train, "rb") as handle:
        trainval_dataset = dill.load(handle)

    with open(file_test, "rb") as handle:
        test_dataset = dill.load(handle)

    # Make a dataset that inherits from torch_geometric.data.Dataset

    # Ligands in testing set are NOT in training set!

    train_dataset, val_dataset = train_test_split(
        trainval_dataset, random_state=args.seed, test_size=args.val_set_size
    )
    # Ligands in validation set might also be in training set!


    test_loader = DataLoader(
        test_dataset,
        batch_size=len(test_dataset),
        shuffle=False,
        num_workers=args.num_workers,  # num_workers can go into args
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=len(val_dataset),
        shuffle=False,
        num_workers=args.num_workers,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    num_node_features, num_edge_features = get_num_features(train_dataset[0])
    model = GNN(
        num_node_features,
        num_edge_features,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        conv_name=args.conv,
        pool=args.pool,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.8, patience=10, min_lr=1e-4
    )
    min_error = None

    # Early stopping
    last_loss = 100
    patience = 4
    trigger_times = 0

    for epoch in range(1, args.epoch + 1):

        t1 = time.time()
        lr = scheduler.optimizer.param_groups[0]["lr"]

        loss = train_epoch(
            train_loader,
            model,
            optimizer,
            loss_type=args.loss,
            lambda1=args.lambda1,
        )

        rmse_val, pcc_val = test_epoch(val_loader, model)
        scheduler.step(rmse_val)
        if min_error is None or rmse_val <= min_error:
            min_error = rmse_val

        t2 = time.time()
        rmse_test, pcc_test = test_epoch(test_loader, model)
        t3 = time.time()

        if epoch % args.verbose == 0:
            print(
                "Epoch{:4d}[{:.3f}s]: LR: {:.5f}, Loss: {:.5f}, Test Loss: {:.5f}, Test PCC: {:.5f}".format(
                    epoch, t3 - t1, lr, loss, rmse_test, pcc_test
                )
            )
            continue

        print(
            "Epoch{:4d}[{:.3f}s]: LR: {:.5f}, Loss: {:.5f}, Validation Loss: {:.5f}, Val PCC: {:.5f}".format(
                epoch, t2 - t1, lr, loss, rmse_val, pcc_val
            )
        )

        if track_wandb:
            wandb.log({
                'epoch': epoch, 
                'rmse_train': loss,
                'rmse_val': rmse_val,
                'rmse_test': rmse_test
            })

        # Early stopping
        if epoch > 100:
            if loss > last_loss:
                trigger_times += 1
                print("Trigger Times:", trigger_times)

                if trigger_times >= patience:
                    print(
                        f"Early stopping after {epoch} epochs!\nStart to test process."
                    )
                    break

            else:
                print("trigger times: 0")
                trigger_times = 0
        last_loss = loss

    print("Final test rmse: {:.4f}".format(rmse_test))
    print("Final test pcc: {:.4f}".format(pcc_test))

    return model, rmse_test, pcc_test

def save_gnn_scores(args, rmse_test, pcc_test):

    train_params = (
        f"{args.conv}_{args.loss}_{args.pool}_{args.lambda1}_{args.explainer}"
    )
    # Save GNN scores
    os.makedirs(log_dir, exist_ok=True)
    global_res_path = osp.join(
        log_dir, f"model_scores_gnn_{train_params}_{args.target}.csv"
    )
    df = pd.DataFrame(
        [
            [
                args.target,
                args.seed,
                args.conv,
                args.pool,
                args.loss,
                args.lambda1,
                args.explainer,
                args.epoch,
                args.batch_size,
                args.lr,
                rmse_test,
                pcc_test,
            ]
        ],
        columns=[
            "target",
            "seed",
            "conv",
            "pool",
            "loss",
            "lambda1",
            "explainer",
            "epoch",
            "batch_size",
            "lr",
            "rmse_test",
            "pcc_test",
        ],
    )
    # df.to_csv(global_res_path, index=False)

def save_gnn_model(args, model):

    train_params = (
        f"{args.conv}_{args.loss}_{args.pool}_{args.lambda1}_{args.explainer}"
    )
    ###### Save trained GNN ######
    save_path = f"{args.target}_{train_params}.pt"
    os.makedirs(args.model_path, exist_ok=True)
    os.makedirs(osp.join(args.model_path, args.target), exist_ok=True)
    model_file = osp.join(args.model_path, args.target, save_path)
    if os.path.exists(model_file):
        print("This model has already been saved!\n")
        answer = input("Do you want to overwrite it? (y/n)")
        if answer == "yes" or answer == "y":
            torch.save(model.state_dict(), model_file)
            print("Model saved!\n")
        elif answer == "no" or answer == "n":
            print("Model not saved!\n")
        else:
            print("Invalid answer!\n")
    else:
        torch.save(model.state_dict(), model_file)
        print("Model saved!\n")

def main(save_model=False):
    model, rmse_test, pcc_test = train_gnn(args)
    save_gnn_scores(args, rmse_test, pcc_test)
    if save_model:
        save_gnn_model(args, model)


if __name__ == "__main__":


    """
    parser = overall_parser()
    args = parser.parse_args()

    for loss in ["MSE", "MSE+AC", "MSE+UCN"]:
        args.loss = loss

        project_name = 'sweep_for_{}_{}'.format(args.target, args.loss)

        # Define sweep config
        sweep_configuration = {
            'method': 'random',
            'name': 'sweep',
            'metric': {'goal': 'minimize', 'name': 'rmse_test'},
            'parameters': 
            {
                'batch_size': {'values': [16, 32, 64]},
                'epoch': {'values': [100, 200, 300]},
                'lr': {'max': 0.1, 'min': 0.0001}
            }
        }

        # Initialize sweep by passing in config. (Optional) Provide a name of the project.
        sweep_id = wandb.sweep(sweep=sweep_configuration, project=project_name)

        # Start sweep job.
        wandb.init(project=project_name, config=args)
        wandb.agent(sweep_id, function=main, count=10)
    """


    parser = overall_parser()
    args = parser.parse_args()

    for target in ["1DYR-TOP", "1F0R-815"]:
        args.target = target

        project_name = 'gridsearch_arch_for_{}'.format(args.target)

        for loss in ["MSE+UCN"]:
            for batch_size in [16]:
                for num_layers in [2, 3]:
                    for hidden_dim in [16, 32, 64]:
                        for lr in [0.01, 0.005, 0.001]:
                            args.batch_size = batch_size
                            args.num_layers = num_layers
                            args.hidden_dim = hidden_dim
                            args.loss = loss
                            args.lr = lr
                            args.epoch = 300

                            run_name = 'search_for_{}_{}_{}_{}_{}'.format(args.target, args.loss, num_layers, hidden_dim, lr)

                            run = wandb.init(project=project_name, name=run_name)
                            train_gnn(args, save_model=False, track_wandb=True)
                            run.finish()
