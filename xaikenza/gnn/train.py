import os
import os.path as osp
import time

import dill
import pandas as pd
import torch
import wandb
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import DataLoader
from xaikenza.dataset.pair import get_num_features
from xaikenza.gnn.model import GNN
from xaikenza.utils.parser_utils import overall_parser
from xaikenza.utils.train_utils import DEVICE, test_epoch, train_epoch  # move to
from xaikenza.utils.utils import set_seed

import sys

print(sys.path)


os.environ["WANDB_SILENT"] = "true"
save_dir = os.getenv("AMLT_OUTPUT_DIR", "/tmp")
data_dir = os.environ["AMLT_DATA_DIR"]


def train_gnn(args, save_model=True):
    set_seed(args.seed)
    train_params = (
        f"{args.conv}_{args.loss}_{args.pool}_{args.lambda1}_{args.explainer}"
    )

    # wandb.init(project=f'{train_params}_training', entity='k-amara', name=args.target)

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

    # print(len(train_dataset), len(val_dataset), len(test_dataset))

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
        # wandb.watch(model, nn.MSELoss(), log="all")

        rmse_val, pcc_val = test_epoch(val_loader, model)
        scheduler.step(rmse_val)
        if min_error is None or rmse_val <= min_error:
            min_error = rmse_val

        t2 = time.time()
        rmse_test, pcc_test = test_epoch(test_loader, model)
        t3 = time.time()

        # wandb.log({"epoch": epoch, "loss": loss, "pcc_test": pcc_test, "rmse_test": rmse_test})

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

        # Early stopping
        if epoch > 50:
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

    # Save GNN scores
    os.makedirs(args.log_path, exist_ok=True)
    global_res_path = osp.join(args.log_path, f"model_scores_gnn_{train_params}.csv")
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
            "rmse_test",
            "pcc_test",
        ],
    )
    df.to_csv(global_res_path, index=False)

    ###### Save trained GNN ######
    if save_model:
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


if __name__ == "__main__":

    parser = overall_parser()

    # GNN training parameters
    parser.add_argument("--epoch", type=int, default=200, help="Number of epoch.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument(
        "--verbose", type=int, default=10, help="Interval of evaluation."
    )
    parser.add_argument("--num_workers", type=int, default=0, help="number of workers")

    args = parser.parse_args()

    train_gnn(args, save_model=True)
