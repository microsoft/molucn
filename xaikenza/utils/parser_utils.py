import argparse

from xaikenza.utils.path import COLOR_DIR, DATA_DIR, LOG_DIR, MODEL_DIR, RESULT_DIR


# Parse train only at the beginning in train_gnn.py
# Shared parse for explain.py and train_gnn.py
def overall_parser():
    parser = argparse.ArgumentParser(description="Train GNN Model")

    parser.add_argument("--dest", type=str, default="/home/t-kenzaamara/internship2022")
    parser.add_argument(
        "--wandb",
        type=str,
        default="False",
        help="if set to True, the training curves are shown on wandb",
    )
    parser.add_argument("--cuda", type=int, default=0, help="GPU device.")
    parser.add_argument("--seed", type=int, default=1337, help="seed")

    # Saving paths
    parser.add_argument(
        "--data_path", nargs="?", default=DATA_DIR, help="Input data path."
    )
    parser.add_argument(
        "--model_path",
        nargs="?",
        default=MODEL_DIR,
        help="path for saving trained model.",
    )
    parser.add_argument(
        "--log_path",
        nargs="?",
        default=LOG_DIR,
        help="path for saving gnn scores (rmse, pcc).",
    )
    parser.add_argument(
        "-color_path",
        nargs="?",
        default=COLOR_DIR,
        help="path for saving node colors.",
    )

    parser.add_argument(
        "--result_path",
        nargs="?",
        default=RESULT_DIR,
        help="path for saving the feature attribution scores (accs, f1s).",
    )

    # Choose protein target
    parser.add_argument(
        "--target", type=str, default="1D3G-BRE", help="Protein target."
    )

    # Model parameters
    parser.add_argument(
        "--num_layers", type=int, default=2, help="number of Convolution layers(units)"
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=32,
        help="number of neurons in first hidden layer",
    )
    parser.add_argument(
        "--conv", type=str, default="nn", help="Type of convolutional layer."
    )  # gine, gat, gen, nn
    parser.add_argument(
        "--pool", type=str, default="mean", help="pool strategy."
    )  # mean, max, add, att
    # parser.add_argument('--heads', type=int, default=1,
    # help='Number of pool heads. If heads = 1, then same pool is applied to the whole molecule and the uncommon nodes.\
    # If heads = 2, then two different pool is applied to the whole molecule and the uncommon nodes.')

    # Loss type
    parser.add_argument(
        "--loss", type=str, default="MSE", help="Type of loss for training GNN."
    )  # ['MSE', 'MSE+UCN', 'MSE+UCNlocal, 'MSE+UCN+AC']
    parser.add_argument(
        "--lambda1",
        type=float,
        default=1.0,
        help="Hyperparameter determining the importance of UCN Loss",
    )

    # Train test val split
    parser.add_argument(
        "--test_set_size", type=float, default=0.2, help="test set size (ratio)"
    )
    parser.add_argument(
        "--val_set_size", type=float, default=0.1, help="validation set size (ratio)"
    )

    # GNN training parameters
    parser.add_argument("--epoch", type=int, default=200, help="Number of epoch.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument(
        "--verbose", type=int, default=10, help="Interval of evaluation."
    )
    parser.add_argument("--num_workers", type=int, default=0, help="number of workers")

    parser.add_argument(
        "--explainer", type=str, default="gradinput", help="Feature attribution method"
    )  # gradinput, ig

    return parser
