from pathlib import Path
import torch as T
from tqdm import trange
import argparse

from src.distributions import get_distribution
from src.models import get_model
from src.utils import linear_warmup_cosine_decay
from src.plotting import get_cols, make_plots


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--x0_dataset", type=str, default="moons")
    parser.add_argument("--x1_dataset", type=str, default="normal")
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--model_name", type=str, default="linear_uniform")
    parser.add_argument("--plot_dir", type=str, default="plots")
    parser.add_argument("--test_check", type=int, default=50_000)
    parser.add_argument("--test_size", type=int, default=1000)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--warmup", type=int, default=5_000)
    parser.add_argument("--max_iter", type=int, default=50_000)
    return parser.parse_args()


def main() -> None:
    args = get_args()
    device = T.device("cuda" if T.cuda.is_available() else "cpu")

    # Make the datasets
    x0_dataset = get_distribution(args.x0_dataset, device=device)
    x1_dataset = get_distribution(args.x1_dataset, device=device)
    minmax = max(x0_dataset.minmax, x1_dataset.minmax)

    # Define the model
    model = get_model(
        args.model_name,
        input_dim=args.dim,
        base_dist=args.x1_dataset,
        minmax=minmax,
        max_steps=args.max_iter,
    )
    model.to(device)
    print(model)

    # Optimizer and scheduler
    optimizer = T.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = linear_warmup_cosine_decay(
        optimizer, warmup_steps=args.warmup, total_steps=args.max_iter
    )

    # Directories
    suffix = "_".join([args.model_name, args.x0_dataset, args.x1_dataset])
    plot_dir = Path(args.plot_dir, suffix)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Reusable test sets
    x0_test = x0_dataset.sample(args.test_size)
    x1_test = x1_dataset.sample(args.test_size)
    cols = get_cols(x1_test[:, 0].cpu().numpy())

    # Training
    pbar = trange(args.max_iter, mininterval=1, miniters=0)
    for it in pbar:
        model.train()
        optimizer.zero_grad()

        # Get training samples
        x0 = x0_dataset.sample(args.batch_size)
        x1 = x1_dataset.sample(args.batch_size)

        # Get loss
        loss = model.train_step(x0, x1, it)

        # Gradient step
        loss.backward()
        T.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        # Set the tqdm bar to show the loss and the lr
        lr = optimizer.param_groups[0]["lr"]
        pbar.set_postfix(loss=loss.item(), lr=lr, refresh=False)

        # Plotting the density of the model
        if (it % args.test_check == 0 and it > 0) or it == args.max_iter - 1:
            model.eval()
            make_plots(model, plot_dir, x0_test, x1_test, cols)


if __name__ == "__main__":
    main()
