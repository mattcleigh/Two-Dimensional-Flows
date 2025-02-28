import numpy as np
import torch as T
from matplotlib import pyplot as plt
from torch import nn
from pathlib import Path


def get_cols(x):
    cmap = plt.get_cmap("turbo")
    ranks = np.argsort(np.argsort(x))
    normalized_ranks = ranks / np.max(ranks)
    return cmap(normalized_ranks)


def to_np(x: T.Tensor | tuple | None) -> np.ndarray | tuple | None:
    if x is None:
        return None
    if isinstance(x, tuple):
        return tuple(to_np(a) for a in x)
    return x.cpu().detach().numpy()


def plot_interpolations(
    plot_dir: Path,
    truth: T.Tensor,
    all_stages: list,
    cols: np.ndarray,
):
    all_stages = [a.unsqueeze(-1) for a in all_stages]
    all_stages = to_np(T.cat(all_stages, dim=-1))
    truth = to_np(truth)
    plt.figure(figsize=(4, 4))
    plt.scatter(truth[:, 0], truth[:, 1], c="grey", alpha=0.2)
    for s in all_stages:
        plt.plot(s[0], s[1], "k-", alpha=0.1)
    plt.scatter(all_stages[:, 0, -1], all_stages[:, 1, -1], c=cols, alpha=0.5)
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.tight_layout()
    plt.gca().set_aspect("equal", adjustable="box")
    plt.savefig(plot_dir / "interpolate.png")
    plt.close()


def plot_oned(plot_dir, all_stages, times, x0_dataset, x1_dataset, ylim):
    all_stages = [a[:, 0].unsqueeze(-1) for a in all_stages]
    all_stages = T.cat(all_stages, dim=-1).detach().cpu().numpy()
    cols = get_cols(all_stages[:, -1])

    # Plot the learned marginal interpolations
    plt.figure(figsize=(5, 4))
    for i in range(len(all_stages)):
        plt.plot(times, all_stages[i], color=cols[i], alpha=0.1)
    plt.grid()
    plt.xlabel(r"$t$")
    plt.ylim(ylim)
    plt.tight_layout()
    plt.savefig(plot_dir / "oned.png")
    plt.close()

    # Plot the training target interpolations
    n = all_stages.shape[0]
    x0 = x0_dataset.sample(n)[:, 0].cpu().numpy()
    x1 = x1_dataset.sample(n)[:, 0].cpu().numpy()
    cols = get_cols(x1)
    plt.figure(figsize=(5, 4))
    for i in range(n):
        plt.plot([min(times), max(times)], [x0[i], x1[i]], color=cols[i], alpha=0.1)
    plt.xlabel(r"$t$")
    plt.ylim(ylim)
    plt.grid()
    plt.tight_layout()
    plt.savefig(plot_dir / "oned_target.png")
    plt.close()


def plot_stages(
    plot_dir: Path,
    cols: np.ndarray,
    stages: list,
    num_stages: int = 4,
):
    n = len(stages)
    sel_stages = [stages[(n - 1) // num_stages * i] for i in range(num_stages + 1)]
    sel_stages.insert(0, None)  # Add none for the first stage
    for i in range(1, len(sel_stages)):
        cur = to_np(sel_stages[i])
        old = to_np(sel_stages[i - 1])

        plt.figure(figsize=(4, 4))
        if old is not None:
            for j in range(cur.shape[0]):
                plt.plot(
                    [old[j, 0], cur[j, 0]],
                    [old[j, 1], cur[j, 1]],
                    color="black",
                    alpha=0.1,
                )
        plt.scatter(cur[:, 0], cur[:, 1], c=cols, alpha=0.5)
        plt.xlim(-3, 3)
        plt.ylim(-3, 3)
        plt.tight_layout()
        plt.gca().set_aspect("equal", adjustable="box")
        plt.savefig(plot_dir / f"samples_{i}.png")
        plt.close()


def plot_gen(plot_dir: Path, x0_gen: T.Tensor, x0_test: T.Tensor, cols):
    x0_gen = to_np(x0_gen)
    x0_test = to_np(x0_test)
    plt.figure(figsize=(4, 4))
    plt.scatter(x0_test[:, 0], x0_test[:, 1], c="grey", alpha=0.1)
    plt.scatter(x0_gen[:, 0], x0_gen[:, 1], c=cols, alpha=0.5)
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.tight_layout()
    plt.gca().set_aspect("equal", adjustable="box")
    plt.savefig(plot_dir / "gen.png")
    plt.close()


@T.no_grad()
def make_plots(
    model: nn.Module,
    plot_dir: Path,
    x0_test: T.Tensor,
    x1_test: T.Tensor,
    cols: np.ndarray,
):
    stages = model.gen_stages(x1_test)
    plot_stages(plot_dir, cols, stages, num_stages=4)
    plot_gen(plot_dir, stages[-1], x0_test, cols)
    plot_interpolations(plot_dir, x0_test, stages, cols)
    # plot_oned(plot_dir, stages, model.times, x0_dataset, x1_dataset, xlim)
