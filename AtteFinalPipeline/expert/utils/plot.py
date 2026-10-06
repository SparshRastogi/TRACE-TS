import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from AtteFinalPipeline.expert.utils.utils import makedir

__all__ = ["plot_pie", "plot_segment", "plot_confusion"]


def plot_pie(target, prefix, path_save, class_map=None, verbose=False):
    if not os.path.exists(path_save):
        makedir(path_save)
    if not class_map:
        class_map = [str(idx) for idx in range(len(set(target)))]
    color_map = sns.color_palette("husl", n_colors=len(class_map))
    target_dict = {
        label: np.sum(target == label_idx) for label_idx, label in enumerate(class_map)
    }
    target_count = list(target_dict.values())
    if verbose:
        print(f"[-] {prefix} target distribution: {target_dict}")
        print("--" * 50)
    fig, ax = plt.subplots()
    ax.axis("equal")
    explode = tuple(np.ones(len(class_map)) * 0.05)
    patches, texts, autotexts = ax.pie(
        target_count,
        explode=explode,
        labels=class_map,
        autopct="%1.1f%%",
        shadow=False,
        startangle=0,
        colors=color_map,
        wedgeprops={"linewidth": 1, "edgecolor": "k"},
    )
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc="center left", bbox_to_anchor=(1.2, 0.5))
    plt.tight_layout()
    save_name = os.path.join(path_save, prefix + ".png")
    fig.savefig(save_name, bbox_inches="tight")
    plt.close()


def plot_segment(
    data, target, index, prefix, path_save, num_class, target_pred=None, class_map=None
):
    if not os.path.exists(path_save):
        makedir(path_save)
    if not class_map:
        class_map = [str(idx) for idx in range(num_class)]
    gt = int(target)
    title_color = "black"
    if target_pred is not None:
        pred = int(target_pred)
        msg = f"#{int(index)}     ground-truth:{class_map[gt]}     prediction:{class_map[pred]}"
        title_color = "green" if gt == pred else "red"
    else:
        msg = f"#{int(index)}     ground-truth:{class_map[gt]}            "
    fig, ax = plt.subplots(figsize=(5, 2))
    ax.plot(data.numpy())
    ax.set_xlim(0, data.shape[0])
    ax.set_ylim(-5, 5)
    ax.set_title(msg, color=title_color)
    plt.tight_layout()
    save_name = os.path.join(
        path_save,
        prefix + "_" + class_map[int(target)] + "_" + str(int(index)) + ".png",
    )
    fig.savefig(save_name, bbox_inches="tight")
    plt.close()


def plot_confusion(
    y_true, y_pred, path_save, epoch, normalize=True, cmap=plt.cm.Blues, class_map=None
):
    if not os.path.exists(path_save):
        makedir(path_save)
    if not class_map:
        all_labels = sorted(
            set(np.asarray(y_true).tolist()) | set(np.asarray(y_pred).tolist())
        )
        class_map = [str(idx) for idx in all_labels]
        labels_for_cm = all_labels
    else:
        labels_for_cm = list(range(len(class_map)))
    cm = confusion_matrix(y_true, y_pred, labels=labels_for_cm)
    if normalize:
        row_sums = cm.sum(axis=1)[:, np.newaxis].astype("float")
        row_sums[row_sums == 0] = 1.0
        cm = cm.astype("float") / row_sums
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_map,
        yticklabels=class_map,
        title=f"Epoch {epoch}",
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fmt = ".1f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    high, low = ax.get_ylim()
    ax.set_ylim(high + 0.5, low - 0.5)
    fig.tight_layout()
    plt.tight_layout()
    plt.savefig(
        os.path.join(path_save, "cm_" + str(epoch) + ".png"), bbox_inches="tight"
    )
    plt.close()
