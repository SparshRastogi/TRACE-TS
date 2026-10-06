import numpy as np
import torch
import torch.nn.functional as F

MOMENT_TARGET_SEQ_LEN = 512
MOMENT_MODEL_NAME = "AutonLab/MOMENT-1-large"
MOMENT_EMBEDDING_DIM = 1024


def _resize_batch_for_moment(samples_np, target_len=MOMENT_TARGET_SEQ_LEN):
    x = torch.tensor(samples_np, dtype=torch.float32).permute(0, 2, 1)
    x_resized = F.interpolate(x, size=target_len, mode="linear", align_corners=False)
    return x_resized


def load_moment_model(model_name, device):
    from momentfm import MOMENTPipeline

    print(f"  Loading MOMENT from: {model_name}")
    model = MOMENTPipeline.from_pretrained(
        model_name, model_kwargs={"task_name": "embedding"}
    )
    model.init()
    model = model.to(device)
    model.eval()
    return model


def compute_moment_embeddings_batch(
    moment_model, samples_np, device, batch_size=32, seq_budget=12288
):
    N = samples_np.shape[0]
    all_embeddings = []
    n_channels = samples_np.shape[2]
    batch_size = max(batch_size, seq_budget // max(1, n_channels))
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        x_resized = _resize_batch_for_moment(samples_np[start:end]).to(device)
        with torch.no_grad():
            out = moment_model(x_enc=x_resized, reduction="none")
        enc = out.embeddings.mean(dim=2)
        emb = enc.reshape(enc.shape[0], -1).cpu().numpy()
        all_embeddings.append(emb)
    return np.concatenate(all_embeddings, axis=0)
