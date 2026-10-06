import numpy as np
import torch
import torch.nn.functional as F

MANTIS_TARGET_SEQ_LEN = 512
MANTIS_RETURN_TRANSF_LAYER = 2
MANTIS_OUTPUT_TOKEN = "combined"
MANTIS_CHECKPOINT_NAME = "paris-noah/MantisV2"


def _resize_batch_for_mantis(samples_np, target_len=MANTIS_TARGET_SEQ_LEN):
    x = torch.tensor(samples_np, dtype=torch.float32).permute(0, 2, 1)
    x_resized = F.interpolate(x, size=target_len, mode="linear", align_corners=False)
    return x_resized.numpy()


def load_mantis_model(checkpoint_name, device):
    from mantis.architecture import MantisV2
    from mantis.trainer import MantisTrainer

    print(f"  Loading MantisV2 from: {checkpoint_name}")
    network = MantisV2(
        return_transf_layer=MANTIS_RETURN_TRANSF_LAYER,
        output_token=MANTIS_OUTPUT_TOKEN,
        device=device,
    )
    network = network.from_pretrained(checkpoint_name)
    trainer = MantisTrainer(device=device, network=network)
    return trainer


def compute_mantis_embeddings_batch(mantis_trainer, samples_np, device, batch_size=32):
    N = samples_np.shape[0]
    all_embeddings = []
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        x_resized = _resize_batch_for_mantis(samples_np[start:end])
        emb = mantis_trainer.transform(x_resized)
        all_embeddings.append(emb)
    return np.concatenate(all_embeddings, axis=0)
