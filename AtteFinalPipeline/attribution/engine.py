import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.attr import IntegratedGradients


class _LogitWrapper(nn.Module):
    def __init__(self, model, use_amp=False, device_type="cuda"):
        super().__init__()
        self.model = model
        self.use_amp = use_amp and device_type == "cuda"
        self.device_type = device_type

    def forward(self, x):
        if self.use_amp:
            with torch.autocast(device_type=self.device_type, dtype=torch.float16):
                _, logits, _ = self.model(x)
        else:
            _, logits, _ = self.model(x)
        return logits.float()


class XAIEngine:
    def __init__(
        self,
        model,
        ig_baseline,
        shap_background,
        device,
        use_compile=True,
        use_amp=False,
    ):
        self.device = device
        device_type = device.type
        self.wrapper = _LogitWrapper(model, use_amp=use_amp, device_type=device_type)
        self.wrapper.train()
        for m in self.wrapper.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                m.eval()
        if use_compile and torch.cuda.is_available():
            print("  Compiling wrapper with torch.compile(mode='default') ...")
            self.compiled_wrapper = torch.compile(
                self.wrapper, mode="default", fullgraph=False
            )
            print("  Compilation registered (JIT will fire on first call).")
        else:
            self.compiled_wrapper = self.wrapper
        self.ig = IntegratedGradients(self.compiled_wrapper)
        self.ig_baseline = ig_baseline
        self._shap_background = shap_background
        self._n_bg = shap_background.shape[0]
        self._shap_rng = torch.Generator(device=device)
        self._shap_rng.manual_seed(42)
        print("  Caching num_classes (may trigger first JIT compile, ~30s) ...")
        with torch.no_grad():
            self._num_classes = int(self.compiled_wrapper(self.ig_baseline).shape[-1])
        print(
            f"  XAIEngine ready | num_classes={self._num_classes} | compile={('ON' if use_compile else 'OFF')} | AMP={('ON' if use_amp else 'OFF')} | cuDNN=ON"
        )

    def warm_up(self, sample_shape, target_class=0, n_steps=5, production_n_steps=None):
        print(f"  Warm-up pass 1: fast kernel (n_steps={n_steps}) ...")
        dummy = torch.zeros((1, *sample_shape), device=self.device)
        _ = self.ig.attribute(
            dummy,
            baselines=self.ig_baseline,
            target=target_class,
            n_steps=n_steps,
            internal_batch_size=n_steps,
        )
        _ = self.compute_shap_batch(
            dummy, torch.tensor([target_class], device=self.device)
        )
        print(f"  Warm-up pass 1 done.")
        if production_n_steps is not None and production_n_steps != n_steps:
            print(
                f"  Warm-up pass 2: production kernel (n_steps={production_n_steps}) ..."
            )
            _ = self.ig.attribute(
                dummy,
                baselines=self.ig_baseline,
                target=target_class,
                n_steps=production_n_steps,
                internal_batch_size=production_n_steps,
            )
            print(f"  Warm-up pass 2 done.")
        print("  Warm-up complete — all Triton kernels cached.")

    def compute_ig(self, data_tensor_on_gpu, target_class, n_steps=25):
        inp = data_tensor_on_gpu.unsqueeze(0)
        attr = self.ig.attribute(
            inp,
            baselines=self.ig_baseline,
            target=int(target_class),
            n_steps=n_steps,
            internal_batch_size=n_steps,
        )
        attr = attr[0].detach().cpu().numpy()
        np.abs(attr, out=attr)
        mx = attr.max()
        if mx > 0:
            attr *= 1.0 / mx
        return attr

    def compute_shap(self, data_tensor_on_gpu, target_class):
        return self.compute_shap_batch(
            data_tensor_on_gpu.unsqueeze(0),
            torch.tensor([int(target_class)], device=self.device),
        )[0]

    def compute_ig_batch(self, batch_tensor, target_classes, n_steps=25):
        B = batch_tensor.shape[0]
        internal_bs = max(1, min(n_steps * B, 512))
        baseline = self.ig_baseline.expand(B, -1, -1)
        attr = self.ig.attribute(
            batch_tensor,
            baselines=baseline,
            target=target_classes,
            n_steps=n_steps,
            internal_batch_size=internal_bs,
        )
        attr_np = attr.detach().cpu().numpy()
        np.abs(attr_np, out=attr_np)
        maxvals = attr_np.reshape(B, -1).max(axis=1)
        nonzero = maxvals > 0
        attr_np[nonzero] /= maxvals[nonzero, None, None]
        return attr_np

    def compute_shap_batch(self, batch_tensor, target_classes):
        B = batch_tensor.shape[0]
        bg = self._shap_background.detach()
        n_bg = self._n_bg
        alphas = torch.rand(
            B,
            n_bg,
            1,
            1,
            device=self.device,
            dtype=batch_tensor.dtype,
            generator=self._shap_rng,
        )
        x_exp = batch_tensor.unsqueeze(1)
        bg_exp = bg.unsqueeze(0)
        diffs = x_exp - bg_exp
        interp = (bg_exp + alphas * diffs).reshape(B * n_bg, *batch_tensor.shape[1:])
        interp = interp.requires_grad_(True)
        self.compiled_wrapper.zero_grad(set_to_none=True)
        logits = self.compiled_wrapper(interp)
        targets_expanded = target_classes.repeat_interleave(n_bg)
        target_logits = logits.gather(1, targets_expanded.unsqueeze(1)).squeeze(1)
        target_logits.sum().backward()
        grads = interp.grad.reshape(B, n_bg, *batch_tensor.shape[1:])
        diffs_r = diffs
        attr = (grads * diffs_r).mean(dim=1)
        attr_np = attr.detach().cpu().numpy()
        np.abs(attr_np, out=attr_np)
        maxvals = attr_np.reshape(B, -1).max(axis=1)
        nonzero = maxvals > 0
        attr_np[nonzero] /= maxvals[nonzero, None, None]
        return attr_np

    def compute_combined_batch(
        self, batch_tensor, target_classes, n_steps=25, eps=1e-08
    ):
        ig_attrs = self.compute_ig_batch(batch_tensor, target_classes, n_steps)
        shap_attrs = self.compute_shap_batch(batch_tensor, target_classes)
        return self._combine_batch(ig_attrs, shap_attrs, eps=eps)

    def _combine_batch(self, ig_attrs, shap_attrs, eps=1e-08):
        combined = np.sqrt((ig_attrs + eps) * (shap_attrs + eps))
        B = combined.shape[0]
        maxvals = combined.reshape(B, -1).max(axis=1)
        nonzero = maxvals > 0
        combined[nonzero] /= maxvals[nonzero, None, None]
        return combined

    def compute_combined(self, data_tensor_on_gpu, target_class, n_steps=25, eps=1e-08):
        ig_attr = self.compute_ig(data_tensor_on_gpu, target_class, n_steps)
        shap_attr = self.compute_shap(data_tensor_on_gpu, target_class)
        return self._combine(ig_attr, shap_attr, eps=eps)

    def _combine(self, ig_attr, shap_attr, eps=1e-08):
        combined = np.sqrt((ig_attr + eps) * (shap_attr + eps))
        mx = combined.max()
        if mx > 0:
            combined *= 1.0 / mx
        return combined


def batched_predict(model, data, device, batch_size=256):
    N = data.shape[0]
    all_probs = []
    with torch.inference_mode():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch = torch.tensor(data[start:end], dtype=torch.float32).to(device)
            _, logits, _ = model(batch)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
    all_probs = np.concatenate(all_probs, axis=0)
    pred_classes = np.argmax(all_probs, axis=1)
    return (pred_classes, all_probs)
