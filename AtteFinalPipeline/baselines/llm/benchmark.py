from __future__ import annotations
from AtteFinalPipeline.paths import UCIHAR_DATA_DIR
import argparse
import json
import os
import sys
import time
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn import metrics as skmetrics

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
NIM_MODEL = "deepseek-ai/deepseek-v4-pro"
RATE_LIMIT_RPM = 35
CONCURRENCY_DEFAULT = 16
EVAL_N_DEFAULT = 400
SEED = 42
STREAM_STALL_TIMEOUT_S = 180.0
HEARTBEAT_S = 10.0
HTTP_TIMEOUT_S = 300.0
USCHAD_DECIM = 4
ALL_DATASETS = [
    "opportunity",
    "pamap2",
    "uschad",
    "ucihar",
    "mhealth",
    "shoaib",
    "capture24",
]
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def _build_args_for_dataset(dataset: str):
    from AtteFinalPipeline.expert.settings import _DATASET_CONFIGS

    if dataset not in _DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset '{dataset}'. Known: {list(_DATASET_CONFIGS)}"
        )
    cfg = dict(_DATASET_CONFIGS[dataset])
    ns = argparse.Namespace(dataset=dataset)
    for k, v in cfg.items():
        setattr(ns, k, v)
    ns.input_dim = cfg["input_dim"]
    ns.num_class = cfg["num_class"]
    ns.window = cfg["window"]
    ns.stride = cfg["stride"]
    ns.stride_test = cfg["stride_test"]
    return ns


def _ensure_processed(dataset: str) -> argparse.Namespace:
    ns = _build_args_for_dataset(dataset)
    if dataset == "ucihar":
        base = Path(str(UCIHAR_DATA_DIR))
        for f in ("X_train.npy", "X_test.npy", "y_train.npy", "y_test.npy"):
            if not (base / f).exists():
                raise FileNotFoundError(
                    f"UCI-HAR file missing: {base / f}. Update SensorDataset path or place the files there."
                )
        return ns
    sw_path = Path(ns.path_processed) / "test_sample_wise.npz"
    if not sw_path.exists():
        print(
            f"[*] {sw_path} missing — running preprocess_pipeline({dataset}) ...",
            flush=True,
        )
        from AtteFinalPipeline.data.preprocess import preprocess_pipeline

        preprocess_pipeline(ns)
    if not sw_path.exists():
        raise FileNotFoundError(
            f"After preprocess_pipeline, {sw_path} still missing. Check the prepare_*.py output for {{dataset}}."
        )
    return ns


def _load_test_windows(
    dataset: str, ns: argparse.Namespace
) -> Tuple[np.ndarray, np.ndarray]:
    if dataset == "ucihar":
        base = Path(str(UCIHAR_DATA_DIR))
        X = np.load(base / "X_test.npy").astype(np.float32)
        y = np.load(base / "y_test.npy").astype(np.int64)
        if y.min() > 0:
            y = y - 1
        return (X, y)
    sw_path = Path(ns.path_processed) / "test_sample_wise.npz"
    npz = np.load(sw_path)
    X = npz["data"].astype(np.float32)
    y = npz["target"].astype(np.int64)
    return (X, y)


def stratified_subsample_indices(
    y: np.ndarray, n_total: int, num_class: int, seed: int
) -> np.ndarray:
    if len(y) < n_total:
        raise ValueError(
            f"Only {len(y)} test windows available; cannot subsample {n_total}."
        )
    rng = np.random.default_rng(seed)
    by_cls: Dict[int, np.ndarray] = {c: np.where(y == c)[0] for c in range(num_class)}
    for c in by_cls:
        rng.shuffle(by_cls[c])
    base_quota = int(np.ceil(n_total / num_class))
    chosen: Dict[int, List[int]] = {}
    for c in range(num_class):
        avail = by_cls[c]
        take = min(base_quota, len(avail))
        chosen[c] = avail[:take].tolist()
    leftovers: Dict[int, List[int]] = {
        c: by_cls[c][len(chosen[c]) :].tolist() for c in range(num_class)
    }
    total = sum((len(v) for v in chosen.values()))
    while total > n_total:
        c_max = max(chosen, key=lambda c: (len(chosen[c]), -c))
        chosen[c_max].pop()
        total -= 1
    while total < n_total:
        candidates = [c for c, lo in leftovers.items() if lo]
        if not candidates:
            raise RuntimeError(
                f"Cannot reach n_total={n_total}: total available = {total + sum((len(v) for v in leftovers.values()))} but we still need {n_total - total}."
            )
        c_pick = max(candidates, key=lambda c: (len(leftovers[c]), -c))
        chosen[c_pick].append(leftovers[c_pick].pop(0))
        total += 1
    indices = np.array(sorted((i for v in chosen.values() for i in v)), dtype=np.int64)
    assert len(indices) == n_total, (len(indices), n_total)
    assert len(np.unique(indices)) == n_total, "indices not unique"
    return indices


SYSTEM_PROMPT = "You are an expert in human activity recognition (HAR) from wearable sensor data. You will be given a short window of multi-channel time-series sensor readings and you must classify it as exactly one activity from a fixed list. The values are z-score normalised per channel using training-set statistics, so they are dimensionless — do NOT interpret them as raw m/s^2, rad/s, mV, etc. Respond with a single JSON object and nothing else."


def _format_window_grid(window: np.ndarray, channel_names: List[str]) -> str:
    T, C = window.shape
    assert C == len(channel_names), (C, len(channel_names))
    header = "t," + ",".join(channel_names)
    rows = [header]
    for i in range(T):
        vals = ",".join((f"{v:.4f}" for v in window[i]))
        rows.append(f"{i},{vals}")
    return "\n".join(rows)


def build_prompt(
    dataset: str,
    window: np.ndarray,
    class_map: List[str],
    channel_names: List[str],
    sample_rate_hz: int,
    *,
    decimated: bool = False,
    decimate_factor: int = 1,
) -> str:
    from AtteFinalPipeline.baselines.llm.channel_metadata import get_dataset_description

    desc = get_dataset_description(dataset)
    T, C = window.shape
    timing = (
        f"effective sample rate {sample_rate_hz // decimate_factor} Hz (original {sample_rate_hz} Hz, decimated {decimate_factor}x)"
        if decimated
        else f"sample rate {sample_rate_hz} Hz"
    )
    class_lines = "\n".join((f"  {i}: {name}" for i, name in enumerate(class_map)))
    grid = _format_window_grid(window, channel_names)
    prompt = f'Dataset: {dataset}\n{desc}\n\nWindow: {T} timesteps × {C} channels, {timing}.\nValues are per-channel z-scores (training mean/std), so 0 ≈ channel mean and ±2 ≈ ~2 standard deviations.\n\nPossible activity labels (you MUST pick exactly one integer in [0, {len(class_map) - 1}]):\n{class_lines}\n\nSensor window (CSV; first column is timestep index):\n{grid}\n\nReturn a single JSON object with these exact keys and nothing else:\n{{\n  "label": <integer 0..{len(class_map) - 1}>,\n  "label_name": "<one of the names above, exact spelling>",\n  "reasoning": "<one or two sentences explaining the choice based on the signal patterns>"\n}}\n'
    return prompt


class RateLimiter:
    def __init__(self, rpm: int):
        self.rpm = rpm
        self.timestamps: List[float] = []
        self._lock = threading.Lock()

    def wait(self):
        while True:
            with self._lock:
                now = time.time()
                self.timestamps = [t for t in self.timestamps if now - t < 60.0]
                if len(self.timestamps) < self.rpm:
                    self.timestamps.append(now)
                    return
                sleep_for = 60.0 - (now - self.timestamps[0]) + 0.05
            if sleep_for > 0:
                time.sleep(min(sleep_for, 5.0))


def _make_client(api_key: str):
    if OpenAI is None:
        raise RuntimeError("openai package not installed. `pip install openai` first.")
    if not api_key:
        api_key = "EMPTY"
    return OpenAI(base_url=NIM_BASE_URL, api_key=api_key, timeout=HTTP_TIMEOUT_S)


_thread_local = threading.local()


def _wid() -> str:
    if not hasattr(_thread_local, "wid"):
        _thread_local.wid = threading.get_ident() & 65535
    return f"w{_thread_local.wid:04x}"


def call_nim(
    client,
    user_prompt: str,
    *,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    max_retries: int = 4,
    verbose_heartbeat: bool = True,
) -> Tuple[str, Dict[str, Any]]:
    last_err = None
    for attempt in range(max_retries):
        try:
            t0 = time.time()
            stream = client.chat.completions.create(
                model=NIM_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            chunks: List[str] = []
            last_byte_at = time.time()
            last_heartbeat_at = time.time()
            n_chunks = 0
            for ev in stream:
                if time.time() - last_byte_at > STREAM_STALL_TIMEOUT_S:
                    raise TimeoutError(
                        f"Stream stalled: no bytes for {STREAM_STALL_TIMEOUT_S:.0f}s."
                    )
                if not getattr(ev, "choices", None):
                    continue
                choice = ev.choices[0]
                delta = getattr(choice, "delta", None)
                if delta is None:
                    continue
                content_piece = getattr(delta, "content", None)
                reasoning_piece = getattr(delta, "reasoning_content", None)
                if content_piece:
                    chunks.append(content_piece)
                    last_byte_at = time.time()
                    n_chunks += 1
                elif reasoning_piece:
                    last_byte_at = time.time()
                if verbose_heartbeat and time.time() - last_heartbeat_at > HEARTBEAT_S:
                    elapsed = time.time() - t0
                    print(
                        f"[{_wid()} {elapsed:5.1f}s ..]",
                        end=" ",
                        flush=True,
                        file=sys.stderr,
                    )
                    last_heartbeat_at = time.time()
            text = "".join(chunks)
            latency = time.time() - t0
            meta = {
                "latency_s": round(latency, 3),
                "prompt_tokens": None,
                "completion_tokens": None,
                "stream_chunks": n_chunks,
                "attempts": attempt + 1,
            }
            return (text, meta)
        except Exception as e:
            last_err = e
            backoff = min(2**attempt, 30) + random.uniform(0, 1)
            print(
                f"  [warn {_wid()}] NIM call failed (attempt {attempt + 1}/{max_retries}): {type(e).__name__}: {e}. Sleeping {backoff:.1f}s.",
                flush=True,
            )
            time.sleep(backoff)
    raise RuntimeError(f"NIM call failed after {max_retries} attempts: {last_err}")


def _strip_think_tags(text: str) -> str:
    if "<think>" not in text:
        return text
    out = []
    i = 0
    while i < len(text):
        start = text.find("<think>", i)
        if start == -1:
            out.append(text[i:])
            break
        out.append(text[i:start])
        end = text.find("</think>", start)
        if end == -1:
            i = start + len("<think>")
            continue
        i = end + len("</think>")
    return "".join(out)


def _extract_json_block(text: str) -> Optional[str]:
    if not text:
        return None
    text = _strip_think_tags(text)
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start != -1:
                return text[start : i + 1]
    return None


def parse_response(
    text: str, num_class: int
) -> Tuple[Optional[int], Optional[str], Optional[str]]:
    block = _extract_json_block(text)
    if block is None:
        return (None, None, None)
    try:
        obj = json.loads(block)
    except json.JSONDecodeError:
        return (None, None, None)
    label = obj.get("label")
    if isinstance(label, bool):
        return (None, None, None)
    if isinstance(label, int):
        pass
    elif isinstance(label, float):
        if not label.is_integer():
            return (None, None, None)
        label = int(label)
    elif isinstance(label, str):
        s = label.strip()
        try:
            f = float(s)
        except (TypeError, ValueError):
            return (None, None, None)
        if not f.is_integer():
            return (None, None, None)
        label = int(f)
    else:
        return (None, None, None)
    if label < 0 or label >= num_class:
        return (None, None, None)
    return (label, obj.get("label_name"), obj.get("reasoning"))


def _process_one_window(
    *,
    win_idx: int,
    window: np.ndarray,
    true_y: int,
    dataset: str,
    class_map: List[str],
    channel_names: List[str],
    sample_rate_hz: int,
    decimated: bool,
    decim_fac: int,
    num_class: int,
    client,
    limiter: RateLimiter,
) -> Dict[str, Any]:
    prompt = build_prompt(
        dataset,
        window,
        class_map,
        channel_names,
        sample_rate_hz=sample_rate_hz,
        decimated=decimated,
        decimate_factor=decim_fac,
    )
    limiter.wait()
    print(
        f"  [{_wid()}] win={win_idx:>6d} dispatching ...", flush=True, file=sys.stderr
    )
    try:
        text, meta = call_nim(client, prompt)
    except Exception as e:
        text = ""
        meta = {
            "latency_s": None,
            "attempts": -1,
            "prompt_tokens": None,
            "completion_tokens": None,
            "error": repr(e),
        }
        print(
            f"  [error {_wid()}] win={win_idx} call_nim raised: {type(e).__name__}: {e}",
            flush=True,
        )
    pred, pred_name, reasoning = parse_response(text, num_class)
    if pred is None and text:
        strict = "Your previous response was not valid JSON or had an out-of-range label. Re-emit ONLY the JSON object with keys label (integer), label_name (string), reasoning (string). No other text, no <think> tags."
        limiter.wait()
        try:
            text2, meta2 = call_nim(client, prompt + "\n\n" + strict, max_tokens=512)
            pred, pred_name, reasoning = parse_response(text2, num_class)
            text = text + "\n--- RETRY ---\n" + (text2 or "")
            meta["retry_latency_s"] = meta2.get("latency_s")
            meta["retry_attempts"] = meta2.get("attempts")
        except Exception as e:
            meta["retry_error"] = repr(e)
    return {
        "dataset": dataset,
        "window_idx": int(win_idx),
        "true_label": int(true_y),
        "true_name": class_map[int(true_y)],
        "pred_label": pred,
        "pred_name": pred_name,
        "reasoning": reasoning,
        "raw_text": text,
        "meta": meta,
    }


def benchmark_one_dataset(
    dataset: str,
    *,
    eval_n: int,
    seed: int,
    client,
    limiter: RateLimiter,
    out_root: Path,
    max_calls_total: Optional[int] = None,
    plot_cm: bool = True,
    concurrency: int = CONCURRENCY_DEFAULT,
) -> Dict[str, Any]:
    from AtteFinalPipeline.baselines.llm.channel_metadata import (
        get_channel_names,
        get_sample_rate,
    )

    print("\n" + "=" * 72, flush=True)
    print(f"[dataset] {dataset}  (concurrency={concurrency})", flush=True)
    print("=" * 72, flush=True)
    ns = _ensure_processed(dataset)
    class_map: List[str] = list(ns.class_map)
    num_class = len(class_map)
    channel_names = get_channel_names(dataset, ns.input_dim)
    sample_rate_hz = get_sample_rate(dataset)
    X_all, y_all = _load_test_windows(dataset, ns)
    print(
        f"  test windows total: X={X_all.shape}  y={y_all.shape}  classes_present={sorted(np.unique(y_all).tolist())}",
        flush=True,
    )
    indices = stratified_subsample_indices(
        y_all, n_total=eval_n, num_class=num_class, seed=seed
    )
    cls_counts = Counter(y_all[indices].tolist())
    print(
        f"  selected {len(indices)} windows; class counts: { {class_map[c]: n for c, n in sorted(cls_counts.items())} }",
        flush=True,
    )
    out_dir = out_root / dataset / "llm_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "eval_indices.npz", indices=indices, y_true=y_all[indices])
    jsonl_path = out_dir / "responses.jsonl"
    cache: Dict[int, Dict[str, Any]] = {}
    if jsonl_path.exists():
        with jsonl_path.open() as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    cache[int(rec["window_idx"])] = rec
                except Exception:
                    continue
        print(
            f"  cache hit: {len(cache)} responses already saved at {jsonl_path}",
            flush=True,
        )
    decimated = dataset == "uschad"
    decim_fac = USCHAD_DECIM if decimated else 1
    todo: List[int] = [int(w) for w in indices.tolist() if int(w) not in cache]
    if max_calls_total is not None and len(todo) > max_calls_total:
        print(
            f"  [info] capping new calls at {max_calls_total} (of {len(todo)} uncached)",
            flush=True,
        )
        todo = todo[:max_calls_total]
    if not todo:
        print(
            f"  [info] all {len(indices)} windows already cached — skipping API calls.",
            flush=True,
        )
    else:
        print(
            f"  [info] dispatching {len(todo)} new calls with {concurrency} workers ...",
            flush=True,
        )
    write_lock = threading.Lock()
    completed = 0
    parse_fails_streaming = 0
    t_start = time.time()
    if todo:
        with (
            ThreadPoolExecutor(max_workers=concurrency) as pool,
            jsonl_path.open("a") as f_log,
        ):
            futures = {}
            for w in todo:
                window = X_all[w]
                if decimated:
                    window = window[::decim_fac]
                fut = pool.submit(
                    _process_one_window,
                    win_idx=int(w),
                    window=window,
                    true_y=int(y_all[w]),
                    dataset=dataset,
                    class_map=class_map,
                    channel_names=channel_names,
                    sample_rate_hz=sample_rate_hz,
                    decimated=decimated,
                    decim_fac=decim_fac,
                    num_class=num_class,
                    client=client,
                    limiter=limiter,
                )
                futures[fut] = int(w)
            for fut in as_completed(futures):
                w = futures[fut]
                try:
                    rec = fut.result()
                except Exception as e:
                    print(
                        f"  [error] worker for window {w} crashed: {type(e).__name__}: {e}",
                        flush=True,
                    )
                    rec = {
                        "dataset": dataset,
                        "window_idx": int(w),
                        "true_label": int(y_all[w]),
                        "true_name": class_map[int(y_all[w])],
                        "pred_label": None,
                        "pred_name": None,
                        "reasoning": None,
                        "raw_text": "",
                        "meta": {"error": f"worker_crash: {e!r}"},
                    }
                with write_lock:
                    f_log.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f_log.flush()
                    cache[int(w)] = rec
                completed += 1
                if rec.get("pred_label") is None:
                    parse_fails_streaming += 1
                report = (
                    concurrency == 1
                    or completed == 1
                    or completed % 5 == 0
                    or (completed == len(todo))
                )
                if report:
                    elapsed = time.time() - t_start
                    rate = completed / elapsed if elapsed > 0 else 0.0
                    eta = (len(todo) - completed) / rate if rate > 0 else float("inf")
                    pred_name = (
                        class_map[rec["pred_label"]]
                        if rec.get("pred_label") is not None
                        else "PARSE_FAIL"
                    )
                    lat = rec.get("meta", {}).get("latency_s")
                    lat_str = f"{lat:5.1f}s" if lat is not None else "  ?  "
                    print(
                        f"\n  [{completed:4d}/{len(todo)}] win={w:>6d} true={rec['true_name']:>22s}  pred={pred_name:>22s}  call={lat_str}  elapsed={elapsed:6.0f}s  rate={rate * 60:5.1f}/min  eta={eta:5.0f}s  fails={parse_fails_streaming}",
                        flush=True,
                    )
    y_true: List[int] = []
    y_pred: List[int] = []
    parse_fail = 0
    for win_idx in indices.tolist():
        rec = cache.get(int(win_idx))
        if rec is None or rec.get("pred_label") is None:
            parse_fail += 1
            continue
        y_true.append(int(rec["true_label"]))
        y_pred.append(int(rec["pred_label"]))
    y_true_a = np.array(y_true, dtype=np.int64)
    y_pred_a = np.array(y_pred, dtype=np.int64)
    if len(y_true_a) > 0:
        acc = 100.0 * skmetrics.accuracy_score(y_true_a, y_pred_a)
        fm = 100.0 * skmetrics.f1_score(
            y_true_a, y_pred_a, average="macro", zero_division=0
        )
        fw = 100.0 * skmetrics.f1_score(
            y_true_a, y_pred_a, average="weighted", zero_division=0
        )
    else:
        acc = fm = fw = 0.0
    metrics_out = {
        "dataset": dataset,
        "n_total": int(len(indices)),
        "n_valid": int(len(y_true_a)),
        "parse_failures": int(parse_fail),
        "accuracy": round(acc, 4),
        "f1_macro": round(fm, 4),
        "f1_weighted": round(fw, 4),
        "class_counts_eval": {
            class_map[c]: int(n)
            for c, n in sorted(Counter(y_all[indices].tolist()).items())
        },
        "model": NIM_MODEL,
        "seed": seed,
        "encoding": "raw_grid_zscore_4dp",
        "decimation": USCHAD_DECIM if dataset == "uschad" else 1,
        "window_shape": [int(X_all.shape[1]), int(X_all.shape[2])],
    }
    with (out_dir / "metrics.json").open("w") as f:
        json.dump(metrics_out, f, indent=2)
    print(
        f"\n  [metrics] acc={acc:.2f}%  F1-macro={fm:.2f}%  F1-weighted={fw:.2f}%  valid={len(y_true_a)}/{len(indices)}  parse_fail={parse_fail}",
        flush=True,
    )
    print(f"  [saved] {out_dir / 'metrics.json'}", flush=True)
    if plot_cm and len(y_true_a) > 0:
        try:
            from AtteFinalPipeline.expert.utils.plot import plot_confusion

            plot_confusion(
                y_true_a, y_pred_a, str(out_dir), epoch=-1, class_map=class_map
            )
            pngs = sorted(out_dir.glob("*.png"), key=lambda p: p.stat().st_mtime)
            if pngs:
                final = out_dir / "confusion_matrix.png"
                pngs[-1].replace(final)
                print(f"  [saved] {final}", flush=True)
        except Exception as e:
            print(f"  [warn] could not save confusion matrix: {e}", flush=True)
    with (out_dir / "run_config.json").open("w") as f:
        json.dump(
            {
                "dataset": dataset,
                "model": NIM_MODEL,
                "base_url": NIM_BASE_URL,
                "rate_limit_rpm": RATE_LIMIT_RPM,
                "concurrency": concurrency,
                "eval_n": eval_n,
                "seed": seed,
                "encoding": "raw_grid_zscore_4dp",
                "decimation": USCHAD_DECIM if dataset == "uschad" else 1,
                "input_dim": ns.input_dim,
                "num_class": num_class,
                "window": ns.window,
                "stride_test": ns.stride_test,
                "class_map": class_map,
                "channel_names": channel_names,
                "padding_applied": False,
                "system_prompt": SYSTEM_PROMPT,
                "streaming": True,
                "extra_body": None,
            },
            f,
            indent=2,
        )
    return metrics_out


def smoke_test(api_key: str) -> int:
    print("=" * 72, flush=True)
    print(
        "SMOKE TEST: one tiny call to confirm the NIM endpoint is reachable.",
        flush=True,
    )
    print("=" * 72, flush=True)
    if not api_key:
        print(
            "[FAIL] NVIDIA_API_KEY not set. `export NVIDIA_API_KEY=nvapi-...`",
            flush=True,
        )
        return 2
    client = _make_client(api_key)
    t0 = time.time()
    try:
        stream = client.chat.completions.create(
            model=NIM_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": 'Reply with exactly the JSON {"ok": true} and nothing else.',
                },
            ],
            temperature=0.0,
            max_tokens=64,
            stream=True,
        )
        print("[*] streaming response:", flush=True)
        text_parts: List[str] = []
        last_byte = time.time()
        for ev in stream:
            if time.time() - last_byte > 60.0:
                raise TimeoutError("Stream stalled for 60s")
            if not getattr(ev, "choices", None):
                continue
            delta = getattr(ev.choices[0], "delta", None)
            if delta is None:
                continue
            piece = getattr(delta, "content", None)
            if piece:
                print(piece, end="", flush=True)
                text_parts.append(piece)
                last_byte = time.time()
        print(f"\n[OK] smoke test completed in {time.time() - t0:.1f}s.", flush=True)
        full = "".join(text_parts)
        if not full.strip():
            print(
                "[WARN] response was empty. Endpoint reachable but model produced no content. Check model name / quota.",
                flush=True,
            )
            return 1
        return 0
    except Exception as e:
        print(f"\n[FAIL] {type(e).__name__}: {e}", flush=True)
        print("\nDiagnostics:", flush=True)
        print(f"  base_url = {NIM_BASE_URL}", flush=True)
        print(f"  model    = {NIM_MODEL}", flush=True)
        print(
            f"  api_key  = {('set (' + str(len(api_key)) + ' chars)' if api_key else 'MISSING')}",
            flush=True,
        )
        print("\nThings to check:", flush=True)
        print("  1. Is NVIDIA_API_KEY actually exported in this shell?", flush=True)
        print("     -> echo $NVIDIA_API_KEY", flush=True)
        print("  2. Can you reach the host at all?", flush=True)
        print("     -> curl -I https://integrate.api.nvidia.com/v1/models", flush=True)
        print("  3. Is the model name still correct on build.nvidia.com?", flush=True)
        return 1


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--datasets", nargs="+", default=ALL_DATASETS, choices=ALL_DATASETS + ["all"]
    )
    p.add_argument("--eval-n", type=int, default=EVAL_N_DEFAULT)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--results-root", default="results")
    p.add_argument("--api-key", default=os.environ.get("NVIDIA_API_KEY", ""))
    p.add_argument("--max-calls-per-dataset", type=int, default=None)
    p.add_argument("--no-cm", action="store_true")
    p.add_argument(
        "--concurrency",
        type=int,
        default=CONCURRENCY_DEFAULT,
        help="Workers in flight. Use 1 for debugging.",
    )
    p.add_argument(
        "--smoke-test",
        action="store_true",
        help="Skip dataset loading and just hit the API once. Use this to confirm your auth/network/model name is OK.",
    )
    args = p.parse_args()
    if args.smoke_test:
        sys.exit(smoke_test(args.api_key))
    if "all" in args.datasets:
        args.datasets = ALL_DATASETS
    random.seed(args.seed)
    np.random.seed(args.seed)
    client = _make_client(args.api_key)
    limiter = RateLimiter(RATE_LIMIT_RPM)
    out_root = Path(args.results_root)
    out_root.mkdir(parents=True, exist_ok=True)
    summary: List[Dict[str, Any]] = []
    for ds in args.datasets:
        try:
            m = benchmark_one_dataset(
                ds,
                eval_n=args.eval_n,
                seed=args.seed,
                client=client,
                limiter=limiter,
                out_root=out_root,
                max_calls_total=args.max_calls_per_dataset,
                plot_cm=not args.no_cm,
                concurrency=args.concurrency,
            )
            summary.append(m)
        except FileNotFoundError as e:
            print(f"[skip] {ds}: {e}", flush=True)
        except Exception as e:
            import traceback

            print(f"[error] {ds}: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
    if summary:
        print("\n" + "=" * 72, flush=True)
        print("SUMMARY", flush=True)
        print("=" * 72, flush=True)
        header = f"{'dataset':<14s}  {'n':>4s}/{'tot':<4s}  {'fail':>4s}  {'acc':>7s}  {'F1-mac':>7s}  {'F1-wgt':>7s}"
        print(header, flush=True)
        print("-" * len(header), flush=True)
        for m in summary:
            print(
                f"{m['dataset']:<14s}  {m['n_valid']:>4d}/{m['n_total']:<4d}  {m['parse_failures']:>4d}  {m['accuracy']:>6.2f}%  {m['f1_macro']:>6.2f}%  {m['f1_weighted']:>6.2f}%",
                flush=True,
            )
        with (out_root / "llm_benchmark_summary.json").open("w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n[saved] {out_root / 'llm_benchmark_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
