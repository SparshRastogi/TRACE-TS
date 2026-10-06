import os
import json
import time
import threading
import subprocess

try:
    import psutil as _psutil

    _PSUTIL_OK = True
except ImportError:
    _PSUTIL_OK = False


def _query_nvidia_smi():
    try:
        out = (
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                timeout=3,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
            .split("\n")[0]
        )
        parts = [p.strip() for p in out.split(",")]
        return (float(parts[0]), float(parts[1]), float(parts[2]))
    except Exception:
        return (None, None, None)


class PerformanceMonitor:
    def __init__(self, output_dir, poll_interval=5):
        self.poll_interval = poll_interval
        self._log_path = os.path.join(output_dir, "perf_log.jsonl")
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._samples_per_sec = 0.0
        self._sample_ig_ms = []
        self._sample_shap_ms = []
        self._last_ig_ms = 0.0
        self._last_shap_ms = 0.0
        self._fh = open(self._log_path, "a", buffering=1)
        self._write(
            {
                "type": "run_start",
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "psutil_available": _PSUTIL_OK,
            }
        )
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def _write(self, record):
        line = json.dumps(record) + "\n"
        with self._lock:
            self._fh.write(line)

    def _poll_loop(self):
        while not self._stop.wait(self.poll_interval):
            gpu_util, gpu_mem_used, gpu_mem_total = _query_nvidia_smi()
            cpu_pct = _psutil.cpu_percent(interval=None) if _PSUTIL_OK else None
            if _PSUTIL_OK:
                vm = _psutil.virtual_memory()
                ram_used = round(vm.used / 1000000000.0, 2)
                ram_total = round(vm.total / 1000000000.0, 2)
            else:
                ram_used = ram_total = None
            self._write(
                {
                    "type": "poll",
                    "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "gpu_util_pct": gpu_util,
                    "gpu_mem_used_mb": gpu_mem_used,
                    "gpu_mem_total_mb": gpu_mem_total,
                    "cpu_util_pct": cpu_pct,
                    "ram_used_gb": ram_used,
                    "ram_total_gb": ram_total,
                    "samples_per_sec": round(self._samples_per_sec, 3),
                }
            )

    def update_rate(self, samples_per_sec):
        with self._lock:
            self._samples_per_sec = samples_per_sec

    class _Timer:
        __slots__ = ("_mon", "_attr", "_t0")

        def __init__(self, mon, attr):
            self._mon = mon
            self._attr = attr

        def __enter__(self):
            self._t0 = time.perf_counter()
            return self

        def __exit__(self, *_):
            setattr(self._mon, self._attr, (time.perf_counter() - self._t0) * 1000.0)

    def time_ig(self):
        self._last_ig_ms = 0.0
        return self._Timer(self, "_last_ig_ms")

    def time_shap(self):
        self._last_shap_ms = 0.0
        return self._Timer(self, "_last_shap_ms")

    def record_sample(self, save_queue_depth=0):
        with self._lock:
            self._sample_ig_ms.append(self._last_ig_ms)
            self._sample_shap_ms.append(self._last_shap_ms)
            n = len(self._sample_ig_ms)
            window = self._sample_ig_ms[-50:]
            shap_w = self._sample_shap_ms[-50:]
            if len(window) > 1:
                total_sec = sum((a + b for a, b in zip(window, shap_w))) / 1000.0
                self._samples_per_sec = len(window) / max(total_sec, 1e-09)
        self._write(
            {
                "type": "sample",
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "ig_ms": round(self._last_ig_ms, 2),
                "shap_ms": round(self._last_shap_ms, 2),
                "save_queue_depth": int(save_queue_depth),
            }
        )

    def start(self):
        pass

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=self.poll_interval + 2)
        self._write({"type": "run_end", "ts": time.strftime("%Y-%m-%d %H:%M:%S")})
        with self._lock:
            self._fh.close()

    def print_summary(self):
        with self._lock:
            ig_ms = sorted(self._sample_ig_ms)
            shap_ms = sorted(self._sample_shap_ms)
        if not ig_ms:
            print("  PerformanceMonitor: no samples timed.")
            return
        n = len(ig_ms)
        p = lambda lst, pct: lst[max(0, int(len(lst) * pct) - 1)]
        total_ms = sorted((a + b for a, b in zip(ig_ms, shap_ms)))
        print(f"\n  {'─' * 58}")
        print(f"  Performance summary  ({n} samples instrumented)")
        print(f"  {'─' * 58}")
        for label, lst in [
            ("IG (ms)", ig_ms),
            ("SHAP (ms)", shap_ms),
            ("IG+SHAP (ms)", total_ms),
        ]:
            mean = sum(lst) / len(lst)
            print(
                f"  {label:<30s} {mean:>7.1f} {p(lst, 0.5):>7.1f} {p(lst, 0.95):>7.1f}"
            )
        print(f"  {'─' * 58}")
        print(f"  Full log → {self._log_path}")

    @property
    def log_path(self):
        return self._log_path
