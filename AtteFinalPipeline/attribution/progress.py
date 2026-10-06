import os
import time
from AtteFinalPipeline.serialization import _json_dumps


class ProgressTracker:
    def __init__(self, output_dir, total_samples_by_split, worker_id=0):
        suffix = f"_worker{worker_id}" if worker_id > 0 else ""
        self.path = os.path.join(output_dir, f"progress{suffix}.json")
        self.start_time = time.time()
        self.total_all = sum(total_samples_by_split.values())
        self.state = {
            "status": "running",
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "splits": {},
            "overall": {
                "total_samples": self.total_all,
                "processed": 0,
                "correct_saved": 0,
                "incorrect_skipped": 0,
                "elapsed_seconds": 0,
                "eta_seconds": None,
                "samples_per_second": 0.0,
            },
        }
        for split, n in total_samples_by_split.items():
            self.state["splits"][split] = {
                "total_samples": n,
                "processed": 0,
                "correct_saved": 0,
                "incorrect_skipped": 0,
                "status": "pending",
            }
        self._write()

    def update(self, split, correct):
        sp = self.state["splits"][split]
        sp["processed"] += 1
        sp["status"] = "running"
        if correct:
            sp["correct_saved"] += 1
        else:
            sp["incorrect_skipped"] += 1
        ov = self.state["overall"]
        ov["processed"] += 1
        if correct:
            ov["correct_saved"] += 1
        else:
            ov["incorrect_skipped"] += 1
        elapsed = time.time() - self.start_time
        ov["elapsed_seconds"] = round(elapsed, 1)
        rate = ov["processed"] / elapsed if elapsed > 0 else 0
        ov["samples_per_second"] = round(rate, 2)
        remaining = self.total_all - ov["processed"]
        ov["eta_seconds"] = round(remaining / rate, 1) if rate > 0 else None
        if sp["processed"] >= sp["total_samples"]:
            sp["status"] = "done"

    def finish_split(self, split):
        self.state["splits"][split]["status"] = "done"
        self._write()

    def finish(self):
        elapsed = time.time() - self.start_time
        self.state["overall"]["elapsed_seconds"] = round(elapsed, 1)
        self.state["overall"]["eta_seconds"] = 0
        self.state["status"] = "completed"
        self.state["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self._write()

    def write_periodic(self, every_n=50):
        if self.state["overall"]["processed"] % every_n == 0:
            self._write()

    def _write(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            f.write(_json_dumps(self.state))
        os.replace(tmp, self.path)
