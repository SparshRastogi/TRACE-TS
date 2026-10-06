import os
import re
import json
import csv

DATASET_ROOTS = {
    "capture24": (
        "./results_chronos/capture24",
        "./results/capture24_Willetts2018_subset",
    ),
    "opportunity": ("./results_chronos/opportunity", "./results/opportunity"),
    "pamap2": ("./results_chronos/pamap2", "./results/pamap2"),
    "uschad": ("./results_chronos/uschad", "./results/uschad"),
    "ucihar": ("./results_chronos/ucihar", "./results/ucihar"),
    "mhealth": ("./results_chronos/mhealth", "./results/mhealth"),
    "shoaib": ("./results_chronos/shoaib", "./results/shoaib"),
}
LINE_RE = re.compile(
    "\\[final\\]\\s+test\\s+loss=([\\d.]+)\\s+acc=([\\d.]+)%\\s+F1-macro=([\\d.]+)%\\s+F1-weighted=([\\d.]+)%"
)
LINE_RE_EPOCH = re.compile(
    "\\[epoch=(\\d+)\\]\\s+test\\s+loss=([\\d.]+)\\s+acc=([\\d.]+)%\\s+F1-macro=([\\d.]+)%\\s+F1-weighted=([\\d.]+)%"
)


def load_chronos_summary(dirpath: str):
    p = os.path.join(dirpath, "summary.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def load_ad_metrics(dirpath: str):
    p = os.path.join(dirpath, "metrics.txt")
    if not os.path.exists(p):
        return None
    last = None
    with open(p) as f:
        for line in f:
            m = LINE_RE.search(line)
            if m:
                _, acc, fm, fw = m.groups()
                last = {
                    "test_acc": float(acc),
                    "test_fm": float(fm),
                    "test_fw": float(fw),
                }
                continue
            m2 = LINE_RE_EPOCH.search(line)
            if m2:
                _, _, acc, fm, fw = m2.groups()
                last = {
                    "test_acc": float(acc),
                    "test_fm": float(fm),
                    "test_fw": float(fw),
                }
    return last


def fmt(x, width=6, prec=2):
    if x is None:
        return "  —   "
    return f"{x:>{width}.{prec}f}"


def main():
    rows = []
    for ds, (cdir, adir) in DATASET_ROOTS.items():
        c = load_chronos_summary(cdir)
        a = load_ad_metrics(adir)
        rows.append(
            {
                "dataset": ds,
                "ad_acc": None if a is None else a["test_acc"],
                "ad_fm": None if a is None else a["test_fm"],
                "ad_fw": None if a is None else a["test_fw"],
                "chronos_acc": None if c is None else c["test_acc"],
                "chronos_fm": None if c is None else c["test_fm"],
                "chronos_fw": None if c is None else c["test_fw"],
            }
        )
    out_csv = "./results_chronos/comparison.csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[+] wrote {out_csv}")
    out_md = "./results_chronos/comparison.md"
    with open(out_md, "w") as f:
        f.write("# A&D vs Chronos-2 + MLP — test-set comparison\n\n")
        f.write("All numbers are percent. `—` = result not yet available.\n\n")
        f.write(
            "|              | A&D Acc | A&D F1-m | A&D F1-w | Chronos Acc | Chronos F1-m | Chronos F1-w |\n"
        )
        f.write(
            "|--------------|--------:|---------:|---------:|------------:|-------------:|-------------:|\n"
        )
        for r in rows:
            f.write(
                f"| {r['dataset']:<12s} | {fmt(r['ad_acc'])}  | {fmt(r['ad_fm'])}   | {fmt(r['ad_fw'])}   | {fmt(r['chronos_acc'])}    | {fmt(r['chronos_fm'])}     | {fmt(r['chronos_fw'])}     |\n"
            )
    print(f"[+] wrote {out_md}")
    print()
    print("=" * 92)
    print(
        f"{'dataset':<12s}  {'A&D Acc':>9s}  {'A&D F1-m':>9s}  {'A&D F1-w':>9s}    {'C2 Acc':>8s}  {'C2 F1-m':>8s}  {'C2 F1-w':>8s}"
    )
    print("-" * 92)
    for r in rows:
        print(
            f"{r['dataset']:<12s}  {fmt(r['ad_acc'], 9):>9s}  {fmt(r['ad_fm'], 9):>9s}  {fmt(r['ad_fw'], 9):>9s}    {fmt(r['chronos_acc'], 8):>8s}  {fmt(r['chronos_fm'], 8):>8s}  {fmt(r['chronos_fw'], 8):>8s}"
        )
    print("=" * 92)


if __name__ == "__main__":
    main()
