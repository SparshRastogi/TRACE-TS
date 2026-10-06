import os
import sys
import urllib.request
import shutil

DATASET_URL = "https://www.utwente.nl/en/eemcs/ps/dataset-folder/sensors-activity-recognition-dataset-shoaib.rar"
OUTPUT_DIR = "shoaib_dataset"
RAR_FILENAME = "sensors-activity-recognition-dataset-shoaib.rar"


def show_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        bar_len = 40
        filled = int(bar_len * pct / 100)
        bar = "█" * filled + "░" * (bar_len - filled)
        mb_done = downloaded / 1048576
        mb_total = total_size / 1048576
        print(
            f"\r  [{bar}] {pct:5.1f}%  {mb_done:.1f}/{mb_total:.1f} MB",
            end="",
            flush=True,
        )
    else:
        mb_done = downloaded / 1048576
        print(f"\r  Downloaded {mb_done:.1f} MB...", end="", flush=True)


def download_dataset():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rar_path = os.path.join(OUTPUT_DIR, RAR_FILENAME)
    if os.path.exists(rar_path):
        print(f"Archive already exists at '{rar_path}'. Skipping download.")
    else:
        print("Downloading Shoaib Sensors Activity Recognition Dataset...")
        print(f"  Source : {DATASET_URL}")
        print(f"  Target : {rar_path}")
        try:
            urllib.request.urlretrieve(DATASET_URL, rar_path, show_progress)
            print()
            size_mb = os.path.getsize(rar_path) / 1048576
            print(f"  Download complete. File size: {size_mb:.1f} MB")
        except Exception as exc:
            print(f"\nERROR: Download failed — {exc}")
            sys.exit(1)
    extract_dir = os.path.join(OUTPUT_DIR, "extracted")
    if os.path.exists(extract_dir):
        print(
            f"\nExtracted folder already exists at '{extract_dir}'. Skipping extraction."
        )
    else:
        print("\nExtracting archive...")
        extracted = False
        try:
            import patoollib

            patoollib.extract_archive(rar_path, outdir=extract_dir)
            extracted = True
            print("  Extraction complete (via patool).")
        except ImportError:
            pass
        if not extracted:
            try:
                import rarfile

                with rarfile.RarFile(rar_path) as rf:
                    rf.extractall(extract_dir)
                extracted = True
                print("  Extraction complete (via rarfile).")
            except ImportError:
                pass
        if not extracted:
            for cmd in (
                ["unrar", "x", rar_path, extract_dir + "/"],
                ["7z", "x", rar_path, f"-o{extract_dir}"],
            ):
                if shutil.which(cmd[0]):
                    import subprocess

                    result = subprocess.run(cmd)
                    if result.returncode == 0:
                        extracted = True
                        print(f"  Extraction complete (via {cmd[0]}).")
                        break
        if not extracted:
            print(
                f"\nWARNING: Could not extract the .rar archive automatically.\n  Please install one of the following and re-run:\n    pip install patool rarfile\n  or install 'unrar' / '7-Zip' on your system.\n  The raw archive is saved at: {rar_path}"
            )
            return
    print("\n" + "=" * 60)
    print("Dataset ready!")
    print(f"  Archive   : {rar_path}")
    print(f"  Extracted : {extract_dir}")
    print()
    print("Citation:")
    print("  Shoaib, M., Bosch, S., Incel, O.D., Scholten, H., &")
    print("  Havinga, P.J.M. (2014). Fusion of Smartphone Motion Sensors")
    print("  for Physical Activity Recognition. Sensors, 14(6), 10146-10176.")
    print("=" * 60)


if __name__ == "__main__":
    download_dataset()
