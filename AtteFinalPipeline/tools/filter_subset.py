import argparse
import shutil
from pathlib import Path


def filter_subset(reference, source, destination):
    reference, source, destination = map(Path, (reference, source, destination))
    expected = {path.name for path in reference.glob("*_class*_s*.json")}
    available = {path.name for path in source.glob("*_class*_s*.json")}
    matched = expected & available
    destination.mkdir(parents=True, exist_ok=True)
    for name in sorted(matched):
        src, dst = (source / name, destination / name)
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
    return (len(matched), sorted(expected - available))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args()
    for path in (args.reference, args.source):
        if not path.is_dir():
            parser.error(f"Directory does not exist: {path}")
    matched, missing = filter_subset(args.reference, args.source, args.destination)
    print(f"Copied {matched} samples; {len(missing)} missing.")
    for name in missing:
        print(name)


if __name__ == "__main__":
    main()
