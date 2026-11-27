"""Build a reproducible, stratified YOLO subset of the D-Fire train/test release."""

import argparse
import csv
import hashlib
import io
import random
import tarfile
import zipfile
from collections import Counter
from contextlib import ExitStack
from pathlib import Path, PurePosixPath

from PIL import Image


CLASSES = ('smoke', 'fire')
SPLITS = ('train', 'val', 'test')
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.webp'}
FIELDS = ('split', 'source_split', 'filename', 'stratum', 'smoke_boxes', 'fire_boxes',
          'image_sha256', 'label_sha256', 'source_image')


def archive_entries(path, stack):
    if path.is_dir():
        return [(p.relative_to(path).as_posix(), p.read_bytes) for p in path.rglob('*') if p.is_file()]
    if zipfile.is_zipfile(path):
        archive = stack.enter_context(zipfile.ZipFile(path))
        return [(name, lambda name=name: archive.read(name)) for name in archive.namelist() if not name.endswith('/')]
    archive = stack.enter_context(tarfile.open(path, 'r:*'))
    return [(info.name, lambda info=info: archive.extractfile(info).read()) for info in archive if info.isfile()]


def read_index(paths, stack):
    index = {}
    for archive in paths:
        for name, read in archive_entries(archive, stack):
            path = PurePosixPath(name)
            if path.name.startswith('.') or len(path.parts) < 3 or path.parts[-3:-1] not in {
                ('train', 'images'), ('train', 'labels'), ('test', 'images'), ('test', 'labels'),
                ('val', 'images'), ('val', 'labels'),
            }:
                continue
            split, kind = path.parts[-3:-1]
            if (kind == 'images' and path.suffix.lower() not in IMAGE_SUFFIXES) or (
                kind == 'labels' and path.suffix.lower() != '.txt'
            ):
                continue
            pair = index.setdefault((split, path.stem), {})
            if kind in pair:
                raise ValueError(f'Duplicate archive entry for {split}/{kind}/{path.stem}')
            pair[kind] = (name, read)
    if not index or any(set(pair) != {'images', 'labels'} for pair in index.values()):
        missing = [key for key, pair in index.items() if len(pair) != 2]
        raise ValueError(f'Expected paired D-Fire split folders. Unpaired examples: {missing[:8]}')
    return index


def annotation_counts(label, name):
    counts = Counter()
    for line in label.decode('utf-8-sig').splitlines():
        if not line.strip():
            continue
        cls, *coords = line.split()
        if (cls not in {'0', '1'} or len(coords) != 4 or not all(0 <= float(v) <= 1 for v in coords)
                or any(float(v) == 0 for v in coords[2:])):
            raise ValueError(f'Invalid YOLO annotation in {name}: {line}')
        counts[CLASSES[int(cls)]] += 1
    return counts


def stratum(counts):
    classes = [cls for cls in CLASSES if counts[cls]]
    return '+'.join(classes) or 'background'


def select_pool(source_split, total, counts_by_key, rng):
    pools = {}
    for key in sorted(counts_by_key):
        if key[0] == source_split:
            pools.setdefault(stratum(counts_by_key[key]), []).append(key)
    population = sum(map(len, pools.values()))
    if not 0 < total < population:
        raise ValueError(f'Requested {total} samples from {source_split}, which has {population} images')
    quotas = {name: total * len(pool) / population for name, pool in pools.items()}
    sizes = {name: int(quota) for name, quota in quotas.items()}
    remainder = total - sum(sizes.values())
    for name in sorted(pools, key=lambda name: (sizes[name] - quotas[name], name))[:remainder]:
        sizes[name] += 1
    for pool in pools.values():
        rng.shuffle(pool)
    return [(pools[name], sizes[name]) for name in sorted(pools)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sources', type=Path, nargs='+', help='D-Fire folder or archive(s)')
    parser.add_argument('--output', type=Path, required=True, help='New YOLO dataset directory')
    parser.add_argument('--manifest', type=Path, required=True, help='Write or verify the split CSV')
    parser.add_argument('--replay', action='store_true', help='Rebuild from and verify the existing CSV')
    parser.add_argument('--seed', type=int, default=5770)
    parser.add_argument('--train', type=int, default=3300)
    parser.add_argument('--val', type=int, default=600)
    parser.add_argument('--test', type=int, default=600)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    seen = set()
    rng = random.Random(args.seed)

    with ExitStack() as stack:
        index = read_index(args.sources, stack)
        split_source = {'test': 'test', 'val': 'val' if any(k[0] == 'val' for k in index) else 'train',
                        'train': 'train'}

        def materialize(key, split, expected=None):
            image_name, read_image = index[key]['images']
            label_name, read_label = index[key]['labels']
            photo, label = read_image(), read_label()
            digest = hashlib.sha256(photo).hexdigest()
            if digest in seen:
                return False
            try:
                with Image.open(io.BytesIO(photo)) as image:
                    image.load()
                    if min(image.size) < 10:
                        return False
                    if image.format == 'JPEG' and not photo.endswith(b'\xff\xd9'):
                        return False  # Prevent Ultralytics from silently rewriting the JPEG on its first scan.
            except (ValueError, OSError):
                return False
            counts = annotation_counts(label, label_name)
            filename = PurePosixPath(image_name).name
            row = dict(zip(FIELDS, (split, key[0], filename, stratum(counts), str(counts['smoke']),
                                    str(counts['fire']), digest, hashlib.sha256(label).hexdigest(), image_name)))
            if expected and any(row[field] != expected[field] for field in FIELDS if field != 'source_image'):
                raise ValueError(f'Bytes or annotation changed for {key}')
            (args.output / split / 'images' / filename).write_bytes(photo)
            (args.output / split / 'labels' / f'{key[1]}.txt').write_bytes(label)
            rows.append(row)
            seen.add(digest)
            return True

        for split in SPLITS:
            for kind in ('images', 'labels'):
                (args.output / split / kind).mkdir(parents=True)

        if args.replay:
            with args.manifest.open(newline='', encoding='utf-8') as stream:
                for expected in csv.DictReader(stream):
                    split, source = expected['split'], expected['source_split']
                    if split not in SPLITS or source not in SPLITS:
                        raise ValueError(f'Invalid split in manifest: {split}/{source}')
                    key = (source, Path(expected['filename']).stem)
                    if not materialize(key, expected['split'], expected):
                        raise ValueError(f'Could not reproduce {key}')
        else:
            counts = {}
            invalid = Counter()
            for key, pair in index.items():
                name, read = pair['labels']
                try:
                    counts[key] = annotation_counts(read(), name)
                except (UnicodeError, ValueError):
                    invalid[key[0]] += 1
            print(f'Excluded source images with invalid YOLO labels: {dict(sorted(invalid.items()))}')
            pools = {}
            for source in dict.fromkeys(split_source.values()):
                total = sum(getattr(args, split) for split in SPLITS if split_source[split] == source)
                pools[source] = select_pool(source, total, counts, rng)
            # Allocate test first, then validation, then training. One iterator per source stratum
            # means validation drawn from train can never consume the same filename as training.
            for source, strata in pools.items():
                targets = [split for split in reversed(SPLITS) if split_source[split] == source]
                if len(targets) == 1:
                    assignments = [(targets[0], [quota for _, quota in strata])]
                else:
                    val_share = args.val / (args.train + args.val)
                    quotas = [quota * val_share for _, quota in strata]
                    val_sizes = [int(quota) for quota in quotas]
                    order = sorted(range(len(quotas)), key=lambda i: (val_sizes[i] - quotas[i], i))
                    for i in order[:args.val - sum(val_sizes)]:
                        val_sizes[i] += 1
                    assignments = [('val', val_sizes), ('train', [quota - val for (_, quota), val in zip(strata, val_sizes)])]
                iterators = [iter(pool) for pool, _ in strata]
                for split, quotas in assignments:
                    for population, quota in zip(iterators, quotas):
                        accepted = 0
                        while accepted < quota:
                            key = next(population, None)
                            if key is None:
                                raise ValueError(f'Not enough unique, readable samples to fill {split}')
                            accepted += materialize(key, split)

            args.manifest.parent.mkdir(parents=True, exist_ok=True)
            with args.manifest.open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(sorted(rows, key=lambda row: (row['split'], row['filename'])))

    dataset_yaml = 'train: train/images\nval: val/images\ntest: test/images\nnames: {0: smoke, 1: fire}\n'
    (args.output / 'data.yaml').write_text(dataset_yaml, encoding='utf-8')
    print(dict(sorted(Counter((row['split'], row['stratum']) for row in rows).items())))


if __name__ == '__main__':
    main()
