"""Preview or remove generated data and output files."""
import argparse
import shutil
from pathlib import Path


def clean(apply=False, root=None):
    root = Path(root or Path(__file__).resolve().parents[1]).resolve()
    # Resolve and check every target before any deletion, including nested links/junctions.
    keep_empty = [root / 'data', root / 'output']
    targets = []
    for folder in keep_empty:
        if folder.exists():
            if folder.is_symlink() or folder.resolve() != folder:
                raise ValueError('拒绝清理链接目录: ' + str(folder))
            targets.extend(folder.iterdir())
    targets.extend(root / p for p in ('__pycache__', '.pytest_cache') if (root / p).exists())
    for folder in (root / 'adcr', root / 'scripts', root / 'tests'):
        if folder.exists():
            targets.extend(folder.rglob('__pycache__'))
    for target in targets:
        resolved = target.resolve()
        resolved.relative_to(root)
        if target.is_symlink() or resolved != target:
            raise ValueError('拒绝清理链接: ' + str(target))
        if target.is_dir():
            for child in target.rglob('*'):
                child.resolve().relative_to(resolved)
                if child.is_symlink():
                    raise ValueError('拒绝清理含链接的目录: ' + str(target))
    for target in targets:
        print(('DELETE ' if apply else 'WOULD DELETE ') + str(target))
        if apply:
            if target.is_dir(): shutil.rmtree(target)
            else: target.unlink()
    if apply:
        for folder in keep_empty: folder.mkdir(exist_ok=True)
    return targets


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true', help='执行清理；默认仅预览')
    clean(parser.parse_args().apply)
