"""Stend videolaridan YOLO-seg dataset yig'ish.

- damaged video: avtomatik topilgan kesma -> 'Tear' yorlig'i
- clean video   : nuqson yo'q kadrlar -> negativ (bo'sh yorliq fayli)
- ikkilanarli kadrlar (kesma kadr chetiga tegib turgan) umuman olinmaydi
"""
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from autolabel import find_cut, mask_to_polygon

SRC = Path(r'g:\belt_damage\ds_public')
DST = Path(r'g:\belt_damage\ds_merged')
TEAR_ID = 5                      # data.yaml: ['Hole','Human','Other Objects','Puncture','Roller','Tear',...]

DAMAGED = r'g:\belt_damage\Kichik_konveyer_damaged.MOV'
CLEAN = r'g:\belt_damage\Kichik_konveyer.MOV'


def border_candidate(frame):
    """Kadr chetiga tegib turgan kesmaga o'xshash soha bormi (ikkilanarli kadr)."""
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (5, 5), 0)
    diff = cv2.subtract(g, cv2.medianBlur(g, 61))
    _, b = cv2.threshold(diff, 28, 255, cv2.THRESH_BINARY)
    b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(b, 8)
    H, W = g.shape
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 400:
            continue
        touches = x <= 2 or y <= 2 or x + w >= W - 2 or y + h >= H - 2
        cx = x + w / 2
        if touches and 0.10 * W < cx < 0.90 * W and max(w, h) >= 40:
            return True
    return False


def prepare_dirs():
    if DST.exists():
        shutil.rmtree(DST)
    for split in ('train', 'valid'):
        (DST / split / 'images').mkdir(parents=True)
        (DST / split / 'labels').mkdir(parents=True)


def copy_public():
    n = 0
    for split in ('train', 'valid'):
        for img in (SRC / split / 'images').glob('*.jpg'):
            lab = SRC / split / 'labels' / (img.stem + '.txt')
            shutil.copy(img, DST / split / 'images' / img.name)
            if lab.exists():
                shutil.copy(lab, DST / split / 'labels' / lab.name)
            n += 1
    print(f'Ochiq dataset: {n} rasm ko`chirildi')


def harvest(video, tag, step, positive, val_every=5):
    cap = cv2.VideoCapture(video)
    i = kept = skipped = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % step == 0:
            masks = find_cut(f) if positive else []
            if positive:
                if not masks:
                    skipped += 1
                    i += 1
                    continue
            else:
                # toza video: kesmaga o'xshash narsa bo'lsa umuman olmaymiz
                if masks or border_candidate(f):
                    skipped += 1
                    i += 1
                    continue

            lines = []
            for m in masks:
                poly = mask_to_polygon(m)
                if poly:
                    flat = ' '.join(f'{x:.6f} {y:.6f}' for x, y in poly)
                    lines.append(f'{TEAR_ID} {flat}')
            if positive and not lines:
                skipped += 1
                i += 1
                continue

            split = 'valid' if (kept % val_every == 0) else 'train'
            name = f'{tag}_{i:06d}'
            cv2.imwrite(str(DST / split / 'images' / f'{name}.jpg'), f,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            (DST / split / 'labels' / f'{name}.txt').write_text('\n'.join(lines))
            kept += 1
        i += 1
    cap.release()
    print(f'{tag}: {kept} kadr olindi, {skipped} tashlandi')
    return kept


if __name__ == '__main__':
    prepare_dirs()
    copy_public()
    p = harvest(DAMAGED, 'stand_dmg', step=6, positive=True)
    n = harvest(CLEAN, 'stand_clean', step=12, positive=False)

    yaml = f"""train: {(DST / 'train' / 'images').as_posix()}
val: {(DST / 'valid' / 'images').as_posix()}

nc: 8
names: ['Hole', 'Human', 'Other Objects', 'Puncture', 'Roller', 'Tear', 'impact damage', 'patch work']
"""
    (DST / 'data.yaml').write_text(yaml, encoding='utf-8')
    print(f'\nTayyor: {DST}\nStend: {p} nuqsonli + {n} toza kadr')
