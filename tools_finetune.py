# Stend + ochiq dataset ustida fine-tune
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

PROJECT = Path(r'g:\belt_damage')
DATA = str(PROJECT / 'ds_merged' / 'data.yaml')


def main():
    assert torch.cuda.is_available(), 'CUDA yo`q'
    print('GPU:', torch.cuda.get_device_name(0), flush=True)

    model = YOLO(str(PROJECT / 'models' / 'best.pt'))   # mavjud nano modeldan davom
    model.train(
        data=DATA,
        epochs=35,
        imgsz=640,
        batch=8,
        device=0,
        workers=0,
        patience=12,
        optimizer='AdamW',
        lr0=0.0005,          # fine-tune: past lr
        cos_lr=True,
        mosaic=0.4,
        copy_paste=0.3,
        degrees=5,
        flipud=0.0,
        fliplr=0.5,
        hsv_v=0.5,
        hsv_s=0.4,
        translate=0.2,
        scale=0.4,
        erasing=0.3,
        project=str(PROJECT / 'runs'),
        name='v2_stand',
        exist_ok=True,
    )

    best = PROJECT / 'runs' / 'v2_stand' / 'weights' / 'best.pt'
    m = YOLO(str(best))
    r = m.val(data=DATA)
    print('\n=== YAKUNIY METRIKALAR (v2, stend bilan) ===')
    print(f'Segment mAP@50    : {r.seg.map50:.3f}')
    print(f'Segment mAP@50-95 : {r.seg.map:.3f}')
    for idx, name in r.names.items():
        if idx < len(r.seg.maps):
            print(f'  {name:20s}: {r.seg.maps[idx]:.3f}')

    shutil.copy(best, PROJECT / 'models' / 'best_stand.pt')
    print('\nmodels/best_stand.pt saqlandi')


if __name__ == '__main__':
    main()
