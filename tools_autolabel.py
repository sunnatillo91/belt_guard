"""Kesmani avtomatik topish: yorug' + ingichka + cho'zilgan soha.
Chang dog'lari yumaloq va tarqoq, kesma esa uzun va tor — shu farq bo'yicha ajratamiz.
"""
import cv2
import numpy as np


def find_cut(frame):
    """Kadrdan kesma maskalarini qaytaradi (ro'yxat)."""
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (5, 5), 0)

    # Lenta qorong'i; kesma undan ancha yorug'. Mahalliy kontrastdan foydalanamiz.
    bg = cv2.medianBlur(g, 61)                    # fon (lenta + chang)
    diff = cv2.subtract(g, bg)                    # fondan yorug' joylar
    _, binr = cv2.threshold(diff, 28, 255, cv2.THRESH_BINARY)
    binr = cv2.morphologyEx(binr, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    n, lab, stats, _ = cv2.connectedComponentsWithStats(binr, 8)
    H, W = g.shape
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 150:                            # juda mayda
            continue
        # kadr chetidagi pol/ramka polosalari kesma emas
        if x <= 2 or y <= 2 or x + w >= W - 2 or y + h >= H - 2:
            continue
        cx = x + w / 2
        if not (0.10 * W < cx < 0.90 * W):        # lenta markazda
            continue
        m = (lab == i).astype(np.uint8)
        pts = cv2.findNonZero(m)
        (_, _), (rw, rh), _ = cv2.minAreaRect(pts)
        long_s, short_s = max(rw, rh), min(rw, rh)
        if short_s < 1:
            continue
        elong = long_s / short_s
        fill = area / max(rw * rh, 1)             # to'rtburchakni qanchalik to'ldirgan
        # kesma: cho'zilgan (>4x), uzun (>40px) va ingichka
        if elong >= 4.0 and long_s >= 40 and short_s <= 0.06 * W and fill > 0.25:
            out.append(m)
    return out


def mask_to_polygon(mask):
    """YOLO-seg formatidagi normallashtirilgan poligon."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    eps = 0.004 * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
    if len(approx) < 3:
        return None
    H, W = mask.shape
    return [(x / W, y / H) for x, y in approx]


if __name__ == "__main__":
    import sys
    out_dir = r'C:\Users\Shaxi\AppData\Local\Temp\claude\g--belt-damage\5e9cac40-f5aa-467c-ac58-c267b1bec62c\scratchpad'
    cap = cv2.VideoCapture(r'g:\belt_damage\Kichik_konveyer_damaged.MOV')
    shown = 0
    i = 0
    hits = 0
    total = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % 15 == 0:
            total += 1
            masks = find_cut(f)
            if masks:
                hits += 1
            if masks and shown < 4 and i > 200:
                vis = f.copy()
                for m in masks:
                    vis[m == 1] = (0, 0, 255)
                    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(vis, cnts, -1, (0, 255, 255), 2)
                cv2.imwrite(f'{out_dir}/auto_{shown}.jpg', vis)
                shown += 1
        i += 1
    cap.release()
    print(f'Tekshirilgan kadr: {total} | kesma topilgan: {hits} ({hits/max(total,1)*100:.0f}%)')
