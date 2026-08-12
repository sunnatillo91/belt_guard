# 🛡️ BeltGuard — konveyer lentasidagi shikastni real vaqtda aniqlash

Konchilik konveyer lentasidagi nuqsonlarni deep learning yordamida topadigan va ularning
**millimetrdagi real o'lchamini** hisoblab, xavflilik darajasini beradigan tizim.

Javob "nuqson bor" emas — **"182 mm uzunlikdagi yirtiq, KRITIK, lentani to'xtating"**.

## Nima ishlaydi

| Bosqich | Izoh |
|---|---|
| Kadr olish | Kamera oqimi, telefon kamerasi (IP Webcam) yoki video fayl |
| Segmentatsiya | YOLO11-seg har nuqson uchun piksel maskasi qaytaradi |
| Geometriya | `minAreaRect` orqali nuqsonning uzunlik/kengligi |
| Kalibratsiya | Lenta kengligi orqali piksel → millimetr |
| Qaror | Klass + o'lcham → xavflilik darajasi, alert, CSV jurnal |

**Nega ramka emas, segmentatsiya:** yirtiq — uzun, ingichka va egri chiziq. To'g'ri to'rtburchak
ramkaga o'ralganda maydonning katta qismi sog'lom lentaga tegishli bo'ladi va o'lcham buziladi.
Maska esa aniq konturni beradi — qiya yirtiqda ham o'lcham to'g'ri chiqadi.

## Natijalar

`yolo11n-seg`, 40 epoch, 640px, GTX 1650 · valid to'plami (185 rasm, 493 obyekt):

| Metrika | Qiymat |
|---|---|
| Segment mAP@50 | **0.729** |
| Segment mAP@50-95 | 0.367 |
| Box mAP@50 | 0.789 |
| Precision / Recall (box) | 0.826 / 0.786 |

## Ishga tushirish

```bash
pip install streamlit ultralytics opencv-python pandas
streamlit run beltguard_dashboard.py
```

Brauzerda: http://localhost:8501

Yon paneldagi **Manba** maydoni:

| Qiymat | Nima |
|---|---|
| `0` | Noutbuk veb-kamerasi |
| `test_video.mp4` | Video fayl |
| `http://192.168.1.50:8080/video` | Telefon kamerasi (Android · IP Webcam ilovasi) |

Mavzu (light/dark): yuqori o'ng burchakdagi ☰ menyu → Settings → Appearance.

## Modelni o'rgatish

- **Colab (GPU bilan tez):** [`BeltGuard_training.ipynb`](BeltGuard_training.ipynb)
- **Lokal:** [`BeltGuard_training_local.ipynb`](BeltGuard_training_local.ipynb)

Dataset avtomatik yuklanadi — Roboflow API kaliti kerak (bepul akkaunt yetadi).
Repoda dataset yo'q, u notebook orqali qayta yuklab olinadi.

## Xavflilik mantiqi

Xavflilik ikki manbadan yig'iladi — nuqson **turi** (bazaviy daraja) va **o'lchami** (kuchaytirish):

| Daraja | Nima kiradi |
|---|---|
| 👁 KUZATUV | Yuza tirnalish, yeyilish |
| ⚠️ JIDDIY | Zarbadan shikast yoki 80 mm dan oshgan yuza nuqson |
| ⛔ KRITIK | Yirtiq, teshik, teshilish yoki 150 mm dan oshgan jiddiy nuqson |

`Roller`, `Human` kabi klasslar alert bermaydi — ular nuqson emas, sahnaning bir qismi.
Har bir klass uchun alert oralig'i qo'yilgan: bitta yirtiq har kadrda qayta signal bermaydi.

## Dataset

[conveyor-belt-damage](https://universe.roboflow.com/sample-wy2mp/conveyor-belt-damage) —
922 rasm (645 / 185 / 92), 8 klass, CC BY 4.0.

Klasslar: `Tear`, `Hole`, `Puncture`, `impact damage`, `patch work`, `Roller`, `Human`, `Other Objects`.

## Loyiha tarkibi

```
beltguard_dashboard.py           Streamlit dashboard (asosiy dastur)
BeltGuard_training.ipynb         Colab training notebook
BeltGuard_training_local.ipynb   Lokal training notebook
models/best.pt                   O'rgatilgan model
.streamlit/config.toml           Light/dark mavzu sozlamalari
hakamlar_uchun.html              Texnik qismning qisqacha bayoni
BeltGuard_reja.md                To'liq loyiha rejasi va tadqiqot
```

## Rejadagi keyingi qadamlar

- Nuqsonni lenta koordinatasiga bog'lash (optik oqim orqali "lenta xaritasi") — har aylanishda
  qayta o'lchab, o'sish tezligini ko'rsatish
- Anomaliya qatlami — datasetda uchramagan yangi turdagi nuqsonni topish
- Edge qurilmada (Jetson) INT8 optimizatsiya va tezlik o'lchovi
