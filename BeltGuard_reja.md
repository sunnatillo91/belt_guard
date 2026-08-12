# BeltGuard AI — Konchilik konveyer lentasidagi shikastlarni deep learning bilan aniqlash

**Hackathon rejasi | 48–72 soat | Colab/Kaggle GPU | Dataset yo'q holatidan boshlab**

---

## 0. Eslatma: siz bergan YouTube havolasi

`youtube.com/watch?v=uT9abmjAiss` — YouTube bu sessiyadan so'rovni bloklab qo'ydi (HTTP 429), shuning uchun videoning aynan mazmunini o'qiy olmadim. Agar video sarlavhasini yoki 3–4 jumlalik mazmunini yozib bersangiz, rejani o'sha yondashuvga moslashtiraman. Quyidagi reja ilmiy maqolalar, ochiq datasetlar va sanoat yechimlari asosida mustaqil tuzilgan.

---

## 1. Muammo va biznes-keys (pitch uchun)

Konveyer lentasi konchilikdagi eng qimmat sarf materiallaridan biri va to'xtash sabablarining asosiy manbai:

- Rejasiz to'xtash **soatiga $10 000 – $50 000** turadi (yuk oqimi va material qiymatiga qarab). Temir rudasi terminalida 4 soatlik to'xtash ≈ $100 000 zarar.
- Bitta ochiq usulda qazib olish konida prediktiv monitoring joriy qilingach, rolik bilan bog'liq yillik xarajat **$1.34 mln → $39 600** ga tushgan, rejasiz to'xtashlar **40%+** kamaygan.
- Uzunasiga yirtilish (longitudinal tear) eng xavflisi: tiqilib qolgan metall bo'lak lentani bir necha o'nlab metrga "pichoqday" kesib ketadi va bu **soniyalar** ichida sodir bo'ladi.

**Bizning gipoteza:** arzon kamera + edge qurilma + deep learning = lentani har aylanishida to'liq skanerlash, shikastni millimetr aniqligida o'lchash va o'sish tendensiyasini kuzatish. Bu inson tekshiruvidan 100× tez va uzluksiz.

---

## 2. Research xulosasi — hozirgi SOTA yondashuvlar

| Yondashuv | Manba / natija | Bizga foydasi |
|---|---|---|
| **YOLOv5 + BotNet attention + Shape-IoU (YOLO-STOD)** | 3 100 rasm, 1 klass (uzunasiga yirtilish), **mAP 91.9%**, **191 FPS** (RTX 3060Ti) | Attention bloki kichik/ingichka nuqsonlarda sezilarli yordam beradi. Kichik dataset + augmentatsiya yetarli ekanini isbotlaydi |
| **FDEP-YOLOv8** (Focal Modulation + DySample + EMA + PIoU v2) | 498 → 1 992 rasm augmentatsiyadan keyin, 4 klass, **mAP@50 93.2%**, ~3.2M parametr, edge uchun | Bizning aynan holatimiz: juda kam data + edge deployment. Klasslar bo'linishini shundan ko'chiramiz |
| **YOLOv8-LDH + multispektral tasvir** | Multispektral kamera bilan yengil model | Ideal, lekin hackathonda kamera yo'q — "kelajak roadmap" sifatida aytamiz |
| **Chuqurlik (depth) sensori — TrueDepth/3D point cloud** | **F1 > 0.97**, RGB dan **+0.05 F1** yuqori, o'lcham xatosi < 3 mm | Chang va yorug'lik o'zgarishiga chidamli. iPhone (Face ID) yoki RealSense bo'lsa — kuchli qo'shimcha |
| **Lazer chiziq / line-scan kamera** | Klassik sanoat usuli — lenta ostidan lazer nurini kuzatish | Ko'rish tizimimiz bilan birlashtirilsa (sensor fusion) hakamlarga jiddiy ko'rinadi |
| **Unsupervised anomaly detection (PatchCore, PaDiM, EfficientAD)** | Faqat "sog'lom" rasmlar bilan o'qitiladi | **Bizning maxfiy quroli.** Lenta — takrorlanuvchi tekstura, shuning uchun anomaliya modeli oz data bilan ham ishlaydi va *hech qachon ko'rilmagan* nuqson turini topadi |
| **Diffusion / copy-paste sintetik nuqson generatsiyasi** | Few-shot sanoat inspeksiyasida mAP ni sezilarli oshiradi | Datasetimizni 10× kengaytirish usuli |

**Asosiy xulosa:** hech kim faqat bitta modeldan foydalanmaydi. G'olib yechim — *detektsiya + anomaliya + kuzatuv (tracking) + lenta xaritasi* kombinatsiyasi.

---

## 3. Tavsiya etilgan arxitektura — "BeltGuard"

```
┌──────────────┐
│  Kamera oqimi │  RGB 1080p / 30fps  (+ ixtiyoriy: depth, IR)
└──────┬───────┘
       ▼
┌────────────────────────────────────────┐
│ 1. PREPROCESSING                       │
│  • ROI kesish (faqat lenta yuzasi)     │
│  • Perspektiv to'g'rilash (homography) │
│  • mm/piksel kalibratsiya              │
│  • CLAHE — chang/qorong'ilikka qarshi  │
└──────┬─────────────────────────────────┘
       ▼
┌───────────────────────┬────────────────────────┐
│ 2A. DETEKTOR          │ 2B. ANOMALIYA MODELI   │
│ YOLO26-s / YOLO11-seg │ PatchCore (anomalib)   │
│ segmentatsiya         │ faqat sog'lom lentada  │
│ 6 klass               │ o'qitilgan             │
│ har kadrda            │ shubhali ROI larda     │
└───────────┬───────────┴─────────┬──────────────┘
            ▼                     ▼
      ┌───────────────────────────────────┐
      │ 3. FUSION + TRACKING              │
      │  • ByteTrack — takroriy alertni   │
      │    yo'q qilish                    │
      │  • Optical flow → lenta tezligi   │
      │  • Nuqsonni lenta koordinatasiga  │
      │    bog'lash (splice dan X metr)   │
      └───────────┬───────────────────────┘
                  ▼
      ┌───────────────────────────────────┐
      │ 4. SEVERITY ENGINE                │
      │  o'lcham(mm) × klass × o'sish     │
      │  → YASHIL / SARIQ / QIZIL         │
      │  QIZIL = avto-STOP signali (PLC)  │
      └───────────┬───────────────────────┘
                  ▼
      ┌───────────────────────────────────┐
      │ 5. DASHBOARD + DB                 │
      │  Jonli video, lenta xaritasi,     │
      │  nuqson tarixi, o'sish grafigi,   │
      │  Telegram/email alert             │
      └───────────────────────────────────┘
```

### Nima uchun bu g'oliblik keltiradi

Ko'pchilik jamoa "YOLO o'rgatdik, bounding box chizdik" bilan cheklanadi. Sizni ajratib turadigan 4 ta narsa:

1. **Virtual line-scan / "lentani yoyish"** — optical flow bilan lenta tezligini o'lchab, kadrlarni bir-biriga yopishtirib, butun lentaning uzun panoramasini quramiz. Endi nuqsonning **lentadagi aniq joyi** (splice dan 47.3 m) ma'lum bo'ladi.
2. **O'sish tendensiyasi** — lenta har aylanganda o'sha nuqson qayta o'lchanadi. "Bu teshik 3 soatda 12 mm → 21 mm ga o'sdi" — bu prediktiv maintenance, shunchaki detektsiya emas.
3. **Zero-shot anomaliya qatlami** — datasetda bo'lmagan nuqson turini ham ushlaydi. Hakamlarga demo: notanish narsani lentaga qo'yasiz, tizim baribir "anomaliya" deb belgilaydi.
4. **Millimetrdagi o'lcham** — lenta kengligi (masalan 1200 mm) ma'lum bo'lgani uchun piksel → mm kalibratsiyasi. "Yirtiq bor" emas, "182 mm uzunlikdagi yirtiq, KRITIK".

---

## 4. Dataset strategiyasi (data yo'q holatidan)

Bu eng katta risk — **birinchi 12 soatni shunga bag'ishlang.**

### 4.1 Ochiq datasetlar (0–2 soat)

| Manba | Tavsif | Litsenziya |
|---|---|---|
| Roboflow Universe — `sample-wy2mp/conveyor-belt-damage` | **922 rasm**, instance segmentation, 9 klass: Hole, Tear, Puncture, impact damage, patch work, Conveyor, Roller, Human, Other | CC BY 4.0 |
| Roboflow Universe — `class:defect` qidiruvi | Yuzlab surface-defect datasetlari (metall, rezina, mato) | Turlicha |
| NEU / DAGM / MVTec AD (leather, carpet, grid) | Rezina/teri teksturasidagi nuqsonlar — **pretraining** uchun a'lo | Akademik |

> **Muhim:** MVTec AD "leather" va "carpet" sinflari lenta rezinasiga tekstura jihatdan juda yaqin. Ularda anomaliya modelini oldindan sozlab, keyin lenta rasmlariga o'tkazing.

### 4.2 O'z stendingiz (2–6 soat) — bu demoning yuragi

Eng kuchli demo — **jonli**. Kerakli narsalar:

- Eski rezina lenta bo'lagi / rezina gilam / velosiped kamerasi / qora rezina mat (bozordan ~50 000 so'm)
- Uni aylanma qilib yopishtiring yoki oddiy ikkita valik ustida harakatlantiring (drel/motor bilan)
- Kanselyariya pichog'i bilan **ataylab nuqsonlar** yasang: uzunasiga kesma, ko'ndalang yirtiq, teshik, chuqur tirnalish, yuza tirnalish, "yamoq"
- Telefon kamerasi shtativda, ustidan LED yorug'lik
- **Har xil sharoitda** suratga oling: yorug'/qorong'i, changli (un/talk sepib), nam (suv purkab), ko'mir/tosh bo'laklari ustida

3–4 soatda **2000–4000 kadr** yig'iladi. Bu YOLO-STOD maqolasidagi datasetdan katta.

### 4.3 Sintetik generatsiya (skript bilan, 3–4 soat)

Bu sizga eng katta metrikani beradi va texnik jihatdan ta'sirli:

```python
# Konseptual pipeline
1. Sog'lom lenta kadrlarini yig'ing (yuzlab)
2. Nuqson "kesmalari" (crops) kutubxonasini yarating (10-20 dona yetadi)
3. Har bir sintetik rasm uchun:
   - tasodifiy sog'lom fon tanlang
   - nuqsonni tasodifiy joy/burchak/masshtabda joylashtiring
   - Poisson (seamless) blending qo'llang → chekkalar tabiiy
   - domain randomization: chang shovqini, motion blur (lenta harakati!),
     notekis yorug'lik gradienti, JPEG artefaktlar, ko'mir changi qatlami
   - maska avtomatik → annotatsiya BEPUL
4. Prosedural yirtiq generatori: Bezier egri chiziq + Perlin shovqin
   → cheksiz realistik uzunasiga yirtiqlar
```

**Motion blur juda muhim** — real lenta 2–5 m/s tezlikda harakat qiladi, statik rasmlarda o'qitilgan model real videoda qulaydi.

### 4.4 Hard negatives (ESLAB QOLMANG)

Yolg'on signal (false alarm) — sanoatda tizimni o'ldiradigan narsa. Datasetga **nuqson emas, lekin o'xshash** narsalarni "fon" sifatida qo'shing:

- Lenta ulanish joyi (splice) va vulkanizatsiya chokи
- Yamoq (patch) — allaqachon tuzatilgan joy
- Soya, suv izlari, ho'l dog'lar
- Ko'mir/ruda bo'laklari, chang qatlamlari
- Kamera linzasidagi chang

### 4.5 Klasslar (yakuniy tavsiya — 6 ta)

| # | Klass | Kritiklik |
|---|---|---|
| 1 | `longitudinal_tear` (uzunasiga yirtiq) | 🔴 KRITIK — darhol to'xtatish |
| 2 | `hole` / `puncture` (teshik) | 🔴 KRITIK |
| 3 | `transverse_tear` (ko'ndalang yirtiq) | 🟠 YUQORI |
| 4 | `deep_gouge` (chuqur kesma, kord ko'rinib qolgan) | 🟠 YUQORI |
| 5 | `surface_wear` / `scratch` (yuza yeyilishi) | 🟡 O'RTA — kuzatuv |
| 6 | `edge_damage` (chekka shikasti) | 🟡 O'RTA |
| — | `splice`, `patch` | ⚪ fon sifatida (alert emas) |

---

## 5. Model tanlash

### Asosiy: `YOLO26-s` yoki `YOLO11s-seg` (Ultralytics)

**Nima uchun segmentatsiya, detektsiya emas:** yirtiq — uzun, ingichka, egri shakl. Bounding box uning maydonini 5× oshirib yuboradi va o'lchamni noto'g'ri hisoblaydi. Maskadan esa real uzunlik/kenglikni mm da olasiz.

YOLO26 (Ultralytics ning eng yangi oilasi) hackathon uchun mos: **NMS-free end-to-end inference**, yengilroq detection head, CPU ONNX da YOLO11n dan ~43% tezroq, YOLO26n T4 TensorRT da 1.7 ms. Segmentatsiyada YOLO11 ga nisbatan +3.7 mask AP.

> **Amaliy maslahat:** avval `yolo11s-seg` bilan boshlang (kutubxona/tutorial ko'p, xatolik kam), ishlagach YOLO26 ga o'ting va **taqqoslash jadvalini** slaydga qo'ying — hakamlar buni yaxshi ko'radi.

### Training konfiguratsiyasi

```yaml
model: yolo11s-seg.pt        # COCO pretrained
imgsz: 960                   # ingichka yirtiqlar uchun 640 kam
epochs: 100                  # early stopping patience=20
batch: 16                    # Colab T4 uchun
optimizer: AdamW
lr0: 0.001
cos_lr: true
# Augmentatsiya — lenta uchun sozlangan
mosaic: 0.5                  # 1.0 emas: kontekst muhim
copy_paste: 0.3              # segmentatsiyada juda kuchli
degrees: 5                   # lenta gorizontal — ko'p burmang
translate: 0.2
scale: 0.5
shear: 2
fliplr: 0.5
flipud: 0.0                  # vertikal aylantirmang — noreal
hsv_v: 0.5                   # yorug'lik o'zgarishi — MUHIM
hsv_s: 0.4
erasing: 0.3
```

### Yaxshilash (agar vaqt qolsa, ⚡ prioritet tartibida)

1. **P2 detection head qo'shish** — kichik/ingichka nuqsonlar uchun eng katta samara
2. **CBAM yoki EMA attention** neck da — FDEP-YOLOv8 maqolasidagi kabi
3. **PIoU v2 / Shape-IoU** loss — noto'g'ri shakldagi obyektlar uchun
4. **Test-time augmentation** (`augment=True`) — demo uchun, real-time uchun emas
5. **Slicing inference (SAHI)** — yuqori aniqlikdagi kadrni bo'laklab tekshirish

### Ikkinchi qatlam: PatchCore (anomalib kutubxonasi)

```bash
pip install anomalib
# faqat "sog'lom lenta" rasmlarida o'qitiladi — annotatsiya KERAK EMAS
# 5-10 daqiqada o'qiydi, heatmap chiqaradi
```

Bu qatlam datasetda yo'q nuqsonni ham topadi. Demoda: notanish nuqson yasab ko'rsating → detektor jim, anomaliya modeli qizil heatmap chiqaradi. **Hakamlarni hayratlantiradi.**

---

## 6. Edge deployment (sizda qurilma bor)

| Qurilma | Yo'l | Kutilgan FPS (YOLO11s-seg @640) |
|---|---|---|
| Jetson Orin Nano | PyTorch → ONNX → **TensorRT FP16** | 40–70 FPS |
| Jetson Nano (eski) | ONNX → TensorRT FP16/INT8 | 15–25 FPS |
| Raspberry Pi 5 | ONNX → **OpenVINO / NCNN**, `yolo11n-seg`, imgsz=480 | 5–10 FPS |
| Intel NUC / CPU | **OpenVINO INT8** | 20–30 FPS |

```bash
# Ultralytics bir qatorda eksport qiladi
yolo export model=best.pt format=engine half=True device=0   # TensorRT
yolo export model=best.pt format=openvino int8=True          # CPU
yolo export model=best.pt format=onnx opset=12               # universal
```

**Slaydga majburiy qo'ying:** FP32 vs FP16 vs INT8 — mAP yo'qotish, model hajmi, FPS, quvvat (Watt). Bu "biz haqiqiy edge haqida o'ylaganmiz" degani.

**Muhim nuance:** lenta 3 m/s tezlikda harakatlansa, 30 FPS da har kadr 10 sm lentani ko'rsatadi. Nuqsonni o'tkazib yubormaslik uchun kadrlar **ustma-ust tushishi** kerak — kamera FOV va FPS ni shunga qarab hisoblang va slaydda ko'rsating.

---

## 7. Dashboard (real-time web)

**Stack tavsiyasi (tezlik uchun):** FastAPI + WebSocket (backend) + React/Vite yoki oddiy HTML+Tailwind (frontend) + SQLite. Agar vaqt tig'iz bo'lsa — **Streamlit** 2 soatda ishlaydi, lekin ko'rinishi oddiyroq.

Ekranlar:

1. **Live** — video oqimi, ustida maskalar, chap tomonda alert oqimi, yuqorida KPI: FPS, ishlagan vaqt, bugungi nuqsonlar
2. **Belt Map** — lentaning yoyilgan panoramasi, nuqsonlar joylashuvi bilan (eng ta'sirli ekran)
3. **Defect Log** — jadval: vaqt, klass, o'lcham (mm), pozitsiya (m), severity, snapshot
4. **Trends** — nuqson o'lchamining vaqt bo'yicha o'sish grafigi, RUL (qolgan xizmat muddati) bahosi
5. **Alerts** — Telegram bot integratsiyasi (5 daqiqada qilinadi, demo da telefonga signal kelishi — kuchli effekt)

---

## 8. 72 soatlik roadmap

### Jamoa rollari (3–4 kishi)
- **A — Data**: yig'ish, annotatsiya, sintetik generator
- **B — ML**: training, tuning, eksport, anomaliya modeli
- **C — Backend/Edge**: pipeline, tracking, severity, deployment
- **D — Frontend/Pitch**: dashboard, slaydlar, video

---

### 🗓 KUN 1 (0–24 soat) — Data va baseline

| Soat | Ish | Kim |
|---|---|---|
| 0–1 | Kick-off, repo, GitHub, rol taqsimoti, klasslar ro'yxatini muzlatish | Hammasi |
| 1–3 | Roboflow datasetini yuklash, tozalash, klasslarni moslashtirish | A |
| 1–3 | Colab muhiti, `ultralytics` o'rnatish, **baseline train** (tayyor datasetda, 30 epoch) — kechqurun natija bo'lishi uchun | B |
| 1–4 | Stend qurish: rezina lenta, valiklar, kamera, yoritish | C |
| 3–6 | Stendda video yozish: 6 klass × 5 sharoit, 2000+ kadr | A + C |
| 4–8 | FastAPI skeleti, video oqim → WebSocket, bo'sh dashboard | C + D |
| 6–10 | Annotatsiya (Roboflow yoki CVAT, SAM-assisted → 3× tez) | A + D |
| 8–12 | Sintetik generator skripti (copy-paste + Poisson blending + motion blur) | B |
| 12–16 | **Train v1**: ochiq dataset + o'z data + sintetik, yolo11s-seg, 100 epoch | B |
| 16–18 | Kalibratsiya: piksel→mm, perspektiv to'g'rilash, ROI | C |
| 18–22 | Optical flow bilan lenta tezligini o'lchash + kadrlarni yopishtirish (belt unrolling) | C |
| 22–24 | ✅ **Checkpoint 1**: model videoda nuqson topmoqda, dashboard video ko'rsatmoqda | Hammasi |

> ⚠️ 24-soatga qadar **ishlaydigan uchidan-uchiga (end-to-end) prototip** bo'lishi shart, sifati past bo'lsa ham. Qolgan vaqt — sifatni oshirish.

---

### 🗓 KUN 2 (24–48 soat) — Sifat va "wow" funksiyalar

| Soat | Ish | Kim |
|---|---|---|
| 24–28 | Xato tahlili: qaysi klass yomon? Yolg'on signallar qayerdan? → hard negatives qo'shish | A + B |
| 24–30 | ByteTrack integratsiyasi — bir nuqson = bir alert | C |
| 28–34 | **Train v2**: kengaytirilgan dataset, imgsz=960, P2 head yoki attention | B |
| 30–34 | Severity engine: o'lcham + klass + o'sish → 3 darajali alert, SQLite log | C |
| 30–36 | Dashboard: Live + Defect Log + Belt Map ekranlari | D |
| 34–38 | **PatchCore** (anomalib) o'qitish sog'lom kadrlarda, heatmap endpoint | B |
| 36–40 | Nuqsonni lenta koordinatasiga bog'lash + o'sish tendensiyasi grafigi | C + D |
| 38–42 | Edge eksport: ONNX → TensorRT/OpenVINO, FPS benchmark jadvali | B + C |
| 42–46 | Telegram alert boti, avto-STOP signali simulyatsiyasi | C |
| 46–48 | ✅ **Checkpoint 2**: barcha funksiyalar ishlaydi, metrikalar yozilgan | Hammasi |

---

### 🗓 KUN 3 (48–72 soat) — Sayqal va pitch

| Soat | Ish | Kim |
|---|---|---|
| 48–52 | **Train v3** (yakuniy): eng yaxshi konfiguratsiya, uzunroq training, TTA baholash | B |
| 48–54 | Baholash: mAP@50, mAP@50-95, per-class recall, confusion matrix, PR egri chiziqlari, FPS | B |
| 52–58 | Model taqqoslash jadvali: YOLO11n/s/m vs YOLO26 vs baseline; FP32/FP16/INT8 | B |
| 54–60 | Dashboard UI sayqallash, dark mode, responsive, demo ma'lumot bilan to'ldirish | D |
| 56–62 | **Edge demo tayyorlash**: Jetson/RPi da jonli ishga tushirish, zaxira video yozib qo'yish | C |
| 60–66 | Pitch deck (10–12 slayd) + 2 daqiqalik demo video yozish | D + Hammasi |
| 66–70 | **Repetitsiya** ×3, savol-javoblarga tayyorgarlik, README + arxitektura diagrammasi | Hammasi |
| 70–72 | Zaxira reja: internet/qurilma ishlamasa — oldindan yozilgan video, offline dashboard | Hammasi |

---

## 9. Metrikalar — nimani o'lchash va ko'rsatish

**Standart ML metrikalari:**
- mAP@50, mAP@50-95 (umumiy va har klass bo'yicha)
- Precision / Recall — **kritik klasslarda Recall ni birinchi o'ringa qo'ying** (yirtiqni o'tkazib yuborish yolg'on signaldan 1000× qimmat)
- Confusion matrix, PR-curve

**Sanoatga xos metrikalar (bu sizni ajratadi):**
- **False alarm/soat** — soatiga nechta yolg'on signal (maqsad: < 1)
- **Detection latency** — nuqson kadrga kirgandan alert chiqquncha ms
- **Lenta aylanishiga to'g'ri kelgan qamrov (%)** — kamera lentaning necha % ini ko'rdi
- **O'lcham xatosi (mm)** — o'lchagich bilan solishtiring, maqsad ±5 mm
- **FPS / Latency / Power (W)** har bir edge qurilmada

**Baholash uchun alohida test to'plami:** stendda **oxirgi kunda** yozilgan, training da umuman ishlatilmagan video. Buni slaydda alohida ta'kidlang — "biz o'zimizni aldamadik".

---

## 10. Risklar va zaxira rejalari

| Risk | Ehtimol | Zaxira reja |
|---|---|---|
| Dataset yetarli emas / model yomon o'qiydi | Yuqori | Sintetik generatorga ko'proq tayaning; klasslar sonini 6 dan 3 ga qisqartiring (tear / hole / wear) |
| Colab GPU uzilib qoladi | Yuqori | Kaggle (30 soat/hafta P100) zaxira; har epoch checkpoint Google Drive ga |
| Real-time FPS yetmaydi | O'rta | `yolo11n-seg`, imgsz=480, har 2-kadrni tashlab ketish + tracking bilan interpolyatsiya |
| Jetson da TensorRT eksporti ishlamaydi | O'rta | ONNX Runtime yoki oddiy PyTorch bilan demo; FPS ni slaydda halol ko'rsating |
| Demo paytida kamera/internet ishlamaydi | O'rta | **Oldindan yozilgan video fayl** rejimi — dashboard uni jonli oqim kabi ko'rsatsin |
| Yolg'on signallar ko'p | Yuqori | Hard negatives, confidence threshold tuning, tracking bilan "3 kadrda tasdiqlansin" qoidasi |

---

## 11. Pitch strukturasi (10 slayd, 5 daqiqa)

1. **Ilgak** — "Bu 4 soatlik to'xtash. $100 000. Sababi — 2 mm lik kesma edi."
2. **Muammo** — konveyer nosozliklari, inson tekshiruvi cheklovlari, statistika
3. **Yechim** — BeltGuard: kamera + edge AI + real-time monitoring (bitta rasm)
4. **Demo** ← eng ko'p vaqt shu yerga (jonli yoki video)
5. **Texnologiya** — arxitektura diagrammasi, 2 qatlamli model (detektsiya + anomaliya)
6. **Natijalar** — mAP, FPS, o'lcham aniqligi, taqqoslash jadvali
7. **Farqimiz** — belt map, o'sish tendensiyasi, zero-shot anomaliya, mm o'lchov
8. **Edge** — Jetson da real-time, oflayn ishlaydi (konda internet yo'q!)
9. **Biznes** — ROI hisobi: qurilma narxi vs oldini olingan to'xtash
10. **Roadmap** — multispektral, depth sensor, rolik harorati, PLC integratsiyasi

---

## 12. Darhol boshlash uchun buyruqlar

```bash
# 1. Muhit
pip install ultralytics anomalib opencv-python fastapi uvicorn[standard] \
            supervision roboflow albumentations

# 2. Dataset (Roboflow)
from roboflow import Roboflow
rf = Roboflow(api_key="SIZNING_KALIT")
ds = rf.workspace("sample-wy2mp").project("conveyor-belt-damage") \
       .version(1).download("yolov8")

# 3. Baseline train (Colab T4 da ~40 daqiqa)
yolo segment train model=yolo11s-seg.pt data=conveyor-belt-damage/data.yaml \
     imgsz=960 epochs=100 batch=8 patience=20 cos_lr=True \
     copy_paste=0.3 hsv_v=0.5 flipud=0.0 degrees=5 project=beltguard

# 4. Baholash
yolo segment val model=beltguard/train/weights/best.pt split=test

# 5. Edge eksport
yolo export model=beltguard/train/weights/best.pt format=engine half=True
```

---

## 13. Manbalar

- [YOLO-STOD: an industrial conveyor belt tear detection model based on Yolov5 algorithm — Scientific Reports](https://www.nature.com/articles/s41598-024-83619-6)
- [Research on conveyor belt damage detection method based on FDEP−YOLOv8 — Scientific Reports](https://www.nature.com/articles/s41598-025-20391-1)
- [Deep learning-based damage detection of mining conveyor belt — Measurement (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S0263224121001561)
- [YOLOv8-LDH: lightweight model for conveyor belt damage detection based on multispectral imaging](https://www.sciencedirect.com/science/article/abs/pii/S026322412500034X)
- [A Surface Defect Detection System for Industrial Conveyor Belt Inspection Using Apple's TrueDepth Camera — MDPI Applied Sciences](https://www.mdpi.com/2076-3417/16/2/609)
- [Visual detection method based on line lasers for longitudinal tears in conveyor belts — Measurement](https://www.sciencedirect.com/science/article/abs/pii/S0263224121007521)
- [Conveyor-belt-damage dataset (922 rasm, 9 klass, CC BY 4.0) — Roboflow Universe](https://universe.roboflow.com/sample-wy2mp/conveyor-belt-damage)
- [Awesome Industrial Anomaly Detection — GitHub (PatchCore, PaDiM, EfficientAD ro'yxati)](https://github.com/m-3lab/awesome-industrial-anomaly-detection)
- [Anomaly detection in images using PatchCore — dataroots](https://dataroots.io/blog/anomaly-detection-in-images-using-patchcore)
- [Ultralytics YOLO26 — rasmiy hujjat](https://docs.ultralytics.com/models/yolo26)
- [SynSur: generative pipeline for synthetic industrial surface defect generation — arXiv](https://arxiv.org/html/2604.26633)
- [Few-Shot Diffusion-Based Defect Synthesis — arXiv](https://arxiv.org/abs/2604.22850)
- [Conveyor Downtime: Quantifying the True Cost of Idler Failure — Vayeron](https://smartidler.com/insights/conveyor-downtime-quantifying-the-true-cost-of-idler-failure/)
- [Conveyor Belt Inspection for Preventative Maintenance — Cognex](https://www.cognex.com/en/applications/automated-defect-detection/conveyor-belt-inspection-for-preventative-maintenance)
- [Conveyor Belt Monitoring — Ripik.ai (sanoat yechimi)](https://www.ripik.ai/conveyor-belt-monitoring/)
