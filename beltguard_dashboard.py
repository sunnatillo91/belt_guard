"""
BeltGuard — Konveyer lentasi monitoring dashboard (Streamlit) — v2 UI

Ishga tushirish:
    pip install streamlit ultralytics opencv-python pandas
    streamlit run beltguard_dashboard.py

Model:  models/best.pt  (Colab'dan yuklab olingan fayl)
Manba:  veb-kamera (0/1) yoki video fayl

Eslatma: chiroyli dark-mavzu uchun yonidagi .streamlit/config.toml faylini ham
loyiha papkasiga qo'ying (beltguard_dashboard.py bilan bir joyda).
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------- sozlamalar

MODEL_PATH_DEFAULT = "models/best960.pt"   # stend kadrlari bilan fine-tune qilingan

# Brauzer tab'i yopilganda asyncio har kadrda ConnectionResetError chiqaradi —
# ishlashga ta'sir qilmaydi, faqat terminalni to'ldiradi.
logging.getLogger("asyncio").setLevel(logging.CRITICAL)

CLASS_SEVERITY = {
    "tear": "CRITICAL",
    "longitudinal_tear": "CRITICAL",
    "hole": "CRITICAL",
    "puncture": "CRITICAL",
    "impact damage": "HIGH",
    "impact_damage": "HIGH",
    "deep_gouge": "HIGH",
    "transverse_tear": "HIGH",
    "gouge": "HIGH",
    "scratch": "MEDIUM",
    "surface_wear": "MEDIUM",
    "wear": "MEDIUM",
    "edge_damage": "MEDIUM",
    "conveyor": "IGNORE",
    "roller": "IGNORE",
    "human": "IGNORE",
    "patch work": "INFO",
    "patch_work": "INFO",
    "other objects": "IGNORE",
}

SEVERITY_ORDER = {"CRITICAL": 3, "HIGH": 2, "MEDIUM": 1, "INFO": 0, "IGNORE": -1}

# Status ranglar — har bir mavzu uchun alohida (oq fonda och sariq o'qilmaydi)
SEVERITY_HEX_DARK = {"CRITICAL": "#d03b3b", "HIGH": "#ec835a", "MEDIUM": "#fab219", "INFO": "#8a8a83"}
SEVERITY_HEX_LIGHT = {"CRITICAL": "#b32020", "HIGH": "#c2551f", "MEDIUM": "#9a6a00", "INFO": "#6b6b64"}
SEVERITY_BGR = {  # OpenCV chizish uchun (BGR!) — video mavzuga bog'liq emas
    "CRITICAL": (59, 59, 208),
    "HIGH": (90, 131, 236),
    "MEDIUM": (25, 178, 250),
    "INFO": (160, 160, 160),
}
SEVERITY_LABEL = {"CRITICAL": "KRITIK", "HIGH": "JIDDIY", "MEDIUM": "KUZATUV", "INFO": "MA'LUMOT"}
SEVERITY_ICON = {"CRITICAL": "⛔", "HIGH": "⚠️", "MEDIUM": "👁", "INFO": "ℹ️"}

SIZE_ESCALATION_MM = {"MEDIUM_TO_HIGH": 80.0, "HIGH_TO_CRITICAL": 150.0}

# Mavzu tokenlari — CSS o'zgaruvchilariga aylantiriladi
THEME_TOKENS = {
    "dark": {
        "surface": "#232322", "surface2": "#202020", "border": "#33332f",
        "border2": "#3a3a36", "text": "#e8e8e2", "text_strong": "#ffffff",
        "muted": "#8a8a83", "label": "#c3c2b7", "good": "#0ca30c",
        "topbar_a": "#20201f", "topbar_b": "#262624", "good_bg": "rgba(12,163,12,.12)",
        "good_br": "rgba(12,163,12,.35)", "stop_bg": "rgba(195,194,183,.08)",
    },
    "light": {
        "surface": "#ffffff", "surface2": "#f4f4f1", "border": "#dedcd4",
        "border2": "#cfcdc4", "text": "#1f1f1d", "text_strong": "#111110",
        "muted": "#6e6e66", "label": "#4a4a44", "good": "#0a7d0a",
        "topbar_a": "#ffffff", "topbar_b": "#f2f2ee", "good_bg": "rgba(10,125,10,.10)",
        "good_br": "rgba(10,125,10,.30)", "stop_bg": "rgba(0,0,0,.05)",
    },
}


def active_theme() -> str:
    """Streamlit sozlamalaridagi mavzu (Settings -> Appearance). Default: dark."""
    try:
        return "light" if st.context.theme.type == "light" else "dark"
    except Exception:
        return "dark"


@dataclass
class DefectEvent:
    ts: str
    cls: str
    severity: str
    conf: float
    length_mm: float | None
    width_mm: float | None
    frame_idx: int


@dataclass
class SessionState:
    events: list = field(default_factory=list)
    last_alert_key: dict = field(default_factory=dict)
    fps_hist: deque = field(default_factory=lambda: deque(maxlen=30))


def escalate(severity: str, length_mm: float | None) -> str:
    if length_mm is None:
        return severity
    if severity == "MEDIUM" and length_mm >= SIZE_ESCALATION_MM["MEDIUM_TO_HIGH"]:
        severity = "HIGH"
    if severity == "HIGH" and length_mm >= SIZE_ESCALATION_MM["HIGH_TO_CRITICAL"]:
        severity = "CRITICAL"
    return severity


def mask_size_mm(mask: np.ndarray, mm_per_px: float | None):
    if mm_per_px is None:
        return None, None
    pts = cv2.findNonZero(mask.astype(np.uint8))
    if pts is None or len(pts) < 5:
        return None, None
    (_, _), (w, h), _ = cv2.minAreaRect(pts)
    return round(max(w, h) * mm_per_px, 1), round(min(w, h) * mm_per_px, 1)


@st.cache_resource(show_spinner="Model yuklanmoqda...")
def load_model(path: str):
    from ultralytics import YOLO
    return YOLO(path)


def open_source(src: str):
    return cv2.VideoCapture(int(src)) if src.isdigit() else cv2.VideoCapture(src)


# ---------------------------------------------------------------- UI: mavzu/CSS

st.set_page_config(page_title="BeltGuard", page_icon="🛡️", layout="wide",
                   initial_sidebar_state="expanded")

THEME = active_theme()
T = THEME_TOKENS[THEME]
SEVERITY_HEX = SEVERITY_HEX_LIGHT if THEME == "light" else SEVERITY_HEX_DARK
GOOD_HEX = T["good"]
MUTED = T["label"]

st.markdown(f"""
<style>
:root {{
  --bg-surface: {T['surface']};      --bg-surface2: {T['surface2']};
  --bg-border: {T['border']};        --bg-border2: {T['border2']};
  --bg-text: {T['text']};            --bg-text-strong: {T['text_strong']};
  --bg-muted: {T['muted']};          --bg-label: {T['label']};
  --bg-good: {T['good']};            --bg-crit: {SEVERITY_HEX['CRITICAL']};
  --bg-topbar-a: {T['topbar_a']};    --bg-topbar-b: {T['topbar_b']};
  --bg-good-bg: {T['good_bg']};      --bg-good-br: {T['good_br']};
  --bg-stop-bg: {T['stop_bg']};
}}

/* Streamlit servis elementlari: menyu ko'rinib turadi — u orqali mavzu almashtiriladi */
footer {{visibility: hidden; height: 0;}}
.block-container {{padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1500px;}}

/* ---- sarlavha ---- */
.bg-topbar {{display: flex; align-items: center; justify-content: space-between;
  padding: 14px 22px; border-radius: 14px; margin-bottom: 16px;
  background: linear-gradient(90deg, var(--bg-topbar-a) 0%, var(--bg-topbar-b) 100%);
  border: 1px solid var(--bg-border);}}
.bg-brand {{display: flex; align-items: center; gap: 12px;}}
.bg-logo {{font-size: 26px;}}
.bg-title {{font-size: 20px; font-weight: 700; color: var(--bg-text-strong); letter-spacing: .2px;}}
.bg-sub {{font-size: 12.5px; color: var(--bg-muted); margin-top: 1px;}}
.bg-pill {{display: inline-flex; align-items: center; gap: 8px; font-size: 13px;
  font-weight: 600; padding: 7px 14px; border-radius: 999px;}}
.bg-pill.run  {{color: var(--bg-good); background: var(--bg-good-bg); border: 1px solid var(--bg-good-br);}}
.bg-pill.stop {{color: var(--bg-label); background: var(--bg-stop-bg); border: 1px solid var(--bg-border2);}}
.bg-dot {{width: 8px; height: 8px; border-radius: 50%; background: currentColor;
  animation: bgpulse 1.6s ease-in-out infinite;}}
@keyframes bgpulse {{50% {{opacity: .35;}}}}
@media (prefers-reduced-motion: reduce) {{.bg-dot {{animation: none;}}}}

/* ---- KPI kartalar ---- */
.bg-kpis {{display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 16px;}}
.bg-kpi {{background: var(--bg-surface); border: 1px solid var(--bg-border); border-radius: 12px;
  padding: 14px 18px;}}
.bg-kpi .l {{font-size: 12px; color: var(--bg-muted); text-transform: uppercase;
  letter-spacing: .6px; margin-bottom: 4px;}}
.bg-kpi .v {{font-size: 26px; font-weight: 700; color: var(--bg-text-strong); line-height: 1.15;
  font-variant-numeric: tabular-nums;}}
.bg-kpi .v.crit {{color: var(--bg-crit);}}
.bg-kpi .u {{font-size: 12px; color: var(--bg-muted); margin-left: 3px; font-weight: 400;}}

/* ---- panellar ---- */
.bg-panel-t {{font-size: 13px; font-weight: 700; color: var(--bg-label);
  text-transform: uppercase; letter-spacing: .7px; margin: 4px 0 8px 2px;}}
.bg-empty {{border: 1px dashed var(--bg-border2); border-radius: 12px; padding: 34px 20px;
  text-align: center; color: var(--bg-muted); font-size: 14px; background: var(--bg-surface2);}}
.bg-empty .big {{font-size: 30px; margin-bottom: 8px;}}

/* ---- alert chiplar ---- */
.bg-alert {{display: flex; align-items: center; gap: 10px; padding: 9px 12px;
  border-radius: 10px; margin-bottom: 7px; background: var(--bg-surface);
  border: 1px solid var(--bg-border); border-left: 4px solid var(--sv, var(--bg-muted));
  font-size: 13.5px; color: var(--bg-text);}}
.bg-alert .sv {{font-weight: 700; color: var(--sv);}}
.bg-alert .t {{margin-left: auto; color: var(--bg-muted); font-size: 12px;
  font-variant-numeric: tabular-nums;}}

/* ---- video ramkasi ---- */
div[data-testid="stImage"] img {{border-radius: 12px; border: 1px solid var(--bg-border);}}

/* ---- sidebar ---- */
section[data-testid="stSidebar"] {{border-right: 1px solid var(--bg-border);}}
section[data-testid="stSidebar"] .stMarkdown h3 {{font-size: 14px;}}
</style>
""", unsafe_allow_html=True)


def topbar(running: bool):
    pill = ('<span class="bg-pill run"><span class="bg-dot"></span>MONITORING ISHLAYAPTI</span>'
            if running else '<span class="bg-pill stop">⏸ &nbsp;TO\'XTATILGAN</span>')
    st.markdown(f"""
    <div class="bg-topbar">
      <div class="bg-brand">
        <div class="bg-logo">🛡️</div>
        <div>
          <div class="bg-title">BeltGuard</div>
          <div class="bg-sub">Konveyer lentasi shikastlarini real-time aniqlash · AI monitoring</div>
        </div>
      </div>
      {pill}
    </div>""", unsafe_allow_html=True)


def kpi_html(fps, total, crit, mmpp):
    cal = f"1 px = {mmpp:.3f} mm" if mmpp else "kiritilmagan"
    cal_cls = "" if mmpp else ""
    return f"""
    <div class="bg-kpis">
      <div class="bg-kpi"><div class="l">Model tezligi</div>
        <div class="v">{fps:.1f}<span class="u">FPS</span></div></div>
      <div class="bg-kpi"><div class="l">Jami nuqsonlar</div>
        <div class="v">{total}</div></div>
      <div class="bg-kpi"><div class="l">Kritik</div>
        <div class="v {'crit' if crit else ''}">{crit}</div></div>
      <div class="bg-kpi"><div class="l">Kalibratsiya</div>
        <div class="v" style="font-size:17px; padding-top:6px; color:{'#e8e8e2' if mmpp else '#8a8a83'}">{cal}</div></div>
    </div>"""


def alerts_html(events):
    if not events:
        return ('<div class="bg-empty"><div class="big">✅</div>'
                'Alert yo\'q — lenta holati normal</div>')
    rows = []
    for e in events[-7:][::-1]:
        hexc = SEVERITY_HEX.get(e.severity, "#8a8a83")
        size = f" · {e.length_mm:.0f}×{e.width_mm:.0f} mm" if e.length_mm else ""
        rows.append(
            f'<div class="bg-alert" style="--sv:{hexc}">'
            f'<span>{SEVERITY_ICON.get(e.severity,"")}</span>'
            f'<span class="sv">{SEVERITY_LABEL.get(e.severity, e.severity)}</span>'
            f'<span>{e.cls}{size}</span>'
            f'<span class="t">{e.ts}</span></div>')
    return "".join(rows)


# ---------------------------------------------------------------- sidebar

with st.sidebar:
    st.markdown("### ⚙️ Sozlamalar")
    model_path = st.text_input("Model fayli", MODEL_PATH_DEFAULT)
    source = st.text_input("Manba", "demo/demo.mp4", help="0 = veb-kamera, yoki video fayl yo'li")
    conf_th = st.slider("Confidence chegarasi", 0.10, 0.90, 0.50, 0.05,
                        help="F1-Confidence egri chizig'i bo'yicha optimal ≈ 0.40")
    imgsz = st.select_slider("Inference o'lchami", [480, 640, 960], value=960,
                             help="Model qaysi o'lchamda o'qitilgan bo'lsa, o'shani tanlang")

    st.divider()
    st.markdown("### 📏 Kalibratsiya (px → mm)")
    st.caption("Lenta kengligini kadrda (px) va real hayotda (mm) kiriting — "
               "shunda o'lchamlar mm da chiqadi.")
    belt_px = st.number_input("Lenta kengligi kadrda (px)", 0, 4000, 640)
    belt_mm = st.number_input("Lenta real kengligi (mm)", 0, 3000, 421)
    mm_per_px = (belt_mm / belt_px) if belt_px > 0 and belt_mm > 0 else None
    if mm_per_px:
        st.success(f"1 px = {mm_per_px:.3f} mm")

    st.divider()
    min_area_pct = st.slider("Minimal nuqson o'lchami (kadr %)", 0.0, 2.0, 0.0, 0.05,
                             help="Bundan kichik maskalar chang/dog' deb hisoblanadi. "
                                  "Chang ko'p bo'lsa 0.3–0.5 qiling.")
    alert_cooldown = st.number_input("Bir klass uchun alert oralig'i (s)", 1, 120, 10)
    run = st.toggle("**▶️ Monitoringni boshlash**", value=False)
    st.caption("BeltGuard v1.0 · Hackathon 2026")

# ---------------------------------------------------------------- layout

if "bg" not in st.session_state:
    st.session_state.bg = SessionState()
S: SessionState = st.session_state.bg

topbar(run)
kpi_ph = st.empty()

col_video, col_side = st.columns([2.1, 1], gap="medium")
with col_video:
    st.markdown('<div class="bg-panel-t">📹 Jonli oqim</div>', unsafe_allow_html=True)
    video_ph = st.empty()
with col_side:
    st.markdown('<div class="bg-panel-t">🚨 Oxirgi alertlar</div>', unsafe_allow_html=True)
    alerts_ph = st.empty()

st.markdown('<div class="bg-panel-t" style="margin-top:14px">📋 Nuqsonlar jurnali</div>',
            unsafe_allow_html=True)
table_ph = st.empty()
dl_ph = st.empty()


def render_table(with_download=False):
    # CSV tugmasi faqat sikl tashqarisida chiziladi — bitta ishga tushishda
    # ikki marta chizilsa Streamlit ID to'qnashuvi beradi.
    df = pd.DataFrame([e.__dict__ for e in S.events])
    if df.empty:
        table_ph.markdown('<div class="bg-empty"><div class="big">📋</div>'
                          'Hozircha nuqson qayd etilmagan</div>', unsafe_allow_html=True)
        return
    df = df.rename(columns={"ts": "Vaqt", "cls": "Klass", "severity": "Daraja",
                            "conf": "Ishonch", "length_mm": "Uzunlik (mm)",
                            "width_mm": "Kenglik (mm)", "frame_idx": "Kadr"})
    df["Daraja"] = df["Daraja"].map(lambda s: f"{SEVERITY_ICON.get(s,'')} {SEVERITY_LABEL.get(s,s)}")
    table_ph.dataframe(df.iloc[::-1], width="stretch", height=290, hide_index=True)
    if with_download:
        dl_ph.download_button("⬇️ CSV hisobot", df.to_csv(index=False).encode(),
                              "beltguard_defects.csv", "text/csv")


def render_idle():
    fps = sum(S.fps_hist) / len(S.fps_hist) if S.fps_hist else 0.0
    n_crit = sum(1 for e in S.events if e.severity == "CRITICAL")
    kpi_ph.markdown(kpi_html(fps, len(S.events), n_crit, mm_per_px), unsafe_allow_html=True)
    video_ph.markdown('<div class="bg-empty" style="padding:80px 20px">'
                      '<div class="big">📷</div>Monitoring o\'chiq — boshlash uchun '
                      'chap paneldagi <b>▶️ tugmani</b> yoqing</div>', unsafe_allow_html=True)
    alerts_ph.markdown(alerts_html(S.events), unsafe_allow_html=True)
    render_table(with_download=True)


# ---------------------------------------------------------------- asosiy sikl

if not run:
    render_idle()
    st.stop()

if not Path(model_path).exists():
    st.error(f"Model topilmadi: `{model_path}` — Colab'dan `best.pt` ni yuklab, "
             "`models/` papkaga qo'ying.")
    render_idle()
    st.stop()

model = load_model(model_path)
cap = open_source(source.strip())
if not cap.isOpened():
    st.error(f"Manba ochilmadi: {source} — kamera indeksi yoki fayl yo'lini tekshiring.")
    render_idle()
    st.stop()

is_file = not source.strip().isdigit()
frame_idx = 0

while run:
    ok, frame = cap.read()
    if not ok:
        if is_file:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        st.warning("Kamera oqimi uzildi")
        break

    t0 = time.time()
    res = model.predict(frame, conf=conf_th, imgsz=imgsz, verbose=False)[0]
    S.fps_hist.append(1000.0 / max((time.time() - t0) * 1000, 1e-3))

    annotated = frame.copy()
    frame_severities = []

    if res.masks is not None and res.boxes is not None:
        masks = res.masks.data.cpu().numpy()
        for i, box in enumerate(res.boxes):
            cls_name = res.names[int(box.cls)].strip()
            sev = CLASS_SEVERITY.get(cls_name.lower(), "MEDIUM")
            if sev == "IGNORE":
                continue
            conf = float(box.conf)

            m = masks[i]
            if m.shape[:2] != frame.shape[:2]:
                m = cv2.resize(m, (frame.shape[1], frame.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
            m_bin = (m > 0.5).astype(np.uint8)

            # Chang va mayda dog'lar filtri: kadr maydonining ma'lum foizidan
            # kichik maskalar nuqson deb qabul qilinmaydi.
            if m_bin.sum() / m_bin.size * 100 < min_area_pct:
                continue

            length_mm, width_mm = mask_size_mm(m_bin, mm_per_px)
            sev = escalate(sev, length_mm)
            frame_severities.append(sev)
            color = SEVERITY_BGR.get(sev, (255, 255, 255))

            overlay = annotated.copy()
            overlay[m_bin == 1] = color
            annotated = cv2.addWeighted(overlay, 0.35, annotated, 0.65, 0)
            contours, _ = cv2.findContours(m_bin, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(annotated, contours, -1, color, 2)

            x1, y1 = map(int, box.xyxy[0][:2])
            label = f"{cls_name} {conf:.2f}"
            if length_mm:
                label += f" | {length_mm:.0f}mm"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x1, max(0, y1 - th - 14)),
                          (x1 + tw + 8, max(th + 4, y1 - 2)), color, -1)
            cv2.putText(annotated, label, (x1 + 4, max(th, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            now = time.time()
            if sev != "INFO" and now - S.last_alert_key.get(cls_name, 0) > alert_cooldown:
                S.last_alert_key[cls_name] = now
                S.events.append(DefectEvent(
                    ts=time.strftime("%H:%M:%S"), cls=cls_name, severity=sev,
                    conf=round(conf, 2), length_mm=length_mm, width_mm=width_mm,
                    frame_idx=frame_idx))

    if frame_severities:
        worst = max(frame_severities, key=lambda s: SEVERITY_ORDER[s])
        bc = SEVERITY_BGR.get(worst, (255, 255, 255))
        cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 44), bc, -1)
        txt = {"CRITICAL": "!!! KRITIK SHIKAST - LENTANI TO'XTATING !!!",
               "HIGH": "DIQQAT: jiddiy shikast aniqlandi",
               "MEDIUM": "Kuzatuv: yuza nuqson",
               "INFO": "Ma'lumot"}[worst]
        cv2.putText(annotated, txt, (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2)

    video_ph.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                   channels="RGB", width="stretch")

    fps_now = sum(S.fps_hist) / len(S.fps_hist) if S.fps_hist else 0
    n_crit = sum(1 for e in S.events if e.severity == "CRITICAL")
    kpi_ph.markdown(kpi_html(fps_now, len(S.events), n_crit, mm_per_px),
                    unsafe_allow_html=True)
    alerts_ph.markdown(alerts_html(S.events), unsafe_allow_html=True)

    if frame_idx % 15 == 0:
        render_table()

    frame_idx += 1

cap.release()
render_table(with_download=True)
