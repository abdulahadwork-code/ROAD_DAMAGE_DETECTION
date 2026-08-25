import streamlit as st
import io, os, cv2, sqlite3, tempfile, subprocess
import numpy as np
from PIL import Image
from PIL.ExifTags import GPSTAGS, TAGS
from ultralytics import YOLO
import folium
import imageio_ffmpeg
from streamlit_folium import st_folium

st.set_page_config(page_title="RoadSense AI", page_icon="🛣️", layout="wide")

st.markdown("""
<style>
    .block-container { max-width: 100%; padding: 1rem 1.5rem; }
    section[data-testid="stSidebar"] { display: none; }
    .stApp { background: #0d1526; }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: #111a2b !important; border-color: #1e2a44 !important; border-radius: 12px;
    }
    [data-testid="stBaseButton-primary"] {
        background: #2563eb !important; color: white !important; border: none !important;
    }
    .lbl { font-size:.68rem; letter-spacing:1.2px; color:#8fa3c7; text-transform:uppercase; }
    .badge { font-size:.7rem; background:#1e2a44; color:#c7d2ea; padding:4px 10px; border-radius:6px; }
    [data-testid*="Dropzone" i]:has([data-testid*="Dropzone" i]) {
        background: transparent !important; border: none !important; padding: 0 !important;
    }
    [data-testid*="Dropzone" i]:not(:has([data-testid*="Dropzone" i])) {
        border: 2px dashed #33415e !important; background: transparent !important;
        border-radius: 10px !important; padding: 30px 12px !important;
        display: flex !important; flex-direction: column !important;
        align-items: center !important; justify-content: center !important; text-align: center !important;
    }
    [data-testid*="Dropzone" i]:not(:has([data-testid*="Dropzone" i])) > * { display: none !important; }
    [data-testid*="Dropzone" i]:not(:has([data-testid*="Dropzone" i])) > [data-testid*="File" i],
    [data-testid*="Dropzone" i]:not(:has([data-testid*="Dropzone" i])) > div:has(img) { display: flex !important; }
    [data-testid*="Dropzone" i]:not(:has([data-testid*="Dropzone" i]))::before {
        content: "Drag & drop footage here"; color: #fff; font-weight: 700; font-size: .95rem;
    }
    [data-testid*="Dropzone" i]:not(:has([data-testid*="Dropzone" i]))::after {
        content: "Supports MP4, AVI, JPG, PNG"; color: #8fa3c7; font-size: .75rem; margin-top: 6px;
    }
    [data-testid*="FileUpload" i] label { display: none !important; }
    [data-testid*="Dropzone" i]:has([data-testid*="Dropzone" i]) button { display: none !important; }
    [data-testid="stImage"], [data-testid="stVideo"] {
        background: #000 !important; border-radius: 8px; padding: 8px; min-height: 470px;
        display: flex; align-items: center; justify-content: center;
    }
</style>
""", unsafe_allow_html=True)

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'}

CLASS_META = [
    ("pothole", "Potholes", "#ef4444"), ("alligator", "Alligator Cracks", "#f59e0b"),
    ("longitudinal", "Longitudinal Cracks", "#f59e0b"), ("lateral", "Lateral Cracks", "#f59e0b"),
    ("edge", "Edge Cracking", "#f59e0b"), ("ravelling", "Ravelling", "#a78bfa"),
    ("rutting", "Rutting", "#a78bfa"), ("striping", "Striping", "#38bdf8"),
]

@st.cache_resource
def load_model():
    return YOLO("best.pt")   

def canon(name):
    n = str(name).lower()
    if 'pothole' in n: return 'pothole'
    if 'alligator' in n: return 'alligator'
    if 'longitudinal' in n: return 'longitudinal'
    if 'lateral' in n: return 'lateral'
    if 'edge' in n: return 'edge'
    if 'ravell' in n or 'ravel' in n: return 'ravelling'
    if 'rut' in n: return 'rutting'
    if 'strip' in n: return 'striping'
    return None

def keep(key, conf):
    return conf >= (0.20 if key == 'pothole' else 0.05)

def new_counts():
    return {k: 0 for k in ['pothole', 'alligator', 'longitudinal', 'lateral',
                           'edge', 'ravelling', 'rutting', 'striping']}

def severity_for(key, area):
    if key == 'pothole': return "High" if area > 8000 else "Medium"
    if key == 'striping': return "Low"
    return "High" if area > 10000 else "Medium"

SEV_COLORS = {"High": (0, 0, 255), "Medium": (0, 165, 255), "Low": (0, 200, 0)}

def draw_overlay(img, r):
    H, W = img.shape[:2]
    for box in r.boxes:
        cname = r.names[int(box.cls[0])]
        conf = float(box.conf[0])
        key = canon(cname)
        if key is None or not keep(key, conf): continue
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        area = (x2 - x1) * (y2 - y1)
        sev = severity_for(key, area)
        color = SEV_COLORS[sev]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{cname} {int(conf * 100)}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        top = max(y1 - th - 12, 0)
        cv2.rectangle(img, (x1, top), (x1 + tw + 8, top + th + 12), color, -1)
        cv2.putText(img, label, (x1 + 4, top + th + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(img, f"Severity: {sev}", (x1, min(y2 + 18, H - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img

def count_boxes(r):
    counts = new_counts()
    severe = 0
    boxes = []
    for box in r.boxes:
        cname = r.names[int(box.cls[0])]
        conf = float(box.conf[0])
        key = canon(cname)
        if key is None or not keep(key, conf): continue
        counts[key] += 1
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        area = (x2 - x1) * (y2 - y1)
        if severity_for(key, area) == "High": severe += 1
        boxes.append({"class": cname, "confidence": round(conf, 2),
                      "severity": severity_for(key, area)})
    return counts, severe, boxes

def extract_gps_bytes(data):
    try:
        img = Image.open(io.BytesIO(data))
        exif = img._getexif()
        if not exif: return None
        gps_raw = None
        from PIL.ExifTags import TAGS as T
        for tag_id, value in exif.items():
            if T.get(tag_id) == 'GPSInfo': gps_raw = value
        if not gps_raw: return None
        gps = {GPSTAGS.get(k, k): v for k, v in gps_raw.items()}
        if 'GPSLatitude' not in gps or 'GPSLongitude' not in gps: return None
        def conv(dms, ref):
            d, m, s = [float(x) for x in dms]
            deg = d + m / 60.0 + s / 3600.0
            if ref in ('S', 'W'): deg = -deg
            return round(deg, 6)
        return {"latitude": conv(gps['GPSLatitude'], gps['GPSLatitudeRef']),
                "longitude": conv(gps['GPSLongitude'], gps['GPSLongitudeRef'])}
    except Exception:
        return None

def calc_priority(potholes, severe):
    if severe >= 3 or potholes >= 3: return "HIGH"
    if severe >= 1 or potholes >= 1: return "MEDIUM"
    return "LOW"

def db_save(R):
    conn = sqlite3.connect("cloud_damage.db")
    conn.execute("""CREATE TABLE IF NOT EXISTS damage_records(
        id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT,
        latitude REAL, longitude REAL, potholes INT, cracks INT,
        severe INT, priority TEXT)""")
    cur = conn.execute(
        "INSERT INTO damage_records(timestamp,latitude,longitude,potholes,cracks,severe,priority) VALUES (?,?,?,?,?,?,?)",
        (R["timestamp"], R.get("gps_lat"), R.get("gps_lon"),
         R["counts"]["pothole"], R["total_cracks"], R["severe_damage"], R["priority"]))
    conn.commit()
    rid = cur.lastrowid
    conn.close()
    return rid

def build_pdf(R):
    from fpdf import FPDF
    c = R["counts"]
    gps = R.get("gps")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "ROAD INSPECTION REPORT", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(6)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(60, 8, "Record ID :", new_x="END")
    pdf.cell(0, 8, str(R["id"]), new_x="LMARGIN", new_y="NEXT")
    pdf.cell(60, 8, "Timestamp :", new_x="END")
    pdf.cell(0, 8, R["timestamp"][:19].replace("T", " "), new_x="LMARGIN", new_y="NEXT")
    pdf.cell(60, 8, "GPS :", new_x="END")
    pdf.cell(0, 8, f"{gps['latitude']}, {gps['longitude']}" if gps
             else "Not embedded in file (see note below)", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(6)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 9, "DETECTED DAMAGE", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    for label, val in [("Potholes", c["pothole"]), ("Alligator cracks", c["alligator"]),
                       ("Longitudinal cracks", c["longitudinal"]), ("Lateral cracks", c["lateral"]),
                       ("Edge cracking", c["edge"]), ("Ravelling", c["ravelling"]),
                       ("Rutting", c["rutting"]), ("Striping", c["striping"]),
                       ("Severe damage", R["severe_damage"])]:
        pdf.cell(120, 8, label, new_x="END")
        pdf.cell(0, 8, str(val), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 9, f"Estimated Repair Priority: {R['priority']}", new_x="LMARGIN", new_y="NEXT")
    if not gps:
        pdf.ln(4)
        pdf.set_font("Helvetica", "I", 10)
        pdf.multi_cell(0, 6, "Note: No GPS metadata was embedded in the uploaded file. "
                       "In field deployment, coordinates are automatically recorded by the "
                       "vehicle's GPS receiver at capture time.")
    return bytes(pdf.output())
if st.session_state.get("do_analyze") and st.session_state.get("file_stash"):
    st.session_state["do_analyze"] = False
    name, data, mtype = st.session_state["file_stash"]
    ext = os.path.splitext(name)[1].lower()
    try:
        model = load_model()
        gps = None
        boxes_list = []
        if ext in VIDEO_EXTS:
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tf:
                tf.write(data)
                src = tf.name
            cap = cv2.VideoCapture(src)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            raw = src + "_raw.avi"
            writer = None
            frame_max = new_counts()
            sev_max = 0
            while True:
                ret, frame = cap.read()
                if not ret: break
                r = model(frame, conf=0.05, verbose=False)[0]
                ann = draw_overlay(frame, r)
                if writer is None:
                    h, w = ann.shape[:2]
                    writer = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
                fc, fs, _ = count_boxes(r)
                for k in frame_max: frame_max[k] = max(frame_max[k], fc[k])
                sev_max = max(sev_max, fs)
                writer.write(ann)
            cap.release()
            if writer: writer.release()
            out = src + "_ann.mp4"
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", raw,
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", out],
                           check=True, capture_output=True)
            ann_bytes = open(out, "rb").read()
            for f in (raw, out, src): os.remove(f)
            counts, severe = frame_max, sev_max
            media = "video"
        else:
            gps = extract_gps_bytes(data)
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            r = model(img, conf=0.05, verbose=False)[0]
            ann = draw_overlay(img.copy(), r)
            ann_bytes = cv2.imencode(".jpg", ann)[1].tobytes()
            counts, severe, boxes_list = count_boxes(r)
            media = "image"

        from datetime import datetime
        total_cracks = counts['alligator'] + counts['longitudinal'] + counts['lateral'] + counts['edge']
        R = {
            "timestamp": datetime.now().isoformat(),
            "gps": gps,
            "gps_lat": gps["latitude"] if gps else None,
            "gps_lon": gps["longitude"] if gps else None,
            "media_type": media,
            "ann_bytes": ann_bytes,
            "counts": counts,
            "total_cracks": total_cracks,
            "total_pavement": counts['ravelling'] + counts['rutting'],
            "total_markings": counts['striping'],
            "severe_damage": severe,
            "priority": calc_priority(counts['pothole'], severe),
            "boxes": boxes_list,
        }
        R["id"] = db_save(R)
        st.session_state["result"] = R
    except Exception as e:
        st.session_state["error"] = str(e)

R = st.session_state.get("result")

h1, h2 = st.columns([4, 1])
with h1:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:12px;padding:6px 0 14px 0;">
      <div style="width:42px;height:42px;border-radius:10px;background:#2563eb;
                  display:flex;align-items:center;justify-content:center;font-size:1.3rem;">🗺️</div>
      <div>
        <div style="font-size:1.25rem;font-weight:800;">RoadSense AI</div>
        <div style="font-size:.75rem;color:#8fa3c7;">Automated Inspection System</div>
      </div>
    </div>""", unsafe_allow_html=True)
with h2:
    _R = st.session_state.get("result")
    st.download_button("Export Report",
                       data=build_pdf(_R) if _R else b"No analysis yet.",
                       file_name="road_inspection_report.pdf",
                       mime="application/pdf", type="primary", disabled=_R is None)

if st.session_state.get("error"):
    st.error(st.session_state.pop("error"))
left, right = st.columns([1, 2.2], gap="medium")
with left:
    with st.container(border=True):
        st.markdown('<div class="lbl">Inspection Controls</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload Image/Video",
                                    type=["jpg", "jpeg", "png", "mp4", "avi", "mov", "mkv"])
        if uploaded:
            st.session_state["file_stash"] = (uploaded.name, uploaded.getvalue(), uploaded.type)
        st.button("Run AI Analysis", type="primary", use_container_width=True,
                  on_click=lambda: st.session_state.update(do_analyze=True))
with right:
    with st.container(border=True):
        w1, w2 = st.columns([3, 1])
        w1.markdown('<div class="lbl">Detection Workspace</div>', unsafe_allow_html=True)
        w2.markdown('<div style="text-align:right;"><span class="badge">YOLOv8 Inference</span></div>',
                    unsafe_allow_html=True)
        if R:
            if R["media_type"] == "image":
                st.image(R["ann_bytes"], width="stretch")
            else:
                st.video(R["ann_bytes"], format="video/mp4")
        else:
            st.markdown("""<div style="background:#000;border-radius:8px;height:430px;display:flex;
                align-items:center;justify-content:center;color:#556;">
                Media Overlay Workspace Canvas</div>""", unsafe_allow_html=True)

b1, b2 = st.columns([1.5, 1], gap="medium")
with b1:
    with st.container(border=True):
        st.markdown('<div class="lbl">Class Distribution</div>', unsafe_allow_html=True)
        if R:
            c = R["counts"]
            maxc = max(max(c.values()), 1)
            rows = ""
            any_det = False
            for key, label, color in CLASS_META:
                n = c[key]
                if n > 0:
                    any_det = True
                    pct = int(n / maxc * 100)
                    rows += f"""<div style="margin:12px 0;">
                      <div style="display:flex;justify-content:space-between;font-size:.85rem;">
                        <span>{label}</span>
                        <span style="color:{color};font-weight:700;">{n} detected</span></div>
                      <div style="height:6px;background:rgba(128,128,128,.15);border-radius:3px;margin-top:5px;">
                        <div style="height:6px;width:{pct}%;background:{color};border-radius:3px;"></div>
                      </div></div>"""
            st.markdown(rows if any_det else
                        '<div style="color:#8fa3c7;padding:12px 0;">No damage detected.</div>',
                        unsafe_allow_html=True)
            st.markdown(f"""<div style="margin-top:14px;padding:10px 14px;border-radius:8px;
                background:rgba(128,128,128,.08);border-left:4px solid
                {'#ef4444' if R['priority']=='HIGH' else '#f59e0b' if R['priority']=='MEDIUM' else '#22c55e'};">
                <span style="font-size:.85rem;">Estimated Repair Priority:</span>
                <b> {R['priority']}</b> &nbsp;•&nbsp; Severe zones: <b>{R['severe_damage']}</b></div>""",
                unsafe_allow_html=True)
        else:
            st.markdown('<div style="color:#556;padding:30px 0;text-align:center;">Awaiting analysis…</div>',
                        unsafe_allow_html=True)
with b2:
    with st.container(border=True):
        g1, g2 = st.columns([2, 1])
        g1.markdown('<div class="lbl">Geospatial Mapping</div>', unsafe_allow_html=True)
        g2.markdown('<div style="text-align:right;color:#22c55e;font-size:.68rem;">● GPS EXIF Sync</div>',
                    unsafe_allow_html=True)
        if R and R.get("gps"):
            lat, lon = R["gps"]["latitude"], R["gps"]["longitude"]
            st.markdown(f"""<div style="background:rgba(128,128,128,.08);border-radius:8px;
                padding:8px 12px;font-size:.8rem;margin:8px 0;">
                Coordinates: <b>{lat}, {lon}</b></div>""", unsafe_allow_html=True)
            m = folium.Map(location=[lat, lon], zoom_start=16, tiles=None)
            folium.TileLayer("cartodbdark_matter", attr="© OpenStreetMap © CARTO").add_to(m)
            main_damage = "Pothole Detected" if R["counts"]["pothole"] else \
                          "Crack Detected" if R["total_cracks"] else "Damage Detected"
            folium.Marker([lat, lon],
                          popup=folium.Popup(f"<b>{main_damage}</b><br>Severity: {R['priority']}",
                                             max_width=200)).add_to(m)
            st_folium(m, height=260, use_container_width=True)
        elif R:
            st.markdown("""<div style="color:#8fa3c7;font-size:.85rem;padding:20px 0;">
                📭 No GPS metadata embedded in this file.<br><br>
                <span style="color:#556;">In-vehicle deployment mode: coordinates are
                auto-tagged by the GPS receiver in real time.</span></div>""", unsafe_allow_html=True)
        else:
            st.markdown('<div style="color:#556;padding:30px 0;text-align:center;">Awaiting analysis…</div>',
                        unsafe_allow_html=True)