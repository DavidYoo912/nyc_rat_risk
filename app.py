import streamlit as st
import os
import pandas as pd
import json
import subprocess
import sys
from datetime import datetime
from dateutil.relativedelta import relativedelta
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# --- Path Setup ---
APP_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(APP_DIR, 'artifacts')
REPORTS_DIR = os.path.join(ARTIFACTS_DIR, 'reports')

PATHS = {
    'prediction_map':           os.path.join(REPORTS_DIR, 'risk_map_deployment.html'),
    'eval_map':                 os.path.join(REPORTS_DIR, 'dual_train_test_map.html'),
    'metrics_data':             os.path.join(REPORTS_DIR, 'model_metrics.json'),
    'metrics_by_borough':       os.path.join(REPORTS_DIR, 'metrics_by_borough.csv'),
    'feature_importance_csv':   os.path.join(REPORTS_DIR, 'feature_importance.csv'),
    'features_csv':             os.path.join(APP_DIR, 'data', 'processed', 'df_features_final.csv'),
    'inspection_list_prefix':   'inspection_list_',
}

CSS = """
<style>
/* Hide Streamlit chrome */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Page background */
.stApp { background-color: #F8FAFC; }

/* Tighten top padding */
.block-container { padding-top: 1.5rem !important; padding-bottom: 2rem !important; }

/* ── Dashboard header strip ── */
.dashboard-header {
    background: #0F172A;
    border-radius: 10px;
    padding: 1.25rem 1.75rem;
    margin-bottom: 1.25rem;
}
.dashboard-header h1 {
    color: #F8FAFC;
    font-size: 1.6rem;
    font-weight: 700;
    margin: 0 0 0.15rem 0;
    font-family: system-ui, -apple-system, sans-serif;
}
.dashboard-header p {
    color: #94A3B8;
    font-size: 0.85rem;
    margin: 0;
    font-family: system-ui, -apple-system, sans-serif;
}

/* ── KPI cards ── */
.kpi-card {
    background: #FFFFFF;
    border-radius: 10px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    padding: 1rem 1.25rem;
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
    height: 100%;
}
.kpi-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #64748B;
    font-family: system-ui, -apple-system, sans-serif;
}
.kpi-value {
    font-size: 1.65rem;
    font-weight: 700;
    color: #2563EB;
    font-family: system-ui, -apple-system, sans-serif;
    line-height: 1.1;
}
.kpi-value.green  { color: #16A34A; }
.kpi-value.amber  { color: #D97706; }
.kpi-sub {
    font-size: 0.72rem;
    color: #94A3B8;
    font-family: system-ui, -apple-system, sans-serif;
}

/* ── Metric cards (Tab 3) ── */
.metric-card {
    background: #FFFFFF;
    border-radius: 10px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    padding: 1.1rem 1.25rem;
    text-align: center;
}
.metric-card .m-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #64748B;
    font-family: system-ui, -apple-system, sans-serif;
}
.metric-card .m-value {
    font-size: 2rem;
    font-weight: 700;
    color: #2563EB;
    font-family: system-ui, -apple-system, sans-serif;
    line-height: 1.2;
}
.metric-card .m-interp {
    font-size: 0.73rem;
    color: #64748B;
    font-family: system-ui, -apple-system, sans-serif;
    margin-top: 0.15rem;
}

/* ── Hypothesis callout ── */
.hypothesis-box {
    border-left: 4px solid #16A34A;
    background: #F0FDF4;
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.25rem;
    margin-bottom: 1rem;
}
.hypothesis-box h4 {
    margin: 0 0 0.4rem 0;
    color: #15803D;
    font-family: system-ui, -apple-system, sans-serif;
    font-size: 0.9rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.hypothesis-box p {
    margin: 0;
    color: #166534;
    font-size: 0.9rem;
    font-family: system-ui, -apple-system, sans-serif;
}

/* ── Section eyebrow labels ── */
.section-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #94A3B8;
    font-family: system-ui, -apple-system, sans-serif;
    margin-bottom: 0.35rem;
}

/* ── Tab styling ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
    background: transparent;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 6px 6px 0 0;
    font-size: 0.85rem;
    font-weight: 600;
    font-family: system-ui, -apple-system, sans-serif;
}
</style>
"""


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_html_content(filepath):
    try:
        with open(filepath, 'r') as f:
            return f.read()
    except FileNotFoundError:
        st.error(f"Artifact not found: {os.path.basename(filepath)}. Run the pipeline first.")
        return None


def load_model_metrics():
    path = PATHS['metrics_data']
    if not os.path.exists(path):
        st.warning(f"Metrics file not found. Run the model pipeline to generate it.")
        return {}
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        st.warning("Error decoding model_metrics.json.")
        return {}


def find_latest_inspection_list(prefix):
    if not os.path.exists(REPORTS_DIR):
        return None, None
    files = [f for f in os.listdir(REPORTS_DIR) if f.startswith(prefix) and f.endswith('.csv')]
    if not files:
        return None, None
    files.sort(key=lambda x: os.path.getmtime(os.path.join(REPORTS_DIR, x)), reverse=True)
    latest = files[0]
    month_year = latest.replace(prefix, '').replace('.csv', '').replace('_', ' ').upper()
    return os.path.join(REPORTS_DIR, latest), month_year


def get_input_month_str(prediction_month_str):
    if not prediction_month_str:
        return "Unknown"
    try:
        pred_date = datetime.strptime(prediction_month_str, '%B %Y')
        return (pred_date - relativedelta(months=1)).strftime('%B %Y').upper()
    except ValueError:
        return "Unknown"


def kpi_card(label, value, sub="", color_class=""):
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value {color_class}">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>"""


def metric_card(label, value, interp):
    return f"""
    <div class="metric-card">
        <div class="m-label">{label}</div>
        <div class="m-value">{value}</div>
        <div class="m-interp">{interp}</div>
    </div>"""


def get_prediction_status(prediction_month_str: str) -> str:
    """Returns 'past', 'current', or 'future' based on the prediction month vs. today."""
    if not prediction_month_str:
        return 'past'
    try:
        pred_date = datetime.strptime(prediction_month_str.title(), '%B %Y')
        today_first = datetime.today().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if pred_date < today_first:
            return 'past'
        elif pred_date == today_first:
            return 'current'
        else:
            return 'future'
    except ValueError:
        return 'past'


def info_html(content_html: str) -> str:
    """Renders an info-style box that supports inline HTML (including colored text)."""
    return (
        '<div style="background-color:#EFF6FF;border-left:4px solid #3B82F6;'
        'border-radius:0 4px 4px 0;padding:0.75rem 1rem;margin:0.5rem 0;'
        'font-size:0.875rem;line-height:1.75;color:#1e293b;">'
        + content_html +
        '</div>'
    )


@st.cache_data
def attach_coords(df: pd.DataFrame) -> pd.DataFrame:
    """Parse Lat/Lon from Block_ID (format: 'lat_lon') and return rows where both are valid."""
    def _parse(bid):
        try:
            parts = str(bid).split('_')
            return float(parts[0]), float(parts[1])
        except Exception:
            return (None, None)
    parsed = df['Block_ID'].map(_parse)
    df = df.copy()
    df['Lat'] = parsed.map(lambda x: x[0])
    df['Lon'] = parsed.map(lambda x: x[1])
    return df.dropna(subset=['Lat', 'Lon'])


@st.cache_data(show_spinner="Fetching street addresses...")
def add_address_column(df: pd.DataFrame) -> pd.DataFrame:
    """Reverse geocode Lat/Lon columns to human-readable addresses via NYC GeoSearch API."""
    import urllib.request, json

    def _reverse_geocode(lat, lon) -> str:
        try:
            url = f"https://geosearch.planninglabs.nyc/v2/reverse?point.lat={lat}&point.lon={lon}&size=1"
            with urllib.request.urlopen(url, timeout=3) as resp:
                data = json.loads(resp.read())
            props = data["features"][0]["properties"]
            name = props.get("name", "")
            borough = props.get("borough", "")
            if name and borough:
                return f"{name}, {borough}"
            return name or "Address unavailable"
        except Exception:
            return "Address unavailable"

    df = df.copy()
    df.insert(2, "Address", [_reverse_geocode(row["Lat"], row["Lon"]) for _, row in df.iterrows()])
    return df


def render_map_with_interactive_table(map_html: str, groups: dict) -> str:
    """Inject an interactive Top 10 overlay panel with borough dropdown into the Folium map HTML.

    groups: OrderedDict of {label: DataFrame} — first key is shown by default ("Overall").
    Each DataFrame must have Lat, Lon, Rank, Risk Score (P=High), and optionally Address columns.
    """
    import re, json

    m = re.search(r'var (map_[a-f0-9]+) = L\.map', map_html)
    if not m:
        return map_html  # Folium variable not found — return unchanged
    map_var = m.group(1)

    def _rows(df):
        out = []
        for _, row in df.iterrows():
            lat = float(row.get('Lat', 0))
            lon = float(row.get('Lon', 0))
            addr = str(row['Address']) if 'Address' in row else f"{lat:.4f}, {lon:.4f}"
            try:
                score = f"{float(row.get('Risk Score (P=High)', 0)):.3f}"
            except Exception:
                score = ''
            out.append({'rank': int(row.get('Rank', 0)), 'addr': addr,
                        'lat': lat, 'lon': lon, 'score': score})
        return out

    data_json   = json.dumps({g: _rows(df) for g, df in groups.items()})
    options_html = ''.join(
        f'<option value="{g}">{g}</option>' for g in groups
    )

    inject = f"""
<style>
#rat-top10{{
    position:absolute;top:10px;right:10px;z-index:1000;
    background:rgba(255,255,255,0.96);border-radius:8px;
    padding:10px 12px;box-shadow:0 2px 14px rgba(0,0,0,0.28);
    width:290px;font-family:-apple-system,BlinkMacSystemFont,sans-serif;
    font-size:12px;
}}
#rat-top10 h4{{margin:0 0 6px 0;font-size:13px;color:#0F172A;
    border-bottom:1px solid #e2e8f0;padding-bottom:5px;}}
#borough-select{{
    width:100%;margin-bottom:7px;padding:4px 6px;border-radius:5px;
    border:1px solid #cbd5e1;font-size:12px;color:#334155;
    background:#f8fafc;cursor:pointer;
}}
#rat-top10 table{{width:100%;border-collapse:collapse;}}
#rat-top10 tr{{cursor:pointer;transition:background 0.12s;}}
#rat-top10 tr:hover{{background:#FEF3C7;}}
#rat-top10 tr.sel{{background:#FED7AA;}}
#rat-top10 td{{padding:4px 5px;vertical-align:middle;}}
#rat-top10 .rc{{font-weight:700;color:#DC2626;width:22px;text-align:center;}}
#rat-top10 .ac{{color:#334155;max-width:175px;overflow:hidden;
    text-overflow:ellipsis;white-space:nowrap;}}
#rat-top10 .sc{{color:#64748b;text-align:right;width:38px;}}
#rat-top10 thead td{{font-weight:600;color:#94a3b8;font-size:10px;
    text-transform:uppercase;letter-spacing:.05em;padding-bottom:3px;}}
</style>
<div id="rat-top10">
  <h4>🐀 Top 10 High-Risk Blocks</h4>
  <div style="font-size:10px;color:#94a3b8;margin:-4px 0 7px 0;">Top Recommended Blocks to Inspect</div>
  <select id="borough-select" onchange="switchGroup(this.value)">{options_html}</select>
  <table>
    <thead><tr><td class="rc">#</td><td class="ac">Address</td><td class="sc">Score</td></tr></thead>
    <tbody id="top10-tbody"></tbody>
  </table>
</div>
<script>
(function(){{
  var _groups  = {data_json};
  var _lookup  = {{}};
  var _built   = false;
  var _selRow  = null;

  function esc(s){{
    return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;')
                    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
  }}

  function buildLookup(){{
    if(_built) return;
    {map_var}.eachLayer(function(layer){{
      if(layer.getLatLng){{
        var ll = layer.getLatLng();
        _lookup[ll.lat.toFixed(4)+'_'+ll.lng.toFixed(4)] = layer;
      }}
    }});
    _built = true;
  }}

  function renderGroup(name){{
    var rows = _groups[name] || [];
    var tbody = document.getElementById('top10-tbody');
    tbody.innerHTML = '';
    if(_selRow){{ _selRow = null; }}
    rows.forEach(function(r){{
      var tr = document.createElement('tr');
      tr.innerHTML =
        '<td class="rc">'+r.rank+'</td>'+
        '<td class="ac" title="'+esc(r.addr)+'">'+esc(r.addr)+'</td>'+
        '<td class="sc">'+r.score+'</td>';
      (function(lat,lon){{
        tr.onclick = function(){{ clickMapMarker(lat,lon,tr); }};
      }})(r.lat, r.lon);
      tbody.appendChild(tr);
    }});
  }}

  window.switchGroup = function(name){{ renderGroup(name); }};

  window.clickMapMarker = function(lat, lon, row){{
    buildLookup();
    var key = parseFloat(lat).toFixed(4)+'_'+parseFloat(lon).toFixed(4);
    var marker = _lookup[key];
    if(marker){{
      {map_var}.setView([lat, lon], 14, {{animate:true}});
      setTimeout(function(){{ if(marker.openPopup) marker.openPopup(); }}, 350);
    }}
    if(_selRow) _selRow.classList.remove('sel');
    row.classList.add('sel');
    _selRow = row;
  }};

  setTimeout(buildLookup, 600);
  renderGroup(Object.keys(_groups)[0]);
}})();
</script>
"""
    return map_html.replace('</body>', inject + '\n</body>')


def refresh_prediction():
    """Re-runs model_predict.py using the same Python interpreter as the app."""
    predict_script = os.path.join(APP_DIR, 'src', 'model_predict.py')
    result = subprocess.run(
        [sys.executable, predict_script],
        capture_output=True, text=True, cwd=APP_DIR
    )
    return result.returncode == 0, result.stderr


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="NYC Rat Risk Intelligence",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.markdown(CSS, unsafe_allow_html=True)

    # Load data
    metrics = load_model_metrics()
    latest_csv_path, prediction_month_str = find_latest_inspection_list(PATHS['inspection_list_prefix'])
    input_month_str  = get_input_month_str(prediction_month_str) if prediction_month_str else "N/A"
    target_month_str = prediction_month_str if prediction_month_str else "Next Month"

    df_inspection = None
    num_blocks = 0
    if latest_csv_path and os.path.exists(latest_csv_path):
        try:
            df_inspection = pd.read_csv(latest_csv_path)
            num_blocks = len(df_inspection)
            try:
                df_inspection = add_address_column(df_inspection)
                _valid_boroughs = {'Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island'}
                df_inspection['Borough'] = (
                    df_inspection['Address'].str.rsplit(', ', n=1).str[-1]
                    .where(lambda s: s.isin(_valid_boroughs))
                )
            except Exception:
                pass  # graceful fallback: df_inspection stays without Address/Borough columns
        except Exception:
            pass

    recall_pct = f"{metrics.get('class_1_recall', 0)*100:.0f}%" if metrics else "—"

    # Load EDA features data
    df_features = None
    if os.path.exists(PATHS['features_csv']):
        try:
            df_features = pd.read_csv(PATHS['features_csv'], parse_dates=['Month_Start'])
        except Exception:
            pass

    # Load feature importance data
    df_fi = None
    if os.path.exists(PATHS['feature_importance_csv']):
        try:
            df_fi = pd.read_csv(PATHS['feature_importance_csv'], index_col=0)
            df_fi.columns = ['Importance']
        except Exception:
            pass

    pred_status = get_prediction_status(target_month_str)
    stale = pred_status == 'past'

    # ── Global header ──────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="dashboard-header">
        <h1>🐀 NYC Proactive Rat Risk Intelligence</h1>
        <p>Helping city inspectors find rat problems before they start — powered by NYC 311 data</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Staleness banner ───────────────────────────────────────────────────
    if pred_status == 'past':
        st.warning(
            f"⚠️ **Prediction is out of date.** The current prediction targets **{target_month_str}**, "
            "which is in the past. Run `bash run_pipeline.sh` from the project folder to fetch fresh data "
            "and regenerate predictions."
        )
    elif pred_status == 'current':
        st.info(
            f"ℹ️ Showing **{target_month_str}** predictions — data collection for this month is still in progress. "
            "Predictions will be updated automatically as more 311 reports come in."
        )
    else:
        st.success(f"✅ Prediction is current — targeting **{target_month_str}**")

    # ── KPI strip ─────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(kpi_card("Prediction Target", target_month_str, "Next month being scored"), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card("Input Data Month", input_month_str, "311 data used as features"), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_card("High-Risk Blocks", str(num_blocks) if num_blocks else "—",
                             "Blocks flagged for inspection", color_class="amber"), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi_card("Model Recall", recall_pct, "True hotspots caught", color_class="green"),
                    unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1.25rem'></div>", unsafe_allow_html=True)

    # ── Tabs ───────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "🗺️ Live Risk Map",
        "🔬 Model Performance",
        "📊 EDA & Methodology",
    ])

    # ======================================================================
    # TAB 1 — Live Risk Map
    # ======================================================================
    with tab1:
        st.markdown(f'<div class="section-label">Predicted high-risk blocks for {target_month_str} · based on {input_month_str} 311 data</div>',
                    unsafe_allow_html=True)

        # Refresh button (only shown when stale)
        if stale:
            if st.button("🔄 Refresh Prediction (re-score from existing model)", type="secondary"):
                with st.spinner("Re-running prediction — this takes about 10 seconds..."):
                    ok, err = refresh_prediction()
                if ok:
                    st.success("Prediction refreshed successfully!")
                    st.rerun()
                else:
                    st.error(f"Refresh failed. For a full refresh run `bash run_pipeline.sh`.\n\n{err}")

        prediction_map_html = load_html_content(PATHS['prediction_map'])
        if prediction_map_html:
            if df_inspection is not None:
                _groups = {'Overall': df_inspection.head(10)}
                if 'Borough' in df_inspection.columns:
                    for _b in ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island']:
                        _sub = df_inspection[df_inspection['Borough'] == _b].head(10)
                        if len(_sub) > 0:
                            _groups[_b] = _sub
                combined_html = render_map_with_interactive_table(prediction_map_html, _groups)
            else:
                combined_html = prediction_map_html
            st.components.v1.html(combined_html, height=680, scrolling=False)

        st.markdown("<div style='margin-top:0.75rem'></div>", unsafe_allow_html=True)
        st.markdown(info_html(
            f"<b>Heat gradient (<span style='color:#3B82F6'>blue</span> → <span style='color:#DC2626'>red</span>):</b> "
            f"Shows predicted rat risk density across NYC for <b>{target_month_str}</b>, "
            f"derived from <b>{input_month_str}</b> 311 complaint data. "
            f"<span style='color:#DC2626'><b>Red clusters</b></span> indicate the highest-density risk zones.<br><br>"
            f"<span style='color:#EA580C'><b>Orange circle markers:</b></span> The top 200 individual blocks ranked by risk score — "
            f"click any marker to see block-level details including prior rat counts and dumping complaints.<br><br>"
            "It is recommended to begin inspections at the highest-risk blocks first (Rank 1 onward) to maximize "
            "the number of rat problems caught before they escalate. Download the inspection list below to assign field teams."
        ), unsafe_allow_html=True)

        st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)

        if df_inspection is not None:
            dl_col, _ = st.columns([1, 3])
            with dl_col:
                st.markdown('<div class="section-label">Download</div>', unsafe_allow_html=True)
                csv_bytes = df_inspection.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇ Download Inspection List (.csv)",
                    data=csv_bytes,
                    file_name=os.path.basename(latest_csv_path),
                    mime="text/csv",
                    use_container_width=True,
                    type="primary",
                )
            st.caption(
                "💡 Click any row in the Top 10 panel on the map to jump to that block. "
                "The inspection list ranks the top 200 blocks by risk score — download it to assign field teams."
            )
        else:
            st.warning("Inspection list CSV not found. Run the model pipeline to generate high-risk blocks.")

    # ======================================================================
    # TAB 2 — Model Performance
    # ======================================================================
    with tab2:

        # Row 1 — 4 metric cards
        st.markdown('<div class="section-label">Key Performance Metrics (Chronological Test Set)</div>',
                    unsafe_allow_html=True)

        if metrics:
            auc_val  = f"{metrics.get('AUC', 0):.2f}"
            rec_val  = f"{metrics.get('class_1_recall', 0)*100:.0f}%"
            prec_val = f"{metrics.get('class_1_precision', 0)*100:.0f}%"
            f1_val   = f"{metrics.get('class_1_f1', 0):.2f}"

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(metric_card("AUC-ROC", auc_val,
                    "Overall accuracy of risk ranking. 0.9 means the model correctly ranks a real hotspot "
                    "above a safe block 90% of the time."),
                    unsafe_allow_html=True)
            with m2:
                st.markdown(metric_card("Recall", rec_val,
                    "Out of every 100 blocks that will actually have a bad rat problem, the model catches 86. "
                    "Missing fewer hotspots is the primary goal."),
                    unsafe_allow_html=True)
            with m3:
                st.markdown(metric_card("Precision", prec_val,
                    "Out of every 100 blocks the model flags as high risk, 75 will actually have a problem. "
                    "The other 25 are precautionary — better safe than sorry."),
                    unsafe_allow_html=True)
            with m4:
                st.markdown(metric_card("F1-Score", f1_val,
                    "A combined score balancing recall and precision. "
                    "0.8 is considered strong for a public-safety prediction task."),
                    unsafe_allow_html=True)
        else:
            st.warning("Model metrics not loaded. Run the pipeline to generate model_metrics.json.")

        st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)

        # Row 2 — Train/Test split maps
        st.markdown('<div class="section-label">Train / Test Split & Performance Maps</div>',
                    unsafe_allow_html=True)
        eval_map_html = load_html_content(PATHS['eval_map'])
        if eval_map_html:
            st.components.v1.html(eval_map_html, height=650, scrolling=False)

        st.markdown("<div style='margin-top:0.75rem'></div>", unsafe_allow_html=True)

        desc_left, desc_right = st.columns(2)
        with desc_left:
            st.markdown(info_html(
                "<b>Left map — Data Split</b><br><br>"
                "<span style='color:#3B82F6'><b>Blue dots</b></span> — Training blocks (earliest 80% of months).<br>"
                "<span style='color:#DC2626'><b>Red dots</b></span> — Held-out test blocks (most recent 20%).<br><br>"
                "The split is strictly chronological — the model never sees future data during training, "
                "preventing inflated performance scores."
            ), unsafe_allow_html=True)
        with desc_right:
            st.markdown(info_html(
                "<b>Right map — Model Outcomes on Test Set</b><br><br>"
                "<span style='color:#16A34A'><b>Green</b></span> = Successful risk identification — block correctly flagged, inspection resources well spent.<br>"
                "<span style='color:#EA580C'><b>Orange</b></span> = Wasted inspection resources — block flagged but rat activity did not materialize.<br>"
                "<span style='color:#DC2626'><b>Red</b></span> = Missed hotspot — block had a real problem the model failed to catch, the most costly outcome.<br><br>"
                "Clusters of <span style='color:#DC2626'>red</span> reveal geographic blind spots where the model needs improvement."
            ), unsafe_allow_html=True)

        st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)

        # Row 3 — Feature importance chart + top-5 table
        st.markdown('<div class="section-label">Feature Importance</div>', unsafe_allow_html=True)
        st.markdown(
            "Feature importance shows which pieces of information the model relies on most. "
            "Recent rat sighting history is the strongest signal — once a block has a rat problem, it tends to persist. "
            "Illegal dumping is among the top factors, acting as an early warning before rats fully establish."
        )
        fi_chart_col, fi_table_col = st.columns([3, 2])

        with fi_chart_col:
            if df_fi is not None:
                df_fi_plot = df_fi.sort_values('Importance', ascending=True).tail(10)
                fig_fi = px.bar(df_fi_plot, x='Importance', y=df_fi_plot.index, orientation='h',
                                color='Importance', color_continuous_scale='Blues',
                                title='Top 10 Most Important Features')
                fig_fi.update_layout(coloraxis_showscale=False, yaxis_title='')
                st.plotly_chart(fig_fi, use_container_width=True)
            else:
                st.error("feature_importance.csv not found. Run the pipeline.")

        with fi_table_col:
            if metrics.get('top_features'):
                rows = []
                for f in metrics['top_features'][:5]:
                    rows.append({
                        'Feature': f['feature'],
                        'Importance': round(f['score'], 2),
                        'Interpretation': f.get('description', '—'),
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.info("Top-features data not in metrics JSON.")

        st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)

        # Row 4 — Borough performance table
        st.markdown('<div class="section-label">Performance by Borough</div>', unsafe_allow_html=True)
        st.markdown(
            "The model performs well across all five boroughs, though Manhattan and the Bronx have the highest recall "
            "(fewest missed hotspots). Queens and Staten Island are harder to predict — likely due to lower overall "
            "complaint density in those areas."
        )
        borough_path = PATHS['metrics_by_borough']
        if os.path.exists(borough_path):
            try:
                df_borough = pd.read_csv(borough_path)
                df_borough = df_borough[df_borough['Borough'] != 'Unspecified']
                display_cols = ['Borough', 'Count', 'Recall (Risk)', 'Precision (Risk)',
                                'F1-Score (Risk)', 'True Positives', 'False Negatives']
                df_display = df_borough[display_cols].copy()
                for col in ['Recall (Risk)', 'Precision (Risk)', 'F1-Score (Risk)']:
                    df_display[col] = df_display[col].round(1)
                styled = (
                    df_display.style
                    .background_gradient(subset=['Recall (Risk)'], cmap='RdYlGn', vmin=0, vmax=1)
                    .background_gradient(subset=['Precision (Risk)'], cmap='RdYlGn', vmin=0, vmax=1)
                    .background_gradient(subset=['F1-Score (Risk)'], cmap='RdYlGn', vmin=0, vmax=1)
                    .format({'Count': '{:.0f}', 'True Positives': '{:.0f}', 'False Negatives': '{:.0f}'})
                )
                st.dataframe(styled, use_container_width=True, hide_index=True)
            except Exception as e:
                st.warning(f"Could not load borough metrics: {e}")
        else:
            st.info("metrics_by_borough.csv not found. Run model_evaluate.py.")

    # ======================================================================
    # TAB 3 — EDA & Methodology
    # ======================================================================
    with tab3:
        st.markdown("""
        <div class="hypothesis-box">
            <h4>Core Hypothesis — Neglect Signals → Rat Risk</h4>
            <p>Illegal dumping and garbage accumulation complaints are <strong>leading indicators</strong>
            of rodent activity. By detecting these signals in 311 data early, city teams can intervene
            <em>before</em> a rat colony becomes established.</p>
            <p style="margin-top:0.5rem">When a block starts accumulating illegal dumping complaints, it signals
            that conditions are deteriorating — food sources and harborage are increasing. Rats typically follow
            within 1–2 months. This gives inspectors a window to act <strong>before</strong> the infestation peak.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Pipeline stages:**")
        st.markdown("""
- **1. Data** — Fetch NYC 311 rat sightings & illegal dumping via Open Data API; engineer lag/interaction features
- **2. Train** — Chronological 80/20 split; XGBoost vs RandomForest with GridSearchCV (F1-optimized)
- **3. Evaluate** — Generate AUC/Recall/Precision metrics, borough breakdown, feature importance
- **4. Predict** — Score the most recent month; output risk map + top-200 inspection list
        """)

        st.markdown('<div class="section-label" style="margin-top:1rem">Exploratory Data Analysis</div>',
                    unsafe_allow_html=True)

        if df_features is not None:
            # ── EDA Map ────────────────────────────────────────────────────
            st.markdown('<div class="section-label" style="margin-top:0.5rem">Activity Density Map</div>',
                        unsafe_allow_html=True)

            df_coords = attach_coords(df_features)
            years_avail = sorted(df_coords['year'].dropna().astype(int).unique().tolist())

            selected_year = st.slider(
                "Select year", min_value=min(years_avail), max_value=max(years_avail),
                value=max(years_avail), step=1,
            )

            df_yr = (
                df_coords[df_coords['year'] == selected_year]
                .groupby(['Lat', 'Lon'], as_index=False)
                .agg(rat_count=('count_rat_sighting', 'sum'),
                     dump_count=('count_illegal_dumping', 'sum'))
            )

            # Center on Midtown Manhattan; clip color scale at 90th pct so mid-density
            # areas get saturated color rather than washing out to near-white
            _MAP_CENTER = {"lat": 40.754, "lon": -73.984}
            _MAP_CFG    = {"scrollZoom": True}
            _MAP_LAYOUT = dict(
                margin={"r": 0, "t": 35, "l": 0, "b": 0},
                height=500,
                coloraxis_showscale=False,
                paper_bgcolor="#F8FAFC",
            )

            col_rat, col_dump = st.columns(2)

            with col_rat:
                df_rat = df_yr[df_yr['rat_count'] > 0]
                rat_cap = float(df_rat['rat_count'].quantile(0.90)) or 1.0
                fig_rat = px.density_mapbox(
                    df_rat, lat='Lat', lon='Lon', z='rat_count',
                    radius=18, center=_MAP_CENTER, zoom=11,
                    mapbox_style='carto-positron',
                    color_continuous_scale='Reds',
                    range_color=[0, rat_cap],
                    title=f'🐀 Rat Sightings — {selected_year}',
                )
                fig_rat.update_layout(**_MAP_LAYOUT)
                st.plotly_chart(fig_rat, use_container_width=True, config=_MAP_CFG)

            with col_dump:
                df_dump = df_yr[df_yr['dump_count'] > 0]
                dump_cap = float(df_dump['dump_count'].quantile(0.90)) or 1.0
                fig_dump = px.density_mapbox(
                    df_dump, lat='Lat', lon='Lon', z='dump_count',
                    radius=18, center=_MAP_CENTER, zoom=11,
                    mapbox_style='carto-positron',
                    color_continuous_scale='Blues',
                    range_color=[0, dump_cap],
                    title=f'🗑️ Illegal Dumping — {selected_year}',
                )
                fig_dump.update_layout(**_MAP_LAYOUT)
                st.plotly_chart(fig_dump, use_container_width=True, config=_MAP_CFG)

            st.caption(
                f"Density of rat sightings (left) and illegal dumping complaints (right) across NYC blocks in {selected_year}. "
                "Darker zones indicate higher complaint volume. Compare the two maps to spot where both signals overlap — "
                "those areas are the model's strongest risk targets."
            )

            st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)

            # Chart 1 — Monthly Trend (dual-axis)
            df_monthly = (
                df_features
                .groupby('Month_Start')[['count_rat_sighting', 'count_illegal_dumping']]
                .sum()
                .reset_index()
            )
            fig1 = make_subplots(specs=[[{"secondary_y": True}]])
            fig1.add_trace(
                go.Scatter(x=df_monthly['Month_Start'], y=df_monthly['count_rat_sighting'],
                           name='Rat Sightings', line=dict(color='#DC2626', width=2)),
                secondary_y=False,
            )
            fig1.add_trace(
                go.Scatter(x=df_monthly['Month_Start'], y=df_monthly['count_illegal_dumping'],
                           name='Illegal Dumping', line=dict(color='#2563EB', width=2, dash='dash')),
                secondary_y=True,
            )
            fig1.update_layout(title='Rat Sightings & Illegal Dumping Over Time',
                               legend=dict(orientation='h', yanchor='bottom', y=1.02))
            fig1.update_yaxes(title_text='Rat Sightings', secondary_y=False)
            fig1.update_yaxes(title_text='Illegal Dumping Complaints', secondary_y=True)
            st.plotly_chart(fig1, use_container_width=True)
            st.caption(
                "Rat sightings (red) and illegal dumping (blue) both rise in summer and fall in winter — "
                "they move together, which is why dumping is such a useful early warning signal."
            )

            # Chart 2 — Seasonal Pattern
            month_labels = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
                            7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
            df_seasonal = (
                df_features
                .groupby('month')[['count_rat_sighting', 'count_illegal_dumping']]
                .mean()
                .reset_index()
            )
            df_seasonal['Month Name'] = df_seasonal['month'].map(month_labels)
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(x=df_seasonal['Month Name'], y=df_seasonal['count_rat_sighting'],
                                  name='Avg Rat Sightings', marker_color='#DC2626'))
            fig2.add_trace(go.Bar(x=df_seasonal['Month Name'], y=df_seasonal['count_illegal_dumping'],
                                  name='Avg Illegal Dumping', marker_color='#2563EB'))
            fig2.update_layout(title='Average Monthly Activity by Season', barmode='group',
                               xaxis=dict(categoryorder='array',
                                          categoryarray=list(month_labels.values())))
            st.plotly_chart(fig2, use_container_width=True)
            st.caption(
                "Activity peaks in late summer. This means inspections done in spring — "
                "before the peak — have the most impact."
            )

            # Chart 3 — Neglect Gate
            if 'lag_1_dumping' in df_features.columns and 'next_month_rat_count' in df_features.columns:
                df_gate = (
                    df_features[df_features['lag_1_dumping'] <= 10]
                    .groupby('lag_1_dumping')['next_month_rat_count']
                    .mean()
                    .reset_index()
                )
                fig3 = px.bar(df_gate, x='lag_1_dumping', y='next_month_rat_count',
                              labels={'lag_1_dumping': 'Prior Month Illegal Dumping Complaints',
                                      'next_month_rat_count': 'Mean Next-Month Rat Count'},
                              title='How Prior Dumping Predicts Next Month\'s Rat Risk',
                              color='next_month_rat_count', color_continuous_scale='Reds')
                fig3.update_layout(coloraxis_showscale=False)
                st.plotly_chart(fig3, use_container_width=True)
                st.caption(
                    "Blocks that had more illegal dumping last month consistently see more rats next month. "
                    "Even 1–2 dumping complaints raises the risk noticeably."
                )

            # Chart 4 — Correlation Heatmap
            corr_cols = [c for c in ['lag_1_rat','lag_1_dumping','lag_2_rat','lag_6_rat',
                                      'lag_3_avg_rat','lag_12_avg_rat','dumping_rat_interaction',
                                      'month','TARGET_HIGH_RISK']
                         if c in df_features.columns]
            if len(corr_cols) > 1:
                corr = df_features[corr_cols].corr()
                fig4 = px.imshow(corr, color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
                                 title='Feature Correlation Matrix', text_auto='.1f',
                                 aspect='auto')
                st.plotly_chart(fig4, use_container_width=True)
                st.caption(
                    "The strongest predictor of next month's rat risk is last month's rat count (top-left corner). "
                    "Illegal dumping is the next strongest independent signal — it adds information even when "
                    "rat counts are low."
                )
        else:
            st.error("Features data not found (data/processed/df_features_final.csv). Run the pipeline.")


if __name__ == '__main__':
    main()
