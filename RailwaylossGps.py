# app.py
"""
for build using nuitka
python -m nuitka "Path\RailwaylossGps.py" --onefile --standalone --follow-imports --remove-output --lto=yes --clang --assume-yes-for-downloads  --windows-console-mode=attach
"""
import security.guard

# ======================================
# example: python "Railway Loss Gps.py"  --base-dir "C:\Users\m.ghorbani\Documents\KML\Saraju-Atashbey" --excel "Table_View_Data.xlsx" --rail "C:\Users\m.ghorbani\Documents\KML\all_railways\all_railways.shp" --kmz "RSRP_Final.kmz"
# ======================================
import argparse
import os
def parse_args():
    parser = argparse.ArgumentParser(
        prog="RailwayLossGps",
        description=(
            "writted by mohamadrezaghorbani.12345@gmail.com"
            "Corrects GPS points using railway geometry.\n\n"
            "INPUTS:\n"
            "  • Excel file with Timestamp, Latitude, Longitude\n"
            "  • Railway SHP (EPSG:3857 or convertible)\n"
            "  • KMZ containing GPS placemarks (same order as Excel)\n\n"
            "OUTPUTS:\n"
            "  • Corrected KML (points snapped/interpolated along railway)\n"
            "  • Debug Excel with flags (bad points, teleport, order issues)\n"
            "  • GeoJSON virtual track (only if missing seconds)\n"
            "  • Folium HTML map for visual validation\n"
            " here is example: --base-dir path --excel Table_View_Data.xlsx --rail path/rail.shp --kmz RSRP_Final.kmz"
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "--base-dir",
        required=True,
        help=(
            "Base working directory.\n"
            "Must contain the Excel file, KMZ file, and output folder.\n\n"
            "Example:\n"
            "  --base-dir \"C:\\Data\\RailwayTest\""
        )
    )

    parser.add_argument(
        "--excel",
        required=True,
        help=(
           "Excel file name (inside base-dir).\n"
           "Must contain columns:\n"
           "  • Date + Time   OR   Timestamp\n"
           "  • Latitude\n"
           "  • Longitude\n\n"
           "Example:\n"
           "  --excel \"Table_View_Data.xlsx\""
       )
    )

    parser.add_argument(
        "--rail",
        required=True,
        help=(
            "Railway shapefile path (.shp).\n"
            "Can be absolute or relative to base-dir.\n"
            "All related files (.shx, .dbf, .prj) must exist.\n\n"
            "Example:\n"
            "  --rail \"all_railways\\all_railways.shp\""
        )
    )

    parser.add_argument(
        "--kmz",
        required=True,
        help=(
           "KMZ or KML file containing GPS points.\n"
           "Placemark order MUST match Excel row order.\n\n"
           "Example:\n"
           "  --kmz \"RSRP_Final.kmz\""
       )
    )

    parser.add_argument(
        "--out-dir",
        default="output",
       help=(
            "Output directory name or full path.\n"
            "Default: output (inside base-dir)\n\n"
            "Outputs created:\n"
            "  • output/kml/*_corrected.kml\n"
            "  • output/debug_flags.xlsx\n"
            "  • output/virtual_track.geojson\n"
            "  • output/folium_map_corrected.html\n"
        )
    )

    return parser.parse_args()


# =============================================
# main code
# =============================================
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.ops import unary_union, linemerge, nearest_points
from shapely.geometry import Point, LineString, MultiLineString
import xml.etree.ElementTree as ET
import folium

# ============================================================
# PARAMETERS
# ============================================================
EXPECTED_STEP_SEC = 1

STUCK_EPS = 1e-5
STUCK_MIN_RUN = 3

SPIKE_DIST_M = 200.0
MAX_DIST_TO_MAIN_RAIL_M = 50.0

TELEPORT_SPEED_MS = 100.0
TELEPORT_MAX_TIME_GAP_SEC = 20.0

S_BACKWARD_TOL = 3.0
# ============================================================
# KMZ Converter
# ============================================================
import zipfile
import tempfile
import shutil
def kmz_to_kml(kmz_path, output_dir=None):
    """
    Extracts the main KML from a KMZ file.
    Returns the path to the extracted KML.
    """

    if not zipfile.is_zipfile(kmz_path):
        raise ValueError(f"Not a valid KMZ file: {kmz_path}")

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="kmz_")

    with zipfile.ZipFile(kmz_path, "r") as zf:
        kml_files = [f for f in zf.namelist() if f.lower().endswith(".kml")]

        if not kml_files:
            raise FileNotFoundError("No KML file found inside KMZ")

        # Convention: first KML is the main document
        kml_name = kml_files[0]
        zf.extract(kml_name, output_dir)

    extracted_kml = os.path.join(output_dir, kml_name)

    # Normalize path (handle subfolders inside KMZ)
    final_kml = os.path.join(
        output_dir,
        os.path.basename(kml_name)
    )

    if extracted_kml != final_kml:
        shutil.move(extracted_kml, final_kml)

    return final_kml

# ============================================================
# KML HELPERS
# ============================================================
def kml_color_to_html_hex(kml_color: str) -> str:
    s = (kml_color or "").strip()
    if len(s) < 8:
        return "#000000"
    aabbggrr = s[-8:]
    bb = aabbggrr[2:4]
    gg = aabbggrr[4:6]
    rr = aabbggrr[6:8]
    return f"#{rr}{gg}{bb}"

def extract_point_colors_from_kml(path: str) -> pd.DataFrame:
    ns = {"kml": "http://www.opengis.net/kml/2.2"}
    tree = ET.parse(path)
    root = tree.getroot()
    doc = root.find("kml:Document", ns)

    style_map = {}
    if doc is not None:
        for style in doc.findall("kml:Style", ns):
            sid = style.attrib.get("id")
            col = style.findtext("kml:IconStyle/kml:color", default="", namespaces=ns)
            style_map[sid] = col

    rows = []
    for pm in root.findall(".//kml:Placemark", ns):
        style_url = pm.findtext("kml:styleUrl", default="", namespaces=ns)
        sid = style_url[1:] if style_url.startswith("#") else style_url or None
        pt_el = pm.find("kml:Point", ns)
        if pt_el is None:
            continue
        coord_text = pt_el.findtext("kml:coordinates", default="", namespaces=ns)
        if not coord_text:
            continue
        parts = coord_text.strip().split(",")
        try:
            lon = float(parts[0]); lat = float(parts[1])
        except Exception:
            continue
        kcol = style_map.get(sid, "")
        rows.append({
            "lon": lon, "lat": lat,
            "style_id": sid,
            "kml_color": kcol,
            "html_color": kml_color_to_html_hex(kcol)
        })
    return pd.DataFrame(rows)

def update_kml_coordinates(orig_kml_path: str, corrected_df: pd.DataFrame, out_path: str) -> bool:
    ns = {"kml": "http://www.opengis.net/kml/2.2"}
    ET.register_namespace("", ns["kml"])
    tree = ET.parse(orig_kml_path)
    root = tree.getroot()
    placemarks = root.findall(".//kml:Placemark", ns)

    if len(placemarks) != len(corrected_df):
        print("Length mismatch:", orig_kml_path,
              "KML points:", len(placemarks),
              "corrected:", len(corrected_df))
        return False

    for i, pm in enumerate(placemarks):
        lon = float(corrected_df.iloc[i]["lon_corrected"])
        lat = float(corrected_df.iloc[i]["lat_corrected"])
        ts  = corrected_df.iloc[i]["timestamp"]

        coord_elem = pm.find(".//kml:Point/kml:coordinates", ns)
        if coord_elem is not None:
            coord_elem.text = f"{lon:.8f},{lat:.8f},0"

        ts_elem = pm.find("kml:TimeStamp", ns)
        if ts_elem is None:
            ts_elem = ET.SubElement(pm, f"{{{ns['kml']}}}TimeStamp")
        when_elem = ts_elem.find("kml:when", ns)
        if when_elem is None:
            when_elem = ET.SubElement(ts_elem, f"{{{ns['kml']}}}when")

        when_elem.text = pd.to_datetime(ts).strftime("%Y-%m-%dT%H:%M:%SZ")

    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    return True

# ============================================================
# CORE HELPERS
# ============================================================
def ensure_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Timestamp" not in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Date"].astype(str) + " " + df["Time"].astype(str), errors="coerce")
    else:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    if df["Timestamp"].isna().any():
        bad = df[df["Timestamp"].isna()].head(10)
        raise ValueError(f"Un-parseable Timestamp rows:\n{bad}")
    return df

def check_missing_seconds_any_order(df: pd.DataFrame) -> pd.DatetimeIndex:
    t0 = df["Timestamp"].min()
    t1 = df["Timestamp"].max()
    expected = pd.date_range(t0, t1, freq="1S")
    have = pd.DatetimeIndex(df["Timestamp"].values)
    return expected.difference(have)

def add_dt_and_seg_in_row_order(df: pd.DataFrame) -> pd.DataFrame:
    """
    Important: do NOT sort. Keep Excel row order (must match KML placemark order).
    Segment breaks whenever dt != 1s or dt<=0.
    """
    out = df.copy()
    out["dt_sec"] = out["Timestamp"].diff().dt.total_seconds()
    out["is_dt1"] = (out["dt_sec"] == float(EXPECTED_STEP_SEC))
    out["dt_ok"]  = (out["dt_sec"] > 0).fillna(True)

    # fully continuous in-order means: dt==1 for every step after first
    steps = out["dt_sec"].iloc[1:]
    out["fully_continuous_in_order"] = bool((steps == 1.0).all() and out["dt_ok"].iloc[1:].all())

    new_seg = (~out["is_dt1"]).fillna(True)
    out["seg_id"] = new_seg.cumsum() - 1
    return out

def stuck_mask(lat: pd.Series, lon: pd.Series) -> np.ndarray:
    lat = pd.to_numeric(lat, errors="coerce").astype("float64")
    lon = pd.to_numeric(lon, errors="coerce").astype("float64")
    same = (lat.diff().abs() < STUCK_EPS) & (lon.diff().abs() < STUCK_EPS)
    run = same.rolling(window=STUCK_MIN_RUN, min_periods=1).sum() >= (STUCK_MIN_RUN - 1)
    return run.fillna(False).to_numpy()

def explode_lines(geom):
    merged = linemerge(geom)
    if isinstance(merged, LineString):
        return [merged]
    if isinstance(merged, MultiLineString):
        return list(merged.geoms)
    if isinstance(geom, LineString):
        return [geom]
    if isinstance(geom, MultiLineString):
        return list(geom.geoms)
    return [merged]

def nearest_line_to_point(p: Point, rail_lines: list[LineString]) -> LineString:
    # simple brute-force; reliable everywhere (slower if you have tons of rail pieces)
    best = None
    best_d = float("inf")
    for ln in rail_lines:
        d = p.distance(ln)
        if d < best_d:
            best_d = d
            best = ln
    return best if best is not None else rail_lines[0]

def choose_best_line_for_two_anchors(p0: Point, p1: Point, rail_lines: list[LineString]) -> LineString:
    """
    Your rule: find the nearest railway LineString for the anchors, and choose the line that fits both.
    (We evaluate 2-3 natural candidates to avoid random far components.)
    """
    if p0 is None or p1 is None:
        return rail_lines[0]

    ln0 = nearest_line_to_point(p0, rail_lines)
    ln1 = nearest_line_to_point(p1, rail_lines)

    # add a midpoint candidate (helps when ln0 != ln1)
    mid = Point((p0.x + p1.x) / 2.0, (p0.y + p1.y) / 2.0)
    ln_mid = nearest_line_to_point(mid, rail_lines)

    candidates = []
    for ln in (ln0, ln1, ln_mid):
        if ln is not None and ln not in candidates:
            candidates.append(ln)

    # choose by minimizing sum of distances to both anchors
    best = candidates[0]
    best_score = p0.distance(best) + p1.distance(best)
    for ln in candidates[1:]:
        sc = p0.distance(ln) + p1.distance(ln)
        if sc < best_score:
            best_score = sc
            best = ln
    return best

def project_to_line(p: Point, line: LineString):
    q = nearest_points(p, line)[1]
    return q, line.project(q), p.distance(q)

def mark_bad_order_by_main_s(seg: pd.DataFrame, s_main: np.ndarray, good_logic: np.ndarray) -> np.ndarray:
    """
    If good points go backwards along s (beyond tolerance), mark the later good point as bad_order.
    Iteratively removes violators.
    """
    bad_order = np.zeros(len(seg), dtype=bool)
    anchors = [i for i in range(len(seg)) if good_logic[i] and np.isfinite(s_main[i])]

    if len(anchors) < 2:
        return bad_order

    while True:
        ds = np.diff([s_main[i] for i in anchors])
        viol = np.where(ds < -S_BACKWARD_TOL)[0]
        if len(viol) == 0:
            break
        j = anchors[int(viol[0]) + 1]
        bad_order[j] = True
        anchors.pop(int(viol[0]) + 1)
        if len(anchors) < 2:
            break
    return bad_order

# ============================================================
# Weighted mean speed (time-weighted)
# ============================================================
def weighted_mean_speed(seg: pd.DataFrame, ref_idx: int, max_speed=80.0):
    speeds = []
    weights = []

    t_ref = seg.at[ref_idx, "Timestamp"]

    for i, row in seg.iterrows():
        if row.get("is_good", False) and np.isfinite(row.get("raw_speed_m_s", np.nan)):
            dt = abs((row["Timestamp"] - t_ref).total_seconds())
            w = 1.0 / max(dt, 1.0)
            v = min(max(row["raw_speed_m_s"], 0.1), max_speed)
            speeds.append(v)
            weights.append(w)

    if not speeds:
        return max_speed * 0.5

    return float(np.average(speeds, weights=weights))


# ============================================================
# Fill bad runs between two good points along the railway
# ============================================================
def correct_one_segment_fill_bad_runs(seg: pd.DataFrame, rail_lines: list[LineString], main_line_for_s: LineString) -> pd.DataFrame:
    """
    Segment = continuous dt==1 block (by row order).
    Steps:
      1) classify points (stuck/teleport/spike/far + time-order)
      2) determine final good anchors
      3) for each adjacent good anchor pair: (G ... B ... G)
         - choose nearest rail LineString that fits both anchors
         - project anchors to that line
         - relocate every interior point along that line between the anchors
      4) handle leading/trailing bad points by holding nearest anchor position on its rail line
    """
    seg = seg.copy().reset_index(drop=True)

    # build 3857 point geometry
    seg["Latitude"]  = pd.to_numeric(seg["Latitude"], errors="coerce")
    seg["Longitude"] = pd.to_numeric(seg["Longitude"], errors="coerce")

    mask = seg["Latitude"].notna() & seg["Longitude"].notna()
    geoms_3857 = [None] * len(seg)
    if mask.any():
        pts = gpd.GeoSeries(
            gpd.points_from_xy(seg.loc[mask, "Longitude"], seg.loc[mask, "Latitude"]),
            crs="EPSG:4326"
        ).to_crs(epsg=3857)
        it = iter(pts.values)
        for i in range(len(seg)):
            if bool(mask.iloc[i]):
                geoms_3857[i] = next(it)
    seg["geom_3857"] = geoms_3857

    # dt order inside segment (should be 1, but still check)
    seg["time_diff"] = seg["Timestamp"].diff().dt.total_seconds().fillna(0.0)
    seg["dt_ok_local"] = (seg["time_diff"] > 0).fillna(True)
    seg.loc[0, "dt_ok_local"] = True

    # classification: stuck
    seg["is_stuck"] = stuck_mask(seg["Latitude"], seg["Longitude"])

    # distance to nearest rail line (for spike/far)
    dist_to_rail = np.full(len(seg), np.nan)
    for i, g in enumerate(geoms_3857):
        if isinstance(g, Point) and (not g.is_empty):
            ln = nearest_line_to_point(g, rail_lines)
            dist_to_rail[i] = g.distance(ln)
    seg["dist_to_main_rail"] = dist_to_rail

    seg["is_spike"] = seg["dist_to_main_rail"] > SPIKE_DIST_M
    seg["is_far"]   = seg["dist_to_main_rail"] > MAX_DIST_TO_MAIN_RAIL_M

    # teleport
    step_dist = np.zeros(len(seg), dtype=float)
    for i in range(1, len(seg)):
        g0, g1 = geoms_3857[i-1], geoms_3857[i]
        if isinstance(g0, Point) and isinstance(g1, Point):
            step_dist[i] = g1.distance(g0)
    seg["step_dist"] = step_dist

    raw_speed = np.full(len(seg), np.nan)
    for i in range(1, len(seg)):
        dt = float(seg.at[i, "time_diff"])
        if dt > 0:
            raw_speed[i] = step_dist[i] / dt
    seg["raw_speed_m_s"] = raw_speed

    is_tp_back = (seg["time_diff"] <= TELEPORT_MAX_TIME_GAP_SEC) & (seg["raw_speed_m_s"] > TELEPORT_SPEED_MS)

    # forward teleport check
    time_diff_fwd = (seg["Timestamp"].shift(-1) - seg["Timestamp"]).dt.total_seconds().fillna(0.0)
    step_dist_fwd = np.zeros(len(seg), dtype=float)
    for i in range(len(seg)-1):
        g0, g1 = geoms_3857[i], geoms_3857[i+1]
        if isinstance(g0, Point) and isinstance(g1, Point):
            step_dist_fwd[i] = g1.distance(g0)
    speed_fwd = np.where(time_diff_fwd > 0, step_dist_fwd / np.where(time_diff_fwd==0, np.nan, time_diff_fwd), np.nan)
    is_tp_fwd = (time_diff_fwd <= TELEPORT_MAX_TIME_GAP_SEC) & (speed_fwd > TELEPORT_SPEED_MS)

    seg["is_teleport"] = (is_tp_back | is_tp_fwd).fillna(False)

    # "logic" bad + time-order bad
    seg["is_bad_logic"] = seg["is_stuck"] | seg["is_teleport"] | seg["is_spike"] | seg["is_far"]
    seg["is_good_logic"] = ~seg["is_bad_logic"]

    seg["is_bad_timeorder"] = ~seg["dt_ok_local"]

    # main s for order checking (single consistent line just for order test)
    s_main = np.full(len(seg), np.nan)
    for i, g in enumerate(geoms_3857):
        if isinstance(g, Point) and (not g.is_empty):
            _, s, _ = project_to_line(g, main_line_for_s)
            s_main[i] = s
    seg["s_main"] = s_main

    bad_order = mark_bad_order_by_main_s(seg, s_main, seg["is_good_logic"].to_numpy())
    seg["is_bad_order"] = bad_order

    seg["is_bad"] = seg["is_bad_logic"] | seg["is_bad_timeorder"] | seg["is_bad_order"]
    seg["is_good"] = ~seg["is_bad"]

    # ------------------------------------------------------------
    # Now the filling: for each (G ... B ... G) relocate B along railway
    # ------------------------------------------------------------
    corrected_geom = [None] * len(seg)

    good_idx = [i for i in range(len(seg)) if seg.at[i, "is_good"] and isinstance(geoms_3857[i], Point)]

    if len(good_idx) == 0:
        # no good anchors => snap everything individually to nearest railway line
        for i, g in enumerate(geoms_3857):
            if isinstance(g, Point):
                ln = nearest_line_to_point(g, rail_lines)
                corrected_geom[i] = nearest_points(g, ln)[1]
        seg["geom_corrected_3857"] = corrected_geom
        return seg

    # leading part before first good anchor: hold at that anchor's nearest rail point
    first = good_idx[0]
    p_first = geoms_3857[first]
    ln_first = nearest_line_to_point(p_first, rail_lines)
    p_first_on = nearest_points(p_first, ln_first)[1]
    s_first = ln_first.project(p_first_on)
    mean_speed = weighted_mean_speed(seg, first)
    # snap the first anchor
    corrected_geom[first] = p_first_on
    for i in range(first - 1, -1, -1):
        dt = (seg.at[first, "Timestamp"] - seg.at[i, "Timestamp"]).total_seconds()
        ds = mean_speed * dt
        s_new = max(s_first - ds, 0.0)
        corrected_geom[i] = ln_first.interpolate(s_new)

   

    # fill between consecutive good anchors
    for a in range(len(good_idx) - 1):
        i0 = good_idx[a]
        i1 = good_idx[a + 1]

        p0 = geoms_3857[i0]
        p1 = geoms_3857[i1]

        # choose the railway LineString that best fits BOTH anchors
        ln = choose_best_line_for_two_anchors(p0, p1, rail_lines)

        p0_on = nearest_points(p0, ln)[1]
        p1_on = nearest_points(p1, ln)[1]
        s0 = ln.project(p0_on)
        s1 = ln.project(p1_on)

        # force anchors onto this same line (so B points connect properly)
        corrected_geom[i0] = p0_on
        corrected_geom[i1] = p1_on

        # interior points (these are typically B points)
        t0 = seg.at[i0, "Timestamp"]
        t1 = seg.at[i1, "Timestamp"]
        total = (t1 - t0).total_seconds()
        if total <= 0:
            total = float(i1 - i0) if (i1 - i0) > 0 else 1.0

        for k in range(i0 + 1, i1):
            tk = seg.at[k, "Timestamp"]
            # time fraction (preferred)
            frac = (tk - t0).total_seconds() / total if total > 0 else (k - i0) / (i1 - i0)
            frac = min(max(frac, 0.0), 1.0)
            sk = s0 + frac * (s1 - s0)
            corrected_geom[k] = ln.interpolate(float(sk))

    # trailing part after last good anchor
    last = good_idx[-1]
    p_last = geoms_3857[last]
    ln_last = nearest_line_to_point(p_last, rail_lines)
    p_last_on = nearest_points(p_last, ln_last)[1]
    s_last = ln_last.project(p_last_on)
    mean_speed = weighted_mean_speed(seg, last)
    corrected_geom[last] = p_last_on
    for i in range(last + 1, len(seg)):
        dt = (seg.at[i, "Timestamp"] - seg.at[last, "Timestamp"]).total_seconds()
        ds = mean_speed * dt
        s_new = min(s_last + ds, ln_last.length)

    # any leftovers (e.g., missing coords) stay None
    seg["geom_corrected_3857"] = corrected_geom
    return seg

def build_track_geometry_strict_rule(df_all: pd.DataFrame):
    """
    Your strict rule:
      - if fully continuous dt==1 in timestamp order: NO line geometry
      - else: MultiLineString with one LineString per seg_id (dt==1 runs)
              made from corrected points (we include all corrected points).
    """
    fully = bool(df_all["fully_continuous_in_order"].iloc[0])
    if fully:
        return None

    lines = []
    for seg_id, part in df_all.groupby("seg_id", sort=True):
        coords = []
        for g in part["geom_corrected_3857"].values:
            if isinstance(g, Point) and (not g.is_empty):
                coords.append((g.x, g.y))
        if len(coords) >= 2:
            lines.append(LineString(coords))

    if not lines:
        return None
    return MultiLineString(lines)

def multiline_to_folium(track_geom_4326, fmap, color="#0000ff", weight=4, opacity=0.9):
    if track_geom_4326 is None:
        return
    if isinstance(track_geom_4326, LineString):
        coords = [(y, x) for x, y in track_geom_4326.coords]
        folium.PolyLine(coords, color=color, weight=weight, opacity=opacity).add_to(fmap)
    elif isinstance(track_geom_4326, MultiLineString):
        for ln in track_geom_4326.geoms:
            coords = [(y, x) for x, y in ln.coords]
            folium.PolyLine(coords, color=color, weight=weight, opacity=opacity).add_to(fmap)

# ============================================================
# RUN
# ============================================================
def main():
    args = parse_args()

    BASE_DIR = os.path.abspath(args.base_dir)

    EXCEL_IN = os.path.join(BASE_DIR, args.excel)

    RAIL_SHP = os.path.abspath(args.rail)

    KML_DIR = BASE_DIR
    

    OUT_DIR = (
        args.out_dir
        if os.path.isabs(args.out_dir)
        else os.path.join(BASE_DIR, args.out_dir)
    )
    KMLS_TO_UPDATE = kmz_to_kml(os.path.join(BASE_DIR,args.kmz), OUT_DIR)

    OUT_KML_DIR = os.path.join(OUT_DIR, "kml")
    OUT_DEBUG_XLSX = os.path.join(OUT_DIR, "debug_flags.xlsx")
    OUT_MISSING_TXT = os.path.join(OUT_DIR, "missing_seconds.txt")
    OUT_MAP_HTML = os.path.join(OUT_DIR, "folium_map_corrected.html")
    OUT_TRACK_GEOJSON = os.path.join(OUT_DIR, "virtual_track.geojson")

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(OUT_KML_DIR, exist_ok=True)
    
    # ---- Excel ----
    df = pd.read_excel(EXCEL_IN)
    df = ensure_timestamp(df)

    # Missing seconds in [min..max] regardless of order (debug)
    missing_seconds = check_missing_seconds_any_order(df)
    with open(OUT_MISSING_TXT, "w", encoding="utf-8") as f:
        f.write(f"Missing seconds count: {len(missing_seconds)}\n")
        for ts in missing_seconds[:5000]:
            f.write(str(ts) + "\n")
    print("Missing seconds:", len(missing_seconds), "| saved:", OUT_MISSING_TXT)

    # dt + segments in row order (Excel↔KML alignment)
    df = add_dt_and_seg_in_row_order(df)
    fully_cont = bool(df["fully_continuous_in_order"].iloc[0])
    print("Fully continuous dt==1 in timestamp order:", fully_cont)

    # ---- Railway load ----
    rail_gdf = gpd.read_file(RAIL_SHP)
    if rail_gdf.crs is None:
        rail_gdf = rail_gdf.set_crs(epsg=3857, allow_override=True)
    else:
        epsg = rail_gdf.crs.to_epsg()
        if epsg != 3857:
            rail_gdf = rail_gdf.to_crs(epsg=3857)

    rail_union = unary_union(rail_gdf.geometry)
    rail_lines = explode_lines(rail_union)

    # Choose a single "main" line ONLY for s-order checking (not for filling)
    # We choose the longest line component as a stable order reference.
    main_line_for_s = max(rail_lines, key=lambda ln: ln.length)

    print("Rail components:", len(rail_lines), "| main-line length (m):", float(main_line_for_s.length))

    # ---- Correct each dt==1 segment independently using your G..B..G fill logic ----
    parts = []
    for seg_id, seg in df.groupby("seg_id", sort=True):
        parts.append(correct_one_segment_fill_bad_runs(seg, rail_lines, main_line_for_s))
    out = pd.concat(parts, ignore_index=True)

    # ---- Track geometry (your strict rule) ----
    track_geom_3857 = build_track_geometry_strict_rule(out)
    print("Track geometry:", None if track_geom_3857 is None else track_geom_3857.geom_type)

    # ---- Convert corrected points to WGS84 ----
    gdf_corr = gpd.GeoDataFrame(out.copy(), geometry=out["geom_corrected_3857"], crs="EPSG:3857").to_crs(epsg=4326)
    gdf_corr["lat_corrected"] = gdf_corr.geometry.y
    gdf_corr["lon_corrected"] = gdf_corr.geometry.x

    # ---- Debug Excel ----
    debug_cols = [
        "Timestamp","Date","Time","Latitude","Longitude",
        "lat_corrected","lon_corrected",
        "dt_sec","seg_id",
        "dist_to_main_rail",
        "is_stuck","is_teleport","is_spike","is_far",
        "is_bad_logic","is_bad_timeorder","is_bad_order","is_bad","is_good",
        "s_main"
    ]
    debug_cols = [c for c in debug_cols if c in gdf_corr.columns]
    gdf_corr[debug_cols].to_excel(OUT_DEBUG_XLSX, index=False)
    print("Saved debug Excel:", OUT_DEBUG_XLSX)

    # ---- Prepare KML update ----
    corr_for_kml = pd.DataFrame({
        "lon_corrected": gdf_corr["lon_corrected"].astype(float),
        "lat_corrected": gdf_corr["lat_corrected"].astype(float),
        "timestamp": gdf_corr["Timestamp"]
    })

    # ---- Update KMLs ----
    corrected_kml_paths = []
    in_path = os.path.join(KML_DIR, KMLS_TO_UPDATE)
    out_path = os.path.join(OUT_KML_DIR, os.path.splitext(KMLS_TO_UPDATE)[0] + "_corrected.kml")
    ok = update_kml_coordinates(in_path, corr_for_kml, out_path)
    print("Updated", KMLS_TO_UPDATE, "=>", ok, "|", out_path)
    if ok:
        corrected_kml_paths = out_path    

    # ---- Save track GeoJSON (only if it exists) ----
    track_geom_4326 = None
    if track_geom_3857 is not None:
        track_gdf = gpd.GeoDataFrame({"name": ["virtual_track"]}, geometry=[track_geom_3857], crs="EPSG:3857").to_crs(epsg=4326)
        track_gdf.to_file(OUT_TRACK_GEOJSON, driver="GeoJSON")
        print("Saved track GeoJSON:", OUT_TRACK_GEOJSON)
        track_geom_4326 = track_gdf.geometry.iloc[0]

    # ---- Folium map ----
   # rsrp_kml_path = os.path.join(KML_DIR, "RSRP_Final.kml")
    color_df = extract_point_colors_from_kml(in_path)

    if len(color_df) == len(gdf_corr):
        colors = color_df["html_color"].values
    else:
        print("WARNING: Final.kml point count != Excel rows. Using black for all points.")
        colors = np.array(["#000000"] * len(gdf_corr))

    mean_lat = float(np.nanmean(gdf_corr["lat_corrected"]))
    mean_lon = float(np.nanmean(gdf_corr["lon_corrected"]))
    fmap = folium.Map(location=[mean_lat, mean_lon], zoom_start=11, tiles="OpenStreetMap")

    fg_points = folium.FeatureGroup(name="Corrected points (colored)")
    fg_bad    = folium.FeatureGroup(name="Bad points (red)")
    fg_order  = folium.FeatureGroup(name="Bad order (orange)")
    fg_track  = folium.FeatureGroup(name="Track (only if missing seconds)")

    for i, row in gdf_corr.iterrows():
        lat = float(row["lat_corrected"])
        lon = float(row["lon_corrected"])
        ts  = row["Timestamp"]

        # is_bad = bool(row.get("is_bad", False))
        # is_bad_order = bool(row.get("is_bad_order", False))
        col = colors[i]

        folium.CircleMarker(
            location=(lat, lon),
            radius=2,
            color=col,
            fill=True,
            fill_color=col,
            fill_opacity=0.9,
            weight=1,
            tooltip=str(ts)
        ).add_to(fg_points)

        # if is_bad:
        #     folium.CircleMarker(
        #         location=(lat, lon),
        #         radius=4,
        #         color="red",
        #         fill=True,
        #         fill_color="red",
        #         fill_opacity=1.0,
        #         weight=1,
        #         tooltip=f"BAD | {ts}"
        #     ).add_to(fg_bad)

        # if is_bad_order:
        #     folium.CircleMarker(
        #         location=(lat, lon),
        #         radius=4,
        #         color="orange",
        #         fill=True,
        #         fill_color="orange",
        #         fill_opacity=1.0,
        #         weight=1,
        #         tooltip=f"BAD ORDER | {ts}"
        #     ).add_to(fg_order)

    # draw track only if NOT fully continuous in-order (your rule)
    if (not fully_cont) and (track_geom_4326 is not None):
        multiline_to_folium(track_geom_4326, fg_track, color="#0000ff", weight=4, opacity=0.9)

    fg_points.add_to(fmap)
    fg_bad.add_to(fmap)
    fg_order.add_to(fmap)
    if (not fully_cont) and (track_geom_4326 is not None):
        fg_track.add_to(fmap)

    folium.LayerControl().add_to(fmap)
    fmap.save(OUT_MAP_HTML)
    print("Saved folium map:", OUT_MAP_HTML)

    print("\nDONE.")
    print("Corrected KMLs:", corrected_kml_paths)

if __name__ == "__main__":
    main()