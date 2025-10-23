#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import json
import uuid
import shutil
from datetime import datetime, timedelta, timezone
from calendar import monthrange

import pandas as pd
import streamlit as st

# 페이지 설정
st.set_page_config(page_title="자기계발 트래커 / 일정 리마인더", page_icon="⏱️", layout="wide")

# Optional deps
try:
    import requests
except Exception:
    requests = None

# Supabase
try:
    from supabase import create_client, Client as SupabaseClient
except Exception:
    SupabaseClient = None

# (화면에서는 캘린더 표시 안 씀)
try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
except Exception:
    service_account = None
    build = None

# =============================
# 경로 & 상수
# =============================
APP_DIR = os.path.join(".", ".habit_tracker")
TRACKS_CSV = os.path.join(APP_DIR, "tracks.csv")
STATE_JSON  = os.path.join(APP_DIR, "running.json")         # running only (CSV 백엔드)
GOALS_JSON  = os.path.join(APP_DIR, "goals.json")           # goals only (CSV 백엔드, 시간단위 float)
CATEGORIES_JSON = os.path.join(APP_DIR, "categories.json")
REMINDERS_CSV   = os.path.join(APP_DIR, "reminders.csv")

DEFAULT_CATEGORIES = ["공부", "운동", "독서", "글쓰기", "외국어", "명상"]

EN2KR = {
    "study": "공부",
    "workout": "운동",
    "reading": "독서",
    "writing": "글쓰기",
    "language": "외국어",
    "meditation": "명상",
}
KST = timezone(timedelta(hours=9))
os.makedirs(APP_DIR, exist_ok=True)

def ensure_files():
    if not os.path.exists(TRACKS_CSV):
        with open(TRACKS_CSV, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["start_iso", "end_iso", "minutes", "category", "note"])
    if not os.path.exists(CATEGORIES_JSON):
        with open(CATEGORIES_JSON, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_CATEGORIES, f, ensure_ascii=False, indent=2)
    if not os.path.exists(REMINDERS_CSV):
        with open(REMINDERS_CSV, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "id", "title", "category", "note",
                "due_iso", "advance_minutes", "repeat", "active",
                "last_fired_iso"
            ])
ensure_files()

# =============================
# 공통 유틸
# =============================
def now(): return datetime.now(KST)
def iso(dt: datetime) -> str: return dt.astimezone(KST).isoformat(timespec="seconds")
def parse_iso(s: str) -> datetime: return datetime.fromisoformat(s).astimezone(KST)
def fmt_minutes(mins: int):
    h, m = mins // 60, mins % 60
    return f"{h}h {m}m" if h else f"{m}m"

def to_kst_series(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s, errors="coerce")
    if getattr(s.dtype, "tz", None) is None:
        return s.dt.tz_localize("Asia/Seoul", nonexistent="NaT", ambiguous="NaT")
    else:
        return s.dt.tz_convert("Asia/Seoul")

# =============================
# STORAGE LAYER (csv / sqlite / supabase)
# =============================
BACKEND = st.secrets.get("STORAGE_BACKEND", "csv").lower()

# --- SQLite
import sqlite3
SQLITE_PATH = st.secrets.get("SQLITE_PATH", os.path.join(APP_DIR, "data.db"))
def sqlite_conn():
    os.makedirs(os.path.dirname(SQLITE_PATH), exist_ok=True)
    return sqlite3.connect(SQLITE_PATH, check_same_thread=False)
def sqlite_init():
    conn = sqlite_conn(); cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS categories(name TEXT PRIMARY KEY)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS tracks(
            id TEXT PRIMARY KEY,
            start_iso TEXT, end_iso TEXT, minutes INTEGER,
            category TEXT, note TEXT
        )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS reminders(
            id TEXT PRIMARY KEY,
            title TEXT, category TEXT, note TEXT,
            due_iso TEXT, advance_minutes INTEGER,
            repeat TEXT, active INTEGER,
            last_fired_iso TEXT
        )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS state(key TEXT PRIMARY KEY, value TEXT)""")
    cur.execute("SELECT COUNT(*) FROM categories")
    if cur.fetchone()[0] == 0:
        cur.executemany("INSERT OR IGNORE INTO categories(name) VALUES (?)",
                        [(c,) for c in DEFAULT_CATEGORIES])
    conn.commit(); return conn

# --- Supabase
SUPABASE_URL = st.secrets.get("SUPABASE_URL")
SUPABASE_KEY = st.secrets.get("SUPABASE_SERVICE_KEY") or st.secrets.get("SUPABASE_ANON_KEY")
try:
    _supabase: SupabaseClient | None = create_client(SUPABASE_URL, SUPABASE_KEY) \
        if BACKEND == "supabase" and SUPABASE_URL and SUPABASE_KEY and SupabaseClient else None
except Exception:
    _supabase = None

def use_csv(): return BACKEND == "csv" or BACKEND not in ("sqlite", "supabase")
def use_sqlite(): return BACKEND == "sqlite"
def use_supabase(): return BACKEND == "supabase" and _supabase is not None

# =============================
# 카테고리
# =============================
def load_categories() -> list[str]:
    if use_supabase():
        data = _supabase.table("categories").select("name").execute().data or []
        cats = [r["name"] for r in data]
        return cats or DEFAULT_CATEGORIES
    if use_sqlite():
        conn = sqlite_init()
        rows = conn.execute("SELECT name FROM categories ORDER BY name").fetchall()
        return [r[0] for r in rows] or DEFAULT_CATEGORIES
    try:
        with open(CATEGORIES_JSON, "r", encoding="utf-8") as f:
            cats = json.load(f)
            return cats if isinstance(cats, list) else DEFAULT_CATEGORIES
    except Exception:
        return DEFAULT_CATEGORIES

def save_categories(cats: list[str]):
    cats = sorted(set(cats))
    if use_supabase():
        _supabase.table("categories").delete().neq("name", "").execute()
        if cats: _supabase.table("categories").insert([{"name": c} for c in cats]).execute()
        return
    if use_sqlite():
        conn = sqlite_init(); cur = conn.cursor()
        cur.execute("DELETE FROM categories")
        cur.executemany("INSERT INTO categories(name) VALUES (?)", [(c,) for c in cats])
        conn.commit(); return
    with open(CATEGORIES_JSON, "w", encoding="utf-8") as f:
        json.dump(cats, f, ensure_ascii=False, indent=2)

def migrate_categories_to_korean():
    cats = load_categories()
    save_categories([EN2KR.get(c, c) for c in cats])

    # tracks 변환
    if use_supabase():
        data = _supabase.table("tracks").select("id,category").execute().data or []
        for row in data:
            nc = EN2KR.get(row.get("category"), row.get("category"))
            if nc != row.get("category"):
                _supabase.table("tracks").update({"category": nc}).eq("id", row["id"]).execute()
        return
    if use_sqlite():
        conn = sqlite_init(); cur = conn.cursor()
        rows = cur.execute("SELECT id, category FROM tracks").fetchall()
        for i, c in rows:
            nc = EN2KR.get(c, c)
            if nc != c: cur.execute("UPDATE tracks SET category=? WHERE id=?", (nc, i))
        conn.commit(); return
    if os.path.exists(TRACKS_CSV):
        df = pd.read_csv(TRACKS_CSV, encoding="utf-8")
        if not df.empty and "category" in df.columns:
            df["category"] = df["category"].apply(lambda c: EN2KR.get(str(c), c))
            shutil.copy2(TRACKS_CSV, TRACKS_CSV + ".bak")
            df[["start_iso","end_iso","minutes","category","note"]].to_csv(
                TRACKS_CSV, index=False, encoding="utf-8"
            )

# =============================
# 트래킹 데이터 + 편집/삭제
# =============================
def read_state():
    if use_supabase():
        data = _supabase.table("state").select("value").eq("key","running").limit(1).execute().data
        return json.loads(data[0]["value"]) if data else None
    if use_sqlite():
        conn = sqlite_init()
        row = conn.execute("SELECT value FROM state WHERE key='running'").fetchone()
        return json.loads(row[0]) if row else None
    if not os.path.exists(STATE_JSON): return None
    try:
        with open(STATE_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def write_state(data):
    if use_supabase():
        _supabase.table("state").upsert({"key":"running","value": json.dumps(data, ensure_ascii=False)}).execute(); return
    if use_sqlite():
        conn = sqlite_init()
        conn.execute("INSERT OR REPLACE INTO state(key,value) VALUES('running',?)",
                     (json.dumps(data, ensure_ascii=False),))
        conn.commit(); return
    with open(STATE_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def state_get(key: str, default=None):
    if use_supabase():
        data = _supabase.table("state").select("value").eq("key", key).limit(1).execute().data
        return json.loads(data[0]["value"]) if data else default
    if use_sqlite():
        conn = sqlite_init()
        row = conn.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default
    # CSV 백엔드: 별도 goals.json 사용(시간단위 float)
    path = GOALS_JSON if key == "goals" else STATE_JSON
    if not os.path.exists(path): return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def state_set(key: str, value):
    if use_supabase():
        _supabase.table("state").upsert({"key": key, "value": json.dumps(value, ensure_ascii=False)}).execute(); return
    if use_sqlite():
        conn = sqlite_init()
        conn.execute("INSERT OR REPLACE INTO state(key,value) VALUES(?,?)",
                     (key, json.dumps(value, ensure_ascii=False)))
        conn.commit(); return
    path = GOALS_JSON if key == "goals" else STATE_JSON
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)

def append_track(start_dt, end_dt, category, note=""):
    minutes = int(round((end_dt - start_dt).total_seconds() / 60.0))
    if minutes <= 0: raise ValueError("종료 시간이 시작 시간보다 같거나 빠릅니다.")
    if use_supabase():
        _supabase.table("tracks").insert({
            "id": str(uuid.uuid4()),
            "start_iso": iso(start_dt), "end_iso": iso(end_dt),
            "minutes": minutes, "category": category, "note": note
        }).execute(); return minutes
    if use_sqlite():
        conn = sqlite_init()
        conn.execute("INSERT INTO tracks(id,start_iso,end_iso,minutes,category,note) VALUES(?,?,?,?,?,?)",
                     (str(uuid.uuid4()), iso(start_dt), iso(end_dt), minutes, category, note))
        conn.commit(); return minutes
    with open(TRACKS_CSV, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([iso(start_dt), iso(end_dt), str(minutes), category, note])
    return minutes

def read_all_tracks_df() -> pd.DataFrame:
    # 반환: 공통 컬럼 + row_id(수정/삭제용 식별자)
    if use_supabase():
        data = _supabase.table("tracks").select("*").order("start_iso", desc=True).execute().data or []
        df = pd.DataFrame(data)
        if df.empty: return df
        df["row_id"] = df["id"].astype(str)
        df["start_iso"] = pd.to_datetime(df["start_iso"]); df["end_iso"] = pd.to_datetime(df["end_iso"])
        df["start"] = df["start_iso"]; df["end"] = df["end_iso"]
        df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce").fillna(0).astype(int)
        return df
    if use_sqlite():
        conn = sqlite_init()
        df = pd.read_sql_query("SELECT * FROM tracks ORDER BY start_iso DESC", conn)
        if df.empty: return df
        df["row_id"] = df["id"].astype(str)
        df["start_iso"] = pd.to_datetime(df["start_iso"]); df["end_iso"] = pd.to_datetime(df["end_iso"])
        df["start"] = df["start_iso"]; df["end"] = df["end_iso"]
        df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce").fillna(0).astype(int)
        return df
    # CSV
    df = pd.read_csv(TRACKS_CSV, encoding="utf-8")
    if df.empty: return df
    df["row_id"] = df.index.astype(str)  # 파일 내 행 인덱스
    df["start_iso"] = pd.to_datetime(df["start_iso"]); df["end_iso"] = pd.to_datetime(df["end_iso"])
    df["start"] = df["start_iso"]; df["end"] = df["end_iso"]
    df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce").fillna(0).astype(int)
    df = df.sort_values("start", ascending=False)
    return df

def delete_tracks(row_ids: list[str]) -> int:
    if not row_ids: return 0
    if use_supabase():
        for rid in row_ids:
            _supabase.table("tracks").delete().eq("id", rid).execute()
        return len(row_ids)
    if use_sqlite():
        conn = sqlite_init()
        cur = conn.cursor()
        cur.executemany("DELETE FROM tracks WHERE id=?", [(rid,) for rid in row_ids])
        conn.commit(); return len(row_ids)
    # CSV
    df = pd.read_csv(TRACKS_CSV, encoding="utf-8")
    df["__idx"] = df.index.astype(str)
    keep = ~df["__idx"].isin(row_ids)
    kept = df[keep].drop(columns="__idx")
    kept.to_csv(TRACKS_CSV, index=False, encoding="utf-8")
    return int((~keep).sum())

def update_track(row_id: str, new_category: str, new_minutes: int, new_note: str) -> bool:
    # end_iso = start_iso + minutes 로 재계산
    if use_supabase():
        row = _supabase.table("tracks").select("start_iso").eq("id", row_id).limit(1).execute().data
        if not row: return False
        start_dt = pd.to_datetime(row[0]["start_iso"]).to_pydatetime().astimezone(KST)
        end_dt = start_dt + timedelta(minutes=int(new_minutes))
        _supabase.table("tracks").update({
            "category": new_category, "minutes": int(new_minutes),
            "note": new_note, "end_iso": iso(end_dt)
        }).eq("id", row_id).execute()
        return True
    if use_sqlite():
        conn = sqlite_init()
        row = conn.execute("SELECT start_iso FROM tracks WHERE id=?", (row_id,)).fetchone()
        if not row: return False
        start_dt = datetime.fromisoformat(row[0]).astimezone(KST)
        end_dt = start_dt + timedelta(minutes=int(new_minutes))
        conn.execute("""UPDATE tracks
                        SET category=?, minutes=?, note=?, end_iso=?
                        WHERE id=?""",
                     (new_category, int(new_minutes), new_note, iso(end_dt), row_id))
        conn.commit(); return True
    # CSV
    df = pd.read_csv(TRACKS_CSV, encoding="utf-8")
    df["__idx"] = df.index.astype(str)
    if row_id not in set(df["__idx"]): return False
    i = df.index[df["__idx"] == row_id][0]
    start_dt = datetime.fromisoformat(df.at[i, "start_iso"]).astimezone(KST)
    end_dt = start_dt + timedelta(minutes=int(new_minutes))
    df.at[i, "category"] = new_category
    df.at[i, "minutes"]  = int(new_minutes)
    df.at[i, "note"]     = new_note
    df.at[i, "end_iso"]  = iso(end_dt)
    df.drop(columns="__idx").to_csv(TRACKS_CSV, index=False, encoding="utf-8")
    return True

# =============================
# 리마인더
# =============================
REPEAT_CHOICES = ["없음", "매일", "매주", "매월"]

def load_reminders_df() -> pd.DataFrame:
    if use_supabase():
        data = _supabase.table("reminders").select("*").execute().data or []
        df = pd.DataFrame(data)
        if df.empty: return df
        for col in ["due_iso","last_fired_iso"]:
            if col in df.columns: df[col] = pd.to_datetime(df[col], errors="coerce")
        df["active"] = df["active"].astype(bool)
        df["advance_minutes"] = pd.to_numeric(df["advance_minutes"], errors="coerce").fillna(0).astype(int)
        return df
    if use_sqlite():
        conn = sqlite_init(); df = pd.read_sql_query("SELECT * FROM reminders", conn)
        if df.empty: return df
        for col in ["due_iso","last_fired_iso"]:
            df[col] = pd.to_datetime(df[col], errors="coerce")
        df["active"] = df["active"].astype(bool)
        df["advance_minutes"] = pd.to_numeric(df["advance_minutes"], errors="coerce").fillna(0).astype(int)
        return df
    df = pd.read_csv(REMINDERS_CSV, encoding="utf-8")
    if df.empty: return df
    for col in ["due_iso","last_fired_iso"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    df["active"] = df["active"].astype(bool)
    df["advance_minutes"] = pd.to_numeric(df["advance_minutes"], errors="coerce").fillna(0).astype(int)
    return df

def save_reminders_df(df: pd.DataFrame):
    if use_supabase():
        _supabase.table("reminders").delete().neq("id","").execute()
        out = df.copy()
        for c in ["due_iso","last_fired_iso"]:
            if c in out.columns: out[c] = out[c].apply(lambda x: x.isoformat() if pd.notna(x) else None)
        rows = out.to_dict(orient="records")
        if rows: _supabase.table("reminders").insert(rows).execute()
        return
    if use_sqlite():
        conn = sqlite_init(); cur = conn.cursor()
        cur.execute("DELETE FROM reminders")
        out = df.copy()
        for c in ["due_iso","last_fired_iso"]:
            if c in out.columns: out[c] = out[c].apply(lambda x: x.isoformat() if pd.notna(x) else None)
        rows = out.to_dict(orient="records")
        for r in rows:
            cur.execute("""INSERT INTO reminders(id,title,category,note,due_iso,advance_minutes,repeat,active,last_fired_iso)
                           VALUES(?,?,?,?,?,?,?,?,?)""",
                        (r["id"], r["title"], r.get("category"), r.get("note"),
                         r.get("due_iso"), int(r.get("advance_minutes",0)), r.get("repeat"),
                         1 if bool(r.get("active", True)) else 0, r.get("last_fired_iso")))
        conn.commit(); return
    out = df.copy()
    for c in ["due_iso","last_fired_iso"]:
        out[c] = out[c].apply(lambda x: x.isoformat() if pd.notna(x) else "")
    out.to_csv(REMINDERS_CSV, index=False, encoding="utf-8")

def add_reminder(title: str, category: str | None, note: str, due_dt: datetime,
                 advance_minutes: int = 0, repeat: str = "없음", active: bool = True):
    rid = str(uuid.uuid4())
    row = {
        "id": rid, "title": title, "category": category, "note": note,
        "due_iso": due_dt, "advance_minutes": int(advance_minutes),
        "repeat": repeat, "active": bool(active), "last_fired_iso": pd.NaT
    }
    df = load_reminders_df()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    save_reminders_df(df); return rid

def compute_next_due(due_dt: datetime, repeat: str) -> datetime | None:
    if repeat == "없음": return None
    if repeat == "매일": return due_dt + timedelta(days=1)
    if repeat == "매주": return due_dt + timedelta(weeks=1)
    if repeat == "매월":
        y, m = due_dt.year, due_dt.month
        ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
        day = min(due_dt.day, monthrange(ny, nm)[1])
        return due_dt.replace(year=ny, month=nm, day=day)
    return None

def should_fire(row, now_dt: datetime):
    if not row["active"]: return False
    due = row["due_iso"]
    if pd.isna(due): return False
    adv = int(row.get("advance_minutes", 0))
    window_start = due - timedelta(minutes=adv)
    last = row.get("last_fired_iso", pd.NaT)
    if pd.notna(last) and (window_start <= last <= due + timedelta(minutes=5)): return False
    return now_dt >= window_start

def mark_fired(df: pd.DataFrame, rid: str, fired_dt: datetime):
    idx = df.index[df["id"] == rid]
    if len(idx) == 0: return df
    i = idx[0]
    df.at[i, "last_fired_iso"] = pd.to_datetime(fired_dt.isoformat())
    repeat = df.at[i, "repeat"]; due = df.at[i, "due_iso"]
    if pd.notna(due):
        nxt = compute_next_due(due.to_pydatetime().astimezone(KST), repeat)
        if nxt is None: df.at[i, "active"] = False
        else: df.at[i, "due_iso"] = pd.to_datetime(nxt.isoformat())
    return df

def send_slack(title: str, body: str) -> bool:
    if requests is None: return False
    url = st.secrets.get("SLACK_WEBHOOK_URL")
    if not url: return False
    try:
        r = requests.post(url, json={"text": f":alarm_clock: *{title}*\n{body}"}, timeout=5)
        return 200 <= r.status_code < 300
    except Exception:
        return False

# =============================
# 목표(주/월) 저장/불러오기 — 시간 단위(float) 저장
# =============================
def load_goals():
    # 내부 저장은 {"weekly": {cat: hours_float}, "monthly": {...}} 구조
    goals = state_get("goals", default={"weekly": {}, "monthly": {}})
    cats = load_categories()
    for c in cats:
        goals["weekly"].setdefault(c, 0.0)   # 시간
        goals["monthly"].setdefault(c, 0.0)  # 시간
    # 구(분) → 신(시간) 마이그레이션 가능성 처리
    # 값이 큰 정수(예: 600, 1200 등)면 분일 확률↑ → 시간으로 변환
    def normalize(gmap: dict):
        out = {}
        for k, v in gmap.items():
            try:
                x = float(v)
            except Exception:
                x = 0.0
            # 분 단위로 저장된 흔적(>=180 분=3h 이상)을 자동 변환
            if x >= 180 and abs(round(x/60)*60 - x) < 1e-6:
                out[k] = round(x/60.0, 2)
            else:
                out[k] = x
        return out
    goals["weekly"]  = normalize(goals.get("weekly", {}))
    goals["monthly"] = normalize(goals.get("monthly", {}))
    return goals

def save_goals(goals: dict):
    state_set("goals", goals)

# =============================
# 공통 함수(기간, 요약)
# =============================
def daterange_start_end(kind: str):
    now_kst = now()
    if kind == "오늘":
        start = now_kst.replace(hour=0, minute=0, second=0, microsecond=0); end = start + timedelta(days=1)
    elif kind == "어제":
        end = now_kst.replace(hour=0, minute=0, second=0, microsecond=0); start = end - timedelta(days=1)
    elif kind == "이번 주":
        weekday = now_kst.isoweekday()
        start = now_kst.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=weekday-1); end = start + timedelta(days=7)
    elif kind == "이번 달":
        start = now_kst.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        y, m = start.year, start.month
        end = start.replace(year=y+1, month=1) if m == 12 else start.replace(month=m+1)
    elif kind == "전체":
        start = datetime(1970,1,1,tzinfo=KST); end = datetime(2999,1,1,tzinfo=KST)
    else:
        raise ValueError("지원하지 않는 기간")
    return start, end

def summarize(df: pd.DataFrame, start: datetime, end: datetime):
    if df.empty: return {}, 0
    s = pd.to_datetime(start.isoformat()); e = pd.to_datetime(end.isoformat())
    df = df.copy()
    df["overlap_start"] = df["start"].clip(lower=s)
    df["overlap_end"] = df["end"].clip(upper=e)
    mins = ((df["overlap_end"] - df["overlap_start"]).dt.total_seconds() / 60).clip(lower=0)
    df["overlap_minutes"] = mins.round().astype(int)
    by_cat = df.groupby("category")["overlap_minutes"].sum().to_dict()
    total = int(df["overlap_minutes"].sum())
    return by_cat, total

# =============================
# 페이지: 트래커
# =============================
def render_tracker_page():
    st.title("⏱️ 자기계발 시간 트래커")
    st.caption("KST 기준 · CSV/SQLite/Supabase 영속 · 타이머/수동기록 · 최근 기록/요약 2분할 + 필터/편집/삭제 + 목표 게이지(시간 단위)")

    if "running" not in st.session_state: st.session_state.running = read_state()

    col1, col2 = st.columns([2, 3], gap="large")

    # --- 실시간 타이머
    with col1:
        st.subheader("실시간 타이머")
        with st.container(border=True):
            running = st.session_state.running
            if running:
                cat = running["category"]; start_iso = running["start_iso"]
                note = running.get("note", ""); start_dt = parse_iso(start_iso)
                elapsed_min = int((now() - start_dt).total_seconds() // 60)
                st.write(f"**진행 중**: [{cat}] {start_iso} 시작")
                st.write(f"경과: **{elapsed_min}분**")
                if note: st.write(f"메모: {note}")
                stop_note = st.text_input("종료 시 메모(옵션)", value=note, key="stop_note")
                if st.button("🛑 세션 종료/기록"):
                    try:
                        minutes = append_track(start_dt, now(), cat, stop_note)
                        write_state(None)
                        st.session_state.running = None
                        st.success(f"세션 종료: [{cat}] {minutes}분 기록")
                    except Exception as e:
                        st.error(f"기록 실패: {e}")
            else:
                cats = load_categories()
                start_cat = st.selectbox("카테고리", options=sorted(cats) if cats else ["공부"])
                start_note = st.text_input("메모(옵션)", "", key="start_note")
                if st.button("▶️ 세션 시작"):
                    state = {"category": start_cat, "start_iso": iso(now()), "note": start_note}
                    write_state(state); st.session_state.running = state
                    st.success(f"세션 시작: [{start_cat}] {state['start_iso']}")

    # --- 수동 입력
    with col2:
        st.subheader("수동 입력(분 단위)")
        with st.container(border=True):
            cats = load_categories()
            add_cat = st.selectbox("카테고리 선택", options=sorted(cats) if cats else ["공부"], key="add_cat")
            add_min = st.number_input("분(1 이상)", min_value=1, step=5, value=30)
            add_note = st.text_input("메모", "", key="add_note")
            if st.button("➕ 기록 추가"):
                try:
                    end_dt = now(); start_dt = end_dt - timedelta(minutes=int(add_min))
                    append_track(start_dt, end_dt, add_cat, add_note)
                    st.success(f"
