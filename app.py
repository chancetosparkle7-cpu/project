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
import matplotlib.pyplot as plt

# 페이지 설정 (한 번만 호출)
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

# (참고) 구글 캘린더 API는 화면 표시는 제거했지만 함수를 남겨둘 수 있어 import만 유지
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
STATE_JSON = os.path.join(APP_DIR, "running.json")
CATEGORIES_JSON = os.path.join(APP_DIR, "categories.json")
REMINDERS_CSV = os.path.join(APP_DIR, "reminders.csv")

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
def fmt_minutes(mins: int): h, m = mins // 60, mins % 60; return f"{h}h {m}m" if h else f"{m}m"

def to_kst_series(s: pd.Series) -> pd.Series:
    """
    Datetime Series를 KST로 안전 변환.
    - tz-naive: KST로 localize
    - tz-aware: KST로 convert
    - NaT/빈값은 그대로 유지
    """
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
    cur.execute("""
        CREATE TABLE IF NOT EXISTS tracks(
            id TEXT PRIMARY KEY,
            start_iso TEXT, end_iso TEXT, minutes INTEGER,
            category TEXT, note TEXT
        )""")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS reminders(
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
# 트래킹 데이터
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

def clear_state():
    if use_supabase():
        _supabase.table("state").delete().eq("key","running").execute(); return
    if use_sqlite():
        conn = sqlite_init(); conn.execute("DELETE FROM state WHERE key='running'"); conn.commit(); return
    if os.path.exists(STATE_JSON): os.remove(STATE_JSON)

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
    if use_supabase():
        data = _supabase.table("tracks").select("*").execute().data or []
        df = pd.DataFrame(data)
        if df.empty: return df
        df["start_iso"] = pd.to_datetime(df["start_iso"]); df["end_iso"] = pd.to_datetime(df["end_iso"])
        df["start"] = df["start_iso"]; df["end"] = df["end_iso"]
        df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce").fillna(0).astype(int)
        df["category"] = df["category"].astype(str); return df
    if use_sqlite():
        conn = sqlite_init(); df = pd.read_sql_query("SELECT * FROM tracks", conn)
        if df.empty: return df
        df["start_iso"] = pd.to_datetime(df["start_iso"]); df["end_iso"] = pd.to_datetime(df["end_iso"])
        df["start"] = df["start_iso"]; df["end"] = df["end_iso"]
        df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce").fillna(0).astype(int)
        return df
    df = pd.read_csv(TRACKS_CSV, encoding="utf-8")
    if df.empty: return df
    df["start"] = pd.to_datetime(df["start_iso"]); df["end"] = pd.to_datetime(df["end_iso"])
    df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce").fillna(0).astype(int)
    return df

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

# (화면에서는 사용하지 않음)
def fetch_calendar_events(start_dt: datetime, end_dt: datetime) -> list[dict]:
    return []

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
# 페이지
# =============================
def render_tracker_page():
    st.title("⏱️ 자기계발 시간 트래커")
    st.caption("KST 기준 · CSV/SQLite/Supabase 영속 · 타이머/수동기록 · 요약/차트")

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
                        clear_state(); st.session_state.running = None
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
                    st.success(f"수동 입력 완료: [{add_cat}] {int(add_min)}분")
                except Exception as e:
                    st.error(f"입력 실패: {e}")

    st.divider()

    # --- 요약 & 차트: 2단 (원형/막대 + 일별 누적/그룹형)
    df = read_all_tracks_df()
    st.subheader("📊 요약 & 📈 차트")
    period = st.selectbox("기간", ["오늘", "어제", "이번 주", "이번 달", "전체"], index=0, key="sum_period")
    start, end = daterange_start_end(period)

    left, right = st.columns([1, 1], gap="large")

    # ---------- 요약 ----------
    with left:
        st.markdown("#### 요약")
        if df.empty:
            st.info("아직 기록이 없습니다.")
            sum_df = pd.DataFrame(columns=["category", "minutes", "formatted"])
        else:
            by_cat, total = summarize(df, start, end)
            st.caption(f"{start.date()} ~ {(end - timedelta(seconds=1)).date()}")
            if total == 0:
                st.write("해당 기간 기록이 없습니다.")
                sum_df = pd.DataFrame(columns=["category", "minutes", "formatted"])
            else:
                sum_df = (
                    pd.DataFrame([{"category": k, "minutes": v} for k, v in by_cat.items()])
                    .sort_values("minutes", ascending=False)
                    .reset_index(drop=True)
                )
                sum_df["formatted"] = sum_df["minutes"].apply(lambda m: fmt_minutes(int(m)))
                st.dataframe(sum_df, use_container_width=True, hide_index=True)
                st.markdown(f"**합계: {fmt_minutes(total)} ({total}분)**")

    # ---------- 차트 ----------
    with right:
        st.markdown("#### 차트")

        if df.empty or sum_df.empty:
            st.info("차트를 표시할 데이터가 없습니다.")
        else:
            # 기간 내로 클립된 데이터(분) 계산
            s, e = pd.to_datetime(start.isoformat()), pd.to_datetime(end.isoformat())
            clip_df = df.copy()
            clip_df["overlap_start"] = clip_df["start"].clip(lower=s)
            clip_df["overlap_end"] = clip_df["end"].clip(upper=e)
            clip_df["minutes_clip"] = (
                (clip_df["overlap_end"] - clip_df["overlap_start"]).dt.total_seconds() / 60
            ).clip(lower=0).round().astype(int)
            clip_df["date"] = (
                pd.to_datetime(clip_df["overlap_start"]).dt.tz_convert("Asia/Seoul").dt.date
            )

            # 1) 차트 유형
            chart_type = st.radio(
                "차트 유형",
                ["원형(비중)", "막대(합계)", "일별 막대(전체/선택)"],
                index=0,
                horizontal=True,
                key="chart_type_selector",
            )

            # ----- 1-a) 원형(비중)
            if chart_type == "원형(비중)":
                fig1, ax1 = plt.subplots()
                ax1.pie(sum_df["minutes"], labels=sum_df["category"], autopct="%1.0f%%")
                ax1.set_title(f"{period} 카테고리 비중")
                st.pyplot(fig1)

            # ----- 1-b) 막대(합계)
            elif chart_type == "막대(합계)":
                fig2, ax2 = plt.subplots()
                ax2.bar(sum_df["category"], sum_df["minutes"])
                ax2.set_xlabel("카테고리"); ax2.set_ylabel("분"); ax2.set_title(f"{period} 카테고리별 합계(분)")
                plt.xticks(rotation=0)
                st.pyplot(fig2)

            # ----- 1-c) 일별 막대(전체/선택) : 누적/그룹형 옵션
            else:
                st.divider()
                st.markdown("##### 일별 막대 옵션")

                cat_pick = st.radio(
                    "대상",
                    options=["(전체)"] + sum_df["category"].tolist(),
                    horizontal=True,
                    key="bar_cat_pick",
                )

                bar_mode = st.radio(
                    "막대 모드",
                    options=["누적", "그룹형"],
                    horizontal=True,
                    key="bar_mode_selector",
                )

                cat_daily = clip_df.groupby(["date", "category"])["minutes_clip"].sum().reset_index()
                pivot = cat_daily.pivot(index="date", columns="category", values="minutes_clip").fillna(0)

                if cat_pick == "(전체)":
                    if pivot.empty:
                        st.info("표시할 데이터가 없습니다.")
                    else:
                        if bar_mode == "누적":
                            fig3, ax3 = plt.subplots()
                            pivot.plot(kind="bar", stacked=True, ax=ax3)
                            ax3.set_xlabel("날짜"); ax3.set_ylabel("분")
                            ax3.set_title(f"{period} 일별 누적 막대(전체)")
                            plt.xticks(rotation=45, ha="right")
                            st.pyplot(fig3)
                        else:
                            fig4, ax4 = plt.subplots()
                            dates = pivot.index.astype(str).tolist()
                            cats = pivot.columns.tolist()
                            x = range(len(dates))
                            width = 0.8 / max(1, len(cats))
                            for i, c in enumerate(cats):
                                ax4.bar(
                                    [xi + (i - (len(cats)-1)/2)*width for xi in x],
                                    pivot[c].values,
                                    width=width,
                                    label=c,
                                )
                            ax4.set_xticks(list(x), dates, rotation=45, ha="right")
                            ax4.set_xlabel("날짜"); ax4.set_ylabel("분")
                            ax4.set_title(f"{period} 일별 그룹형 막대(전체)")
                            ax4.legend(ncol=min(4, len(cats)))
                            st.pyplot(fig4)
                else:
                    sub = pivot[[cat_pick]] if cat_pick in pivot.columns else pd.DataFrame(index=pivot.index)
                    sub = sub.rename(columns={cat_pick: "minutes"}).fillna(0)
                    if sub.empty:
                        st.info("표시할 데이터가 없습니다.")
                    else:
                        fig5, ax5 = plt.subplots()
                        ax5.bar(sub.index.astype(str), sub["minutes"].values)
                        ax5.set_xlabel("날짜"); ax5.set_ylabel("분")
                        ax5.set_title(f"{period} [{cat_pick}] 일별 합계(분)")
                        plt.xticks(rotation=45, ha="right")
                        st.pyplot(fig5)

    st.divider()

    # --- 최근 로그
    st.subheader("📜 최근 기록")
    if df.empty:
        st.info("기록이 없습니다.")
    else:
        df_view = df.copy().sort_values("start", ascending=False)
        df_view = df_view[["category","start_iso","end_iso","minutes","note"]]
        st.dataframe(df_view, use_container_width=True)

def render_reminder_page():
    st.title("🔔 일정 리마인더")
    st.caption("사전 알림 · 반복 · Slack 연동")

    st.markdown("### 리마인더 추가")
    rc1, rc2 = st.columns(2)
    with rc1:
        r_title = st.text_input("제목", placeholder="예: 오늘 독서 30분", key="reminder_title")
        r_cat = st.selectbox("관련 카테고리(옵션)", options=["(없음)"] + sorted(load_categories()))
        r_note = st.text_input("메모(옵션)", "", key="reminder_note")
    with rc2:
        today = now()
        r_date = st.date_input("기한 날짜", value=today.date(), key="rem_date")
        r_time = st.time_input("기한 시각", value=today.replace(second=0, microsecond=0).time(), key="rem_time")
        r_adv  = st.number_input("사전 알림(분)", min_value=0, step=5, value=10, key="rem_adv")
        r_rep  = st.selectbox("반복", REPEAT_CHOICES, index=0, key="rem_repeat")

    if st.button("➕ 리마인더 생성", key="rem_add_btn"):
        if not r_title.strip():
            st.error("제목은 필수입니다.")
        else:
            due_dt = datetime.combine(r_date, r_time).replace(tzinfo=KST)
            add_reminder(
                title=r_title.strip(),
                category=(None if r_cat == "(없음)" else r_cat),
                note=r_note.strip(),
                due_dt=due_dt,
                advance_minutes=int(r_adv),
                repeat=r_rep,
                active=True
            )
            st.success("리마인더가 추가되었습니다.")

    st.divider()
    st.markdown("### 리마인더 목록")
    rem_df = load_reminders_df()
    if rem_df.empty:
        st.info("리마인더가 없습니다.")
    else:
        view = rem_df.copy()
        view["due_local"] = to_kst_series(view["due_iso"])
        view["last_fired_local"] = to_kst_series(view["last_fired_iso"])
        view = view[[
            "id","active","title","category","note",
            "due_local","advance_minutes","repeat","last_fired_local"
        ]].sort_values(["active","due_local"], ascending=[False, True])
        st.dataframe(view, use_container_width=True, hide_index=True)

        st.markdown("#### 선택 항목 관리")
        sel = st.multiselect("리마인더 선택(ID)", options=view["id"].tolist(), key="rem_select")
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("선택 비활성화", key="rem_disable"):
                if sel:
                    rem_df.loc[rem_df["id"].isin(sel), "active"] = False
                    save_reminders_df(rem_df); st.success("비활성화 완료")
                else:
                    st.info("선택된 항목이 없습니다.")
        with c2:
            if st.button("선택 삭제", key="rem_delete"):
                if sel:
                    rem_df = rem_df[~rem_df["id"].isin(sel)]
                    save_reminders_df(rem_df); st.success("삭제 완료")
                else:
                    st.info("선택된 항목이 없습니다.")
        with c3:
            if st.button("선택 즉시 발송(테스트)", key="rem_test_send"):
                now_dt = now(); fired = 0
                for rid in sel:
                    row = rem_df.loc[rem_df["id"] == rid].iloc[0].to_dict()
                    title = row["title"]; due = row["due_iso"]
                    body = f"기한: {due}\n메모: {row.get('note','')}"
                    st.toast(f"🔔 {title}\n{body}")
                    if send_slack(f"[테스트] {title}", body): st.info(f"Slack 전송: {title}")
                    rem_df = mark_fired(rem_df, rid, now_dt); fired += 1
                if fired:
                    save_reminders_df(rem_df); st.success(f"{fired}건 처리")

# -----------------------------
# 라우팅 & 사이드바
# -----------------------------
st.sidebar.markdown("## 📂 페이지")
PAGE_TRACKER = "자기계발 시간 트래커"
PAGE_REMINDER = "일정 리마인더"
page = st.sidebar.radio("이동", [PAGE_TRACKER, PAGE_REMINDER], index=0, key="nav_page")

st.sidebar.title("⚙️ 설정 / 데이터")
st.sidebar.caption(f"저장소: **{BACKEND.upper()}**")

cats = load_categories()
with st.sidebar:
    st.header("카테고리")
    st.write(", ".join(sorted(cats)) if cats else "(없음)")
    with st.form("cat_form", clear_on_submit=True):
        new_cat = st.text_input("카테고리 추가", "", key="cat_add")
        rm_cat = st.multiselect("카테고리 삭제", options=sorted(cats), key="cat_rm")
        submitted_cat = st.form_submit_button("저장")
        if submitted_cat:
            changed = False
            if new_cat and new_cat not in cats:
                cats.append(new_cat); changed = True
            for c in rm_cat:
                if c in cats:
                    cats.remove(c); changed = True
            if changed:
                save_categories(cats); st.success("카테고리 업데이트 완료")
            else:
                st.info("변경사항이 없습니다.")
    if st.button("🔤 카테고리 한글로 통일"):
        migrate_categories_to_korean(); st.success("카테고리/기록을 한글로 변환했습니다!")

    st.divider()
    st.header("데이터 백업 (CSV)")
    if os.path.exists(TRACKS_CSV):
        with open(TRACKS_CSV, "rb") as f:
            st.download_button("CSV 내보내기(트래킹)", f, file_name="tracks.csv", mime="text/csv")
    if os.path.exists(REMINDERS_CSV):
        with open(REMINDERS_CSV, "rb") as f:
            st.download_button("CSV 내보내기(리마인더)", f, file_name="reminders.csv", mime="text/csv")

# 라우팅
if page == PAGE_TRACKER:
    render_tracker_page()
else:
    render_reminder_page()

# =============================
# 리마인더 감지 & 자동 새로고침
# =============================
def scan_and_fire():
    rem_df = load_reminders_df()
    if rem_df.empty: return
    now_dt = now(); fired_any = False
    for _, row in rem_df.iterrows():
        rowd = row.to_dict()
        if should_fire(rowd, now_dt):
            title = rowd["title"]; due = rowd["due_iso"]
            adv = int(rowd.get("advance_minutes", 0))
            when = "마감 임박" if now_dt < due else "마감 도래"
            body = f"{when} · 기한: {due}\n사전알림: {adv}분\n메모: {rowd.get('note','')}"
            st.toast(f"🔔 {title}\n{body}")
            if send_slack(title, body): st.info(f"Slack 전송: {title}")
            rem_df = mark_fired(rem_df, rowd["id"], now_dt); fired_any = True
    if fired_any: save_reminders_df(rem_df)

scan_and_fire()
st.markdown("<script>setTimeout(() => window.location.reload(), 60*1000);</script>", unsafe_allow_html=True)
st.caption("💡 리마인더는 *앱이 열려 있을 때* 1분 간격으로 감지/발송됩니다. 저장소는 CSV/SQLite/Supabase 중 선택 가능해요.")
