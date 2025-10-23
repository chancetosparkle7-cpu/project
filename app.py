#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, csv, json, uuid, shutil
from datetime import datetime, timedelta, timezone
from calendar import monthrange
import pandas as pd
import streamlit as st

# Streamlit 기본 설정
st.set_page_config(page_title="자기계발 트래커 / 일정 리마인더", page_icon="⏱️", layout="wide")

# ====================================
# 기본 환경 변수 및 파일 경로
# ====================================
APP_DIR = os.path.join(".", ".habit_tracker")
TRACKS_CSV = os.path.join(APP_DIR, "tracks.csv")
REMINDERS_CSV = os.path.join(APP_DIR, "reminders.csv")
CATEGORIES_JSON = os.path.join(APP_DIR, "categories.json")
GOALS_JSON = os.path.join(APP_DIR, "goals.json")
STATE_JSON = os.path.join(APP_DIR, "running.json")

os.makedirs(APP_DIR, exist_ok=True)

DEFAULT_CATEGORIES = ["공부", "운동", "독서", "글쓰기", "외국어", "명상"]
KST = timezone(timedelta(hours=9))

def now(): return datetime.now(KST)
def iso(dt): return dt.astimezone(KST).isoformat(timespec="seconds")

# ====================================
# 파일 초기화
# ====================================
def ensure_files():
    if not os.path.exists(TRACKS_CSV):
        with open(TRACKS_CSV, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["start_iso","end_iso","minutes","category","note"])
    if not os.path.exists(REMINDERS_CSV):
        with open(REMINDERS_CSV, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["id","title","category","note","due_iso","advance_minutes","repeat","active","last_fired_iso"])
    if not os.path.exists(CATEGORIES_JSON):
        with open(CATEGORIES_JSON, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_CATEGORIES, f, ensure_ascii=False, indent=2)
ensure_files()

# ====================================
# 카테고리 로드/저장
# ====================================
def load_categories():
    try:
        with open(CATEGORIES_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return DEFAULT_CATEGORIES

def save_categories(cats):
    cats = sorted(set(cats))
    with open(CATEGORIES_JSON, "w", encoding="utf-8") as f:
        json.dump(cats, f, ensure_ascii=False, indent=2)

# ====================================
# 상태 관리 (러닝, 목표)
# ====================================
def read_state():
    if not os.path.exists(STATE_JSON): return None
    try:
        with open(STATE_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def write_state(data):
    with open(STATE_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_goals():
    if not os.path.exists(GOALS_JSON): return {"weekly": {}, "monthly": {}}
    with open(GOALS_JSON, "r", encoding="utf-8") as f:
        goals = json.load(f)
    for k in ["weekly","monthly"]:
        if k not in goals: goals[k] = {}
    # 모든 카테고리 포함 보정
    for c in load_categories():
        goals["weekly"].setdefault(c, 0)
        goals["monthly"].setdefault(c, 0)
    return goals

def save_goals(goals):
    with open(GOALS_JSON, "w", encoding="utf-8") as f:
        json.dump(goals, f, ensure_ascii=False, indent=2)

# ====================================
# 기록 추가/조회
# ====================================
def append_track(start_dt, end_dt, category, note=""):
    mins = int((end_dt - start_dt).total_seconds() / 60)
    with open(TRACKS_CSV, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([iso(start_dt), iso(end_dt), str(mins), category, note])
    return mins

def read_all_tracks():
    df = pd.read_csv(TRACKS_CSV, encoding="utf-8")
    if df.empty: return df
    df["start_iso"] = pd.to_datetime(df["start_iso"])
    df["end_iso"] = pd.to_datetime(df["end_iso"])
    df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce").fillna(0).astype(int)
    df["row_id"] = df.index.astype(str)
    return df

# ====================================
# 리마인더 로드/저장
# ====================================
def load_reminders():
    df = pd.read_csv(REMINDERS_CSV, encoding="utf-8")
    if df.empty: return df
    df["due_iso"] = pd.to_datetime(df["due_iso"], errors="coerce")
    df["last_fired_iso"] = pd.to_datetime(df["last_fired_iso"], errors="coerce")
    df["advance_minutes"] = pd.to_numeric(df["advance_minutes"], errors="coerce").fillna(0).astype(int)
    df["active"] = df["active"].astype(bool)
    return df

def save_reminders(df):
    out = df.copy()
    for c in ["due_iso","last_fired_iso"]:
        out[c] = out[c].apply(lambda x: x.isoformat() if pd.notna(x) else "")
    out.to_csv(REMINDERS_CSV, index=False, encoding="utf-8")

def add_reminder(title, category, note, due_dt, advance, repeat):
    df = load_reminders()
    rid = str(uuid.uuid4())
    row = {
        "id": rid, "title": title, "category": category,
        "note": note, "due_iso": due_dt, "advance_minutes": int(advance),
        "repeat": repeat, "active": True, "last_fired_iso": pd.NaT
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    save_reminders(df)

# ====================================
# 페이지 1: 트래커
# ====================================
def render_tracker_page():
    st.title("⏱️ 자기계발 시간 트래커")

    if "running" not in st.session_state:
        st.session_state.running = read_state()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("▶️ 실시간 타이머")
        running = st.session_state.running
        if running:
            cat = running["category"]
            start = datetime.fromisoformat(running["start_iso"])
            elapsed = int((now() - start).total_seconds() / 60)
            st.info(f"[{cat}] 진행 중 ({elapsed}분 경과)")
            note = running.get("note","")
            stop_note = st.text_input("메모 수정", value=note)
            if st.button("🛑 종료/기록"):
                mins = append_track(start, now(), cat, stop_note)
                write_state(None)
                st.session_state.running = None
                st.success(f"{cat} {mins}분 기록 완료")
        else:
            cat = st.selectbox("카테고리", load_categories())
            note = st.text_input("메모(옵션)")
            if st.button("시작"):
                st.session_state.running = {"category": cat, "start_iso": iso(now()), "note": note}
                write_state(st.session_state.running)
                st.success(f"{cat} 시작")

    with col2:
        st.subheader("📝 수동 입력")
        cat = st.selectbox("카테고리 선택", load_categories(), key="manual_cat")
        minutes = st.number_input("시간(분 단위)", min_value=1, step=5)
        note = st.text_input("메모", key="manual_note")
        if st.button("추가"):
            end = now()
            start = end - timedelta(minutes=int(minutes))
            append_track(start, end, cat, note)
            st.success(f"{cat} {minutes}분 추가됨")

    st.divider()

    # 기록 요약
    df = read_all_tracks()
    if df.empty:
        st.info("기록이 없습니다.")
        return

    st.subheader("📜 최근 기록")
    st.dataframe(df.tail(20)[["category","minutes","note","start_iso","end_iso"]], use_container_width=True)
    st.caption("최근 20개 기록")

# ====================================
# 페이지 2: 리마인더
# ====================================
def render_reminder_page():
    st.title("🔔 일정 리마인더")
    st.caption("리마인더 추가 및 확인")

    st.subheader("리마인더 추가")
    title = st.text_input("제목")
    cat = st.selectbox("카테고리(선택)", ["(없음)"] + load_categories())
    note = st.text_input("메모(옵션)")
    due_date = st.date_input("기한 날짜", value=now().date())
    due_time = st.time_input("기한 시각", value=now().time())
    advance = st.number_input("사전 알림(분)", min_value=0, step=5, value=10)
    repeat = st.selectbox("반복", ["없음","매일","매주","매월"])
    if st.button("➕ 추가"):
        due_dt = datetime.combine(due_date, due_time).replace(tzinfo=KST)
        add_reminder(title, None if cat=="(없음)" else cat, note, due_dt, advance, repeat)
        st.success("리마인더 추가 완료")

    st.divider()
    df = load_reminders()
    if df.empty:
        st.info("등록된 리마인더 없음")
    else:
        st.dataframe(df[["title","category","due_iso","advance_minutes","repeat","active"]], use_container_width=True)

# ====================================
# 사이드바
# ====================================
st.sidebar.markdown("## 📂 페이지 이동")
PAGE_TRACKER = "자기계발 시간 트래커"
PAGE_REMINDER = "일정 리마인더"
page = st.sidebar.radio("이동", [PAGE_TRACKER, PAGE_REMINDER])

# 트래커 페이지일 때만 목표 설정 표시
if page == PAGE_TRACKER:
    st.sidebar.header("🎯 목표 설정 (시간 단위)")
    goals = load_goals()

    t1, t2 = st.sidebar.tabs(["주간 목표(시간)", "월간 목표(시간)"])
    with t1:
        weekly = {}
        for c in sorted(load_categories()):
            val_hr = round(goals["weekly"].get(c, 0) / 60, 2)
            weekly[c] = st.number_input(f"{c}", min_value=0.0, step=0.5, value=val_hr, key=f"w_{c}")
        if st.button("주간 목표 저장"):
            for c, hr in weekly.items():
                goals["weekly"][c] = int(hr * 60)
            save_goals(goals)
            st.sidebar.success("주간 목표 저장 완료")

    with t2:
        monthly = {}
        for c in sorted(load_categories()):
            val_hr = round(goals["monthly"].get(c, 0) / 60, 2)
            monthly[c] = st.number_input(f"{c}", min_value=0.0, step=0.5, value=val_hr, key=f"m_{c}")
        if st.button("월간 목표 저장"):
            for c, hr in monthly.items():
                goals["monthly"][c] = int(hr * 60)
            save_goals(goals)
            st.sidebar.success("월간 목표 저장 완료")

    st.sidebar.divider()

# 공통: 설정/데이터 관리
st.sidebar.title("⚙️ 설정 / 데이터")
cats = load_categories()
st.sidebar.write("카테고리:", ", ".join(cats))
new_cat = st.sidebar.text_input("새 카테고리 추가")
if st.sidebar.button("추가"):
    if new_cat and new_cat not in cats:
        cats.append(new_cat)
        save_categories(cats)
        st.sidebar.success("추가 완료")

st.sidebar.divider()
st.sidebar.header("📦 데이터 백업")
if os.path.exists(TRACKS_CSV):
    with open(TRACKS_CSV, "rb") as f:
        st.sidebar.download_button("트래킹 CSV 내보내기", f, file_name="tracks.csv")
if os.path.exists(REMINDERS_CSV):
    with open(REMINDERS_CSV, "rb") as f:
        st.sidebar.download_button("리마인더 CSV 내보내기", f, file_name="reminders.csv")

# ====================================
# 라우팅
# ====================================
if page == PAGE_TRACKER:
    render_tracker_page()
else:
    render_reminder_page()
