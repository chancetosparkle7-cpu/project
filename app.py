#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, csv, json, uuid
from datetime import datetime, timedelta, timezone
import pandas as pd
import streamlit as st

# =========================
# 기본 설정
# =========================
st.set_page_config(page_title="자기계발 트래커 / 일정 리마인더", page_icon="⏱️", layout="wide")

APP_DIR = os.path.join(".", ".habit_tracker")
TRACKS_CSV = os.path.join(APP_DIR, "tracks.csv")
REMINDERS_CSV = os.path.join(APP_DIR, "reminders.csv")
CATEGORIES_JSON = os.path.join(APP_DIR, "categories.json")
GOALS_JSON = os.path.join(APP_DIR, "goals.json")
STATE_JSON = os.path.join(APP_DIR, "running.json")

DEFAULT_CATEGORIES = ["공부", "운동", "독서", "글쓰기", "외국어", "명상"]
KST = timezone(timedelta(hours=9))

os.makedirs(APP_DIR, exist_ok=True)

def now(): return datetime.now(KST)
def iso(dt): return dt.astimezone(KST).isoformat(timespec="seconds")

# =========================
# 파일 초기화
# =========================
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

# =========================
# 카테고리
# =========================
def load_categories():
    try:
        with open(CATEGORIES_JSON, "r", encoding="utf-8") as f:
            c = json.load(f)
            return c if isinstance(c, list) else DEFAULT_CATEGORIES
    except Exception:
        return DEFAULT_CATEGORIES

def save_categories(cats):
    cats = sorted(set(cats))
    with open(CATEGORIES_JSON, "w", encoding="utf-8") as f:
        json.dump(cats, f, ensure_ascii=False, indent=2)

# =========================
# 상태/목표 (목표는 분으로 저장)
# =========================
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
    base = {"weekly": {}, "monthly": {}}
    if os.path.exists(GOALS_JSON):
        try:
            with open(GOALS_JSON, "r", encoding="utf-8") as f:
                base = json.load(f) or base
        except Exception:
            pass
    # 카테고리 누락 보정
    for c in load_categories():
        base["weekly"].setdefault(c, 0)   # 저장 값은 "분"
        base["monthly"].setdefault(c, 0)
    return base

def save_goals(goals):
    with open(GOALS_JSON, "w", encoding="utf-8") as f:
        json.dump(goals, f, ensure_ascii=False, indent=2)

# =========================
# 트래킹 데이터
# =========================
def append_track(start_dt, end_dt, category, note=""):
    mins = int((end_dt - start_dt).total_seconds() / 60)
    if mins <= 0: mins = 1
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

# =========================
# 리마인더 (간단)
# =========================
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

# =========================
# 기간/요약 유틸
# =========================
def week_range_kst(dt: datetime):
    # ISO 주: 월요일 시작
    weekday = dt.isoweekday()
    start = dt.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=weekday-1)
    end = start + timedelta(days=7)
    return start, end

def month_range_kst(dt: datetime):
    start = dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if start.month == 12:
        end = start.replace(year=start.year+1, month=1)
    else:
        end = start.replace(month=start.month+1)
    return start, end

def summarize_minutes_by_cat(df: pd.DataFrame, start: datetime, end: datetime):
    """start~end 사이 겹치는 분을 카테고리별 합산"""
    if df.empty: return {}
    s = pd.to_datetime(start.isoformat()); e = pd.to_datetime(end.isoformat())
    tmp = df.copy()
    tmp["overlap_start"] = tmp["start_iso"].clip(lower=s)
    tmp["overlap_end"] = tmp["end_iso"].clip(upper=e)
    mins = ((tmp["overlap_end"] - tmp["overlap_start"]).dt.total_seconds() / 60).clip(lower=0)
    tmp["overlap_minutes"] = mins.round().astype(int)
    grp = tmp.groupby("category")["overlap_minutes"].sum()
    return grp.to_dict()

# =========================
# 페이지: 트래커
# =========================
def render_tracker_page():
    st.title("⏱️ 자기계발 시간 트래커")

    if "running" not in st.session_state:
        st.session_state.running = read_state()

    c1, c2 = st.columns(2, gap="large")

    # 실시간 타이머
    with c1:
        st.subheader("▶️ 실시간 타이머")
        running = st.session_state.running
        if running:
            cat = running["category"]
            start = datetime.fromisoformat(running["start_iso"])
            elapsed = int((now() - start).total_seconds() / 60)
            st.info(f"[{cat}] 진행 중 · {elapsed}분 경과")
            note = running.get("note","")
            stop_note = st.text_input("메모 (수정 가능)", value=note)
            if st.button("🛑 종료/기록"):
                mins = append_track(start, now(), cat, stop_note)
                write_state(None)
                st.session_state.running = None
                st.success(f"[{cat}] {mins}분 기록 완료")
        else:
            cat = st.selectbox("카테고리", load_categories(), key="start_cat")
            note = st.text_input("메모(옵션)", key="start_note")
            if st.button("시작"):
                st.session_state.running = {"category": cat, "start_iso": iso(now()), "note": note}
                write_state(st.session_state.running)
                st.success(f"[{cat}] 시작!")

    # 수동 입력
    with c2:
        st.subheader("📝 수동 입력")
        m_cat = st.selectbox("카테고리 선택", load_categories(), key="manual_cat")
        m_minutes = st.number_input("시간(분 단위 입력)", min_value=1, step=5, value=30, key="manual_min")
        m_note = st.text_input("메모", key="manual_note")
        if st.button("➕ 추가"):
            end = now(); start = end - timedelta(minutes=int(m_minutes))
            append_track(start, end, m_cat, m_note)
            st.success(f"[{m_cat}] {m_minutes}분 추가")

    st.divider()

    # 최근 기록 & 목표 대비 게이지
    df = read_all_tracks()
    st.subheader("📜 최근 기록")
    if df.empty:
        st.info("기록이 없습니다.")
        return
    view = df.sort_values("start_iso", ascending=False).head(20)
    show = view[["category","minutes","note","start_iso","end_iso"]].rename(
        columns={"category":"카테고리","minutes":"분","note":"메모","start_iso":"시작","end_iso":"종료"}
    )
    st.dataframe(show, use_container_width=True, hide_index=True)
    st.caption("최근 20개 기록")

    st.divider()

    # ===== 목표 대비 진행률 (시간 단위 게이지) =====
    st.subheader("🎯 목표 대비 진행률 (시간 단위 게이지)")
    col_unit, col_hint = st.columns([1,3])
    with col_unit:
        agg_unit = st.radio("집계 단위", ["주", "월"], horizontal=True)
    with col_hint:
        st.caption("주/월 목표는 사이드바에서 '시간' 단위로 설정합니다. 진행률은 각 기간(이번 주/이번 달) 누적 시간을 기준으로 계산합니다.")

    # 기간 계산
    now_k = now()
    if agg_unit == "주":
        start_p, end_p = week_range_kst(now_k)
    else:
        start_p, end_p = month_range_kst(now_k)

    by_cat = summarize_minutes_by_cat(df, start_p, end_p)  # 분 단위 dict
    goals = load_goals()
    goal_map_minutes = goals["weekly"] if agg_unit == "주" else goals["monthly"]

    cats = sorted(load_categories())
    any_data = False
    for c in cats:
        cur_min = int(by_cat.get(c, 0))
        target_min = int(goal_map_minutes.get(c, 0))
        cur_hr = cur_min / 60.0
        target_hr = target_min / 60.0
        if cur_min > 0 or target_min > 0:
            any_data = True
        # 게이지
        pct = 1.0 if target_min <= 0 else min(1.0, cur_min / target_min)
        # 표기: 현재시간/목표시간 (소수 1자리)
        st.write(f"- **{c}**: {cur_hr:.1f}시간 / 목표 {target_hr:.1f}시간")
        st.progress(pct, text=f"{int(pct*100)}%")
    if not any_data:
        st.info("표시할 목표 또는 기록이 없습니다. 사이드바에서 목표를 설정해 보세요!")

# =========================
# 페이지: 리마인더
# =========================
def render_reminder_page():
    st.title("🔔 일정 리마인더")
    st.caption("리마인더 추가 및 확인")

    st.subheader("리마인더 추가")
    r1, r2 = st.columns(2)
    with r1:
        title = st.text_input("제목")
        cat = st.selectbox("카테고리(선택)", ["(없음)"] + load_categories())
        note = st.text_input("메모(옵션)")
    with r2:
        due_date = st.date_input("기한 날짜", value=now().date())
        due_time = st.time_input("기한 시각", value=(now().replace(second=0, microsecond=0)).time())
        advance = st.number_input("사전 알림(분)", min_value=0, step=5, value=10)
        repeat = st.selectbox("반복", ["없음","매일","매주","매월"])
    if st.button("➕ 추가"):
        due_dt = datetime.combine(due_date, due_time).replace(tzinfo=KST)
        add_reminder(title, None if cat=="(없음)" else cat, note, due_dt, advance, repeat)
        st.success("리마인더 추가 완료")

    st.divider()
    df = load_reminders()
    if df.empty:
        st.info("등록된 리마인더가 없습니다.")
    else:
        v = df[["title","category","due_iso","advance_minutes","repeat","active"]].rename(
            columns={"title":"제목","category":"카테고리","due_iso":"기한(KST)","advance_minutes":"사전알림(분)","repeat":"반복","active":"활성"}
        )
        st.dataframe(v, use_container_width=True, hide_index=True)

# =========================
# 사이드바
# =========================
st.sidebar.markdown("## 📂 페이지 이동")
PAGE_TRACKER = "자기계발 시간 트래커"
PAGE_REMINDER = "일정 리마인더"
page = st.sidebar.radio("이동", [PAGE_TRACKER, PAGE_REMINDER], index=0)

# (트래커에서만) 목표 설정 표시
if page == PAGE_TRACKER:
    st.sidebar.header("🎯 목표 설정 (시간 단위)")
    goals = load_goals()

    t1, t2 = st.sidebar.tabs(["주간 목표(시간)", "월간 목표(시간)"])
    with t1:
        weekly_hours = {}
        for c in sorted(load_categories()):
            cur_hr = round(goals["weekly"].get(c, 0) / 60.0, 2)
            weekly_hours[c] = st.number_input(f"{c}", min_value=0.0, step=0.5, value=cur_hr, key=f"goal_w_{c}")
        if st.button("주간 목표 저장"):
            for c, hr in weekly_hours.items():
                goals["weekly"][c] = int(hr * 60)  # 내부 저장은 분
            save_goals(goals)
            st.sidebar.success("주간 목표 저장 완료")

    with t2:
        monthly_hours = {}
        for c in sorted(load_categories()):
            cur_hr = round(goals["monthly"].get(c, 0) / 60.0, 2)
            monthly_hours[c] = st.number_input(f"{c}", min_value=0.0, step=0.5, value=cur_hr, key=f"goal_m_{c}")
        if st.button("월간 목표 저장"):
            for c, hr in monthly_hours.items():
                goals["monthly"][c] = int(hr * 60)  # 내부 저장은 분
            save_goals(goals)
            st.sidebar.success("월간 목표 저장 완료")

    st.sidebar.divider()

# (설정/데이터: 카테고리 관리) → 트래커에서만 보이게
if page == PAGE_TRACKER:
    st.sidebar.title("⚙️ 설정 / 데이터")
    cats = load_categories()
    st.sidebar.write("카테고리:", ", ".join(cats) if cats else "(없음)")
    with st.sidebar.form("cat_form", clear_on_submit=True):
        new_cat = st.text_input("새 카테고리 추가")
        rm_cat = st.multiselect("카테고리 삭제", options=sorted(cats))
        submit_cat = st.form_submit_button("저장")
        if submit_cat:
            changed = False
            if new_cat and new_cat not in cats:
                cats.append(new_cat); changed = True
            for c in rm_cat:
                if c in cats:
                    cats.remove(c); changed = True
            if changed:
                save_categories(cats); st.sidebar.success("카테고리 업데이트 완료")
            else:
                st.sidebar.info("변경사항이 없습니다.")

st.sidebar.divider()

# (백업 섹션은 항상 표시)
st.sidebar.header("📦 데이터 백업")
if os.path.exists(TRACKS_CSV):
    with open(TRACKS_CSV, "rb") as f:
        st.sidebar.download_button("트래킹 CSV 내보내기", f, file_name="tracks.csv")
if os.path.exists(REMINDERS_CSV):
    with open(REMINDERS_CSV, "rb") as f:
        st.sidebar.download_button("리마인더 CSV 내보내기", f, file_name="reminders.csv")

# =========================
# 라우팅
# =========================
if page == PAGE_TRACKER:
    render_tracker_page()
else:
    render_reminder_page()
