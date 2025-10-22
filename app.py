#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import json
from datetime import datetime, timedelta, timezone
from collections import defaultdict

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# -----------------------------
# 설정 & 경로
# -----------------------------
APP_DIR = os.path.join(".", ".habit_tracker")  # 리포 루트 기준
TRACKS_CSV = os.path.join(APP_DIR, "tracks.csv")
STATE_JSON = os.path.join(APP_DIR, "running.json")
CATEGORIES_JSON = os.path.join(APP_DIR, "categories.json")

DEFAULT_CATEGORIES = ["study", "workout", "reading", "writing", "language", "meditation"]
KST = timezone(timedelta(hours=9))

os.makedirs(APP_DIR, exist_ok=True)

def ensure_files():
    if not os.path.exists(TRACKS_CSV):
        with open(TRACKS_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["start_iso", "end_iso", "minutes", "category", "note"])
    if not os.path.exists(CATEGORIES_JSON):
        with open(CATEGORIES_JSON, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_CATEGORIES, f, ensure_ascii=False, indent=2)

ensure_files()

# -----------------------------
# 유틸 함수
# -----------------------------
def now():
    return datetime.now(KST)

def iso(dt: datetime) -> str:
    return dt.astimezone(KST).isoformat(timespec="seconds")

def parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s).astimezone(KST)

def load_categories():
    try:
        with open(CATEGORIES_JSON, "r", encoding="utf-8") as f:
            cats = json.load(f)
            return cats if isinstance(cats, list) else DEFAULT_CATEGORIES
    except Exception:
        return DEFAULT_CATEGORIES

def save_categories(cats):
    with open(CATEGORIES_JSON, "w", encoding="utf-8") as f:
        json.dump(sorted(set(cats)), f, ensure_ascii=False, indent=2)

def read_state():
    if not os.path.exists(STATE_JSON):
        return None
    try:
        with open(STATE_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def write_state(data):
    with open(STATE_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def clear_state():
    if os.path.exists(STATE_JSON):
        os.remove(STATE_JSON)

def append_track(start_dt, end_dt, category, note=""):
    minutes = int(round((end_dt - start_dt).total_seconds() / 60.0))
    if minutes <= 0:
        raise ValueError("종료 시간이 시작 시간보다 같거나 빠릅니다.")
    with open(TRACKS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([iso(start_dt), iso(end_dt), str(minutes), category, note])
    return minutes

def read_all_tracks_df() -> pd.DataFrame:
    if not os.path.exists(TRACKS_CSV):
        ensure_files()
    df = pd.read_csv(TRACKS_CSV, encoding="utf-8")
    if df.empty:
        return df
    df["start"] = pd.to_datetime(df["start_iso"])
    df["end"] = pd.to_datetime(df["end_iso"])
    df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce").fillna(0).astype(int)
    return df

def daterange_start_end(kind: str):
    now_kst = now()
    if kind == "오늘":
        start = now_kst.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
    elif kind == "어제":
        end = now_kst.replace(hour=0, minute=0, second=0, microsecond=0)
        start = end - timedelta(days=1)
    elif kind == "이번 주":
        weekday = now_kst.isoweekday()  # 1=월
        start = now_kst.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=weekday-1)
        end = start + timedelta(days=7)
    elif kind == "이번 달":
        start = now_kst.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if start.month == 12:
            end = start.replace(year=start.year+1, month=1)
        else:
            end = start.replace(month=start.month+1)
    elif kind == "전체":
        start = datetime(1970,1,1,tzinfo=KST)
        end = datetime(2999,1,1,tzinfo=KST)
    else:
        raise ValueError("지원하지 않는 기간")
    return start, end

def summarize(df: pd.DataFrame, start: datetime, end: datetime):
    if df.empty:
        return {}, 0
    # 경계 교차 고려하여 겹치는 분만 계산
    s = pd.to_datetime(start.isoformat())
    e = pd.to_datetime(end.isoformat())
    df["overlap_start"] = df["start"].clip(lower=s)
    df["overlap_end"] = df["end"].clip(upper=e)
    mins = ((df["overlap_end"] - df["overlap_start"]).dt.total_seconds() / 60).clip(lower=0)
    df["overlap_minutes"] = mins.round().astype(int)
    by_cat = df.groupby("category")["overlap_minutes"].sum().to_dict()
    total = int(df["overlap_minutes"].sum())
    return by_cat, total

def fmt_minutes(mins: int):
    h = mins // 60
    m = mins % 60
    return f"{h}h {m}m" if h else f"{m}m"

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="자기계발 시간 트래커", page_icon="⏱️", layout="wide")

if "running" not in st.session_state:
    st.session_state.running = read_state()  # 앱 첫 로드시 파일 상태 반영

st.title("⏱️ 자기계발 시간 트래커 (Streamlit)")
st.caption("KST 기준 · CSV 영속 · 시작/중지 · 수동 입력 · 요약/차트 · 데이터 내보내기/가져오기")

# ---- 사이드바: 카테고리 관리 & 데이터 I/O
with st.sidebar:
    st.header("카테고리")
    cats = load_categories()
    st.write(", ".join(sorted(cats)) if cats else "(없음)")
    with st.form("cat_form", clear_on_submit=True):
        new_cat = st.text_input("카테고리 추가", "")
        rm_cat = st.multiselect("카테고리 삭제", options=sorted(cats))
        submitted_cat = st.form_submit_button("저장")
        if submitted_cat:
            changed = False
            if new_cat:
                if new_cat not in cats:
                    cats.append(new_cat)
                    changed = True
            if rm_cat:
                for c in rm_cat:
                    if c in cats:
                        cats.remove(c)
                        changed = True
            if changed:
                save_categories(cats)
                st.success("카테고리 업데이트 완료")
            else:
                st.info("변경사항이 없습니다.")

    st.divider()
    st.header("데이터")
    # 내보내기
    if os.path.exists(TRACKS_CSV):
        with open(TRACKS_CSV, "rb") as f:
            st.download_button("CSV 내보내기", f, file_name="tracks.csv", mime="text/csv")
    # 가져오기
    up = st.file_uploader("CSV 가져오기(열: start_iso,end_iso,minutes,category,note)", type=["csv"])
    if up is not None:
        try:
            new_df = pd.read_csv(up)
            needed = {"start_iso","end_iso","minutes","category","note"}
            if needed.issubset(set(new_df.columns)):
                # 기존 파일 백업
                if os.path.exists(TRACKS_CSV):
                    os.replace(TRACKS_CSV, TRACKS_CSV + ".bak")
                new_df.to_csv(TRACKS_CSV, index=False, encoding="utf-8")
                st.success("CSV 가져오기 완료(이전 파일은 .bak으로 백업)")
            else:
                st.error("CSV 컬럼명이 맞지 않습니다.")
        except Exception as e:
            st.error(f"가져오기 실패: {e}")

st.divider()

# ---- 상단: 실시간 타이머/입력
col1, col2 = st.columns([2, 3], gap="large")

with col1:
    st.subheader("실시간 타이머")
    with st.container(border=True):
        running = st.session_state.running
        if running:
            cat = running["category"]
            start_iso = running["start_iso"]
            note = running.get("note", "")
            start_dt = parse_iso(start_iso)
            elapsed_min = int((now() - start_dt).total_seconds() // 60)
            st.write(f"**진행 중**: [{cat}] {start_iso} 시작")
            st.write(f"경과: **{elapsed_min}분**")
            if note:
                st.write(f"메모: {note}")
            stop_note = st.text_input("종료 시 메모(옵션)", value=note, key="stop_note")
            if st.button("🛑 세션 종료/기록"):
                try:
                    minutes = append_track(start_dt, now(), cat, stop_note)
                    clear_state()
                    st.session_state.running = None
                    st.success(f"세션 종료: [{cat}] {minutes}분 기록")
                except Exception as e:
                    st.error(f"기록 실패: {e}")
        else:
            cats = load_categories()
            start_cat = st.selectbox("카테고리", options=sorted(cats) if cats else ["study"])
            start_note = st.text_input("메모(옵션)", "")
            if st.button("▶️ 세션 시작"):
                state = {"category": start_cat, "start_iso": iso(now()), "note": start_note}
                write_state(state)
                st.session_state.running = state
                st.success(f"세션 시작: [{start_cat}] {state['start_iso']}")

with col2:
    st.subheader("수동 입력(분 단위)")
    with st.container(border=True):
        cats = load_categories()
        add_cat = st.selectbox("카테고리 선택", options=sorted(cats) if cats else ["study"], key="add_cat")
        add_min = st.number_input("분(1 이상)", min_value=1, step=5, value=30)
        add_note = st.text_input("메모", "")
        if st.button("➕ 기록 추가"):
            try:
                end_dt = now()
                start_dt = end_dt - timedelta(minutes=int(add_min))
                append_track(start_dt, end_dt, add_cat, add_note)
                st.success(f"수동 입력 완료: [{add_cat}] {int(add_min)}분")
            except Exception as e:
                st.error(f"입력 실패: {e}")

st.divider()

# ---- 요약/로그/차트
df = read_all_tracks_df()

tabs = st.tabs(["📊 요약", "📜 로그", "📈 차트"])

with tabs[0]:
    period = st.selectbox("기간", ["오늘", "어제", "이번 주", "이번 달", "전체"], index=0)
    start, end = daterange_start_end(period)
    if df.empty:
        st.info("아직 기록이 없습니다.")
    else:
        by_cat, total = summarize(df, start, end)
        st.markdown(f"**{period} 요약**  \n({start.date()} ~ {(end - timedelta(seconds=1)).date()})")
        if total == 0:
            st.write("해당 기간 기록이 없습니다.")
        else:
            # 표
            sum_df = (
                pd.DataFrame([{"category": k, "minutes": v} for k, v in by_cat.items()])
                .sort_values("minutes", ascending=False)
                .reset_index(drop=True)
            )
            sum_df["formatted"] = sum_df["minutes"].apply(lambda m: fmt_minutes(int(m)))
            st.dataframe(sum_df, use_container_width=True, hide_index=True)

            # 파이차트 (matplotlib)
            fig1, ax1 = plt.subplots()
            ax1.pie(sum_df["minutes"], labels=sum_df["category"], autopct="%1.0f%%")
            ax1.set_title(f"{period} 카테고리 비중")
            st.pyplot(fig1)

            st.markdown(f"**합계: {fmt_minutes(total)} ({total}분)**")

with tabs[1]:
    st.markdown("### 최근 기록")
    if df.empty:
        st.info("기록이 없습니다.")
    else:
        df_view = df.copy().sort_values("start", ascending=False)
        df_view = df_view[["category","start_iso","end_iso","minutes","note"]]
        st.dataframe(df_view, use_container_width=True)

with tabs[2]:
    st.markdown("### 일별 합계(막대)")
    if df.empty:
        st.info("기록이 없습니다.")
    else:
        # 일별 합계
        daily = df.copy()
        daily["date"] = pd.to_datetime(daily["start_iso"]).dt.tz_convert("Asia/Seoul").dt.date
        daily_sum = daily.groupby("date")["minutes"].sum().reset_index()
        fig2, ax2 = plt.subplots()
        ax2.bar(daily_sum["date"].astype(str), daily_sum["minutes"])
        ax2.set_xlabel("날짜")
        ax2.set_ylabel("분")
        ax2.set_title("일별 총합(분)")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig2)

        st.markdown("### 카테고리별 일별 추이(선)")
        cat_daily = daily.groupby(["date","category"])["minutes"].sum().reset_index()
        pivot = cat_daily.pivot(index="date", columns="category", values="minutes").fillna(0)
        fig3, ax3 = plt.subplots()
        pivot.plot(ax=ax3)
        ax3.set_xlabel("날짜")
        ax3.set_ylabel("분")
        ax3.set_title("카테고리별 일별 분")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig3)

st.caption("💡 Streamlit Cloud에서는 컨테이너가 재시작되면 파일이 초기화될 수 있어요. 주기적으로 CSV를 다운로드해두거나, 외부 DB 연동이 필요하면 말씀해 주세요.")
