#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import json
import uuid
from datetime import datetime, timedelta, timezone
from calendar import monthrange

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# (옵션) Slack 웹훅 전송용
try:
    import requests
except Exception:
    requests = None

# -----------------------------
# 설정 & 경로
# -----------------------------
APP_DIR = os.path.join(".", ".habit_tracker")  # 리포 루트 기준
TRACKS_CSV = os.path.join(APP_DIR, "tracks.csv")
STATE_JSON = os.path.join(APP_DIR, "running.json")
CATEGORIES_JSON = os.path.join(APP_DIR, "categories.json")
REMINDERS_CSV = os.path.join(APP_DIR, "reminders.csv")

# ✅ 한글 기본 카테고리
DEFAULT_CATEGORIES = ["공부", "운동", "독서", "글쓰기", "외국어", "명상"]
KST = timezone(timedelta(hours=9))

os.makedirs(APP_DIR, exist_ok=True)

def ensure_files():
    # 트래킹 CSV
    if not os.path.exists(TRACKS_CSV):
        with open(TRACKS_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["start_iso", "end_iso", "minutes", "category", "note"])
    # 카테고리 JSON
    if not os.path.exists(CATEGORIES_JSON):
        with open(CATEGORIES_JSON, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_CATEGORIES, f, ensure_ascii=False, indent=2)
    # 리마인더 CSV
    if not os.path.exists(REMINDERS_CSV):
        with open(REMINDERS_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "id", "title", "category", "note",
                "due_iso", "advance_minutes", "repeat", "active",
                "last_fired_iso"
            ])

ensure_files()

# -----------------------------
# 공통 유틸
# -----------------------------
def now():
    return datetime.now(KST)

def iso(dt: datetime) -> str:
    return dt.astimezone(KST).isoformat(timespec="seconds")

def parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s).astimezone(KST)

def fmt_minutes(mins: int):
    h = mins // 60
    m = mins % 60
    return f"{h}h {m}m" if h else f"{m}m"

# -----------------------------
# 카테고리
# -----------------------------
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

# -----------------------------
# 타이머/트래킹
# -----------------------------
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
        # 다음달 1일 계산
        y, m = start.year, start.month
        if m == 12:
            end = start.replace(year=y+1, month=1)
        else:
            end = start.replace(month=m+1)
    elif kind == "전체":
        start = datetime(1970,1,1,tzinfo=KST)
        end = datetime(2999,1,1,tzinfo=KST)
    else:
        raise ValueError("지원하지 않는 기간")
    return start, end

def summarize(df: pd.DataFrame, start: datetime, end: datetime):
    if df.empty:
        return {}, 0
    s = pd.to_datetime(start.isoformat())
    e = pd.to_datetime(end.isoformat())
    df = df.copy()
    df["overlap_start"] = df["start"].clip(lower=s)
    df["overlap_end"] = df["end"].clip(upper=e)
    mins = ((df["overlap_end"] - df["overlap_start"]).dt.total_seconds() / 60).clip(lower=0)
    df["overlap_minutes"] = mins.round().astype(int)
    by_cat = df.groupby("category")["overlap_minutes"].sum().to_dict()
    total = int(df["overlap_minutes"].sum())
    return by_cat, total

# -----------------------------
# 리마인더
# -----------------------------
REPEAT_CHOICES = ["없음", "매일", "매주", "매월"]

def load_reminders_df() -> pd.DataFrame:
    df = pd.read_csv(REMINDERS_CSV, encoding="utf-8")
    if df.empty:
        return df
    # 타입 보정
    for col in ["due_iso", "last_fired_iso"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    if "active" in df.columns:
        df["active"] = df["active"].astype(bool)
    if "advance_minutes" in df.columns:
        df["advance_minutes"] = pd.to_numeric(df["advance_minutes"], errors="coerce").fillna(0).astype(int)
    return df

def save_reminders_df(df: pd.DataFrame):
    out = df.copy()
    if "due_iso" in out.columns:
        out["due_iso"] = out["due_iso"].apply(lambda x: x.isoformat() if pd.notna(x) else "")
    if "last_fired_iso" in out.columns:
        out["last_fired_iso"] = out["last_fired_iso"].apply(lambda x: x.isoformat() if pd.notna(x) else "")
    out.to_csv(REMINDERS_CSV, index=False, encoding="utf-8")

def add_reminder(title: str, category: str | None, note: str, due_dt: datetime,
                 advance_minutes: int = 0, repeat: str = "없음", active: bool = True):
    df = load_reminders_df()
    rid = str(uuid.uuid4())
    new_row = {
        "id": rid,
        "title": title,
        "category": category,
        "note": note,
        "due_iso": due_dt,
        "advance_minutes": int(advance_minutes),
        "repeat": repeat,
        "active": bool(active),
        "last_fired_iso": pd.NaT
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    save_reminders_df(df)
    return rid

def compute_next_due(due_dt: datetime, repeat: str) -> datetime | None:
    if repeat == "없음":
        return None
    if repeat == "매일":
        return due_dt + timedelta(days=1)
    if repeat == "매주":
        return due_dt + timedelta(weeks=1)
    if repeat == "매월":
        # 말일 안전 처리
        y, m = due_dt.year, due_dt.month
        if m == 12:
            ny, nm = y + 1, 1
        else:
            ny, nm = y, m + 1
        last_day = monthrange(ny, nm)[1]
        day = min(due_dt.day, last_day)
        return due_dt.replace(year=ny, month=nm, day=day)
    return None

def should_fire(row, now_dt: datetime):
    if not row["active"]:
        return False
    due = row["due_iso"]
    if pd.isna(due):
        return False
    adv = int(row.get("advance_minutes", 0))
    window_start = due - timedelta(minutes=adv)
    last = row.get("last_fired_iso", pd.NaT)
    # 같은 due에 대해 이미 발송했으면 스킵
    if pd.notna(last) and (window_start <= last <= due + timedelta(minutes=5)):
        return False
    return now_dt >= window_start

def mark_fired(df: pd.DataFrame, rid: str, fired_dt: datetime):
    idx = df.index[df["id"] == rid]
    if len(idx) == 0:
        return df
    i = idx[0]
    df.at[i, "last_fired_iso"] = pd.to_datetime(fired_dt.isoformat())
    repeat = df.at[i, "repeat"]
    due = df.at[i, "due_iso"]
    if pd.notna(due):
        nxt = compute_next_due(due.to_pydatetime().astimezone(KST), repeat)
        if nxt is None:
            df.at[i, "active"] = False
        else:
            df.at[i, "due_iso"] = pd.to_datetime(nxt.isoformat())
    return df

def send_slack(title: str, body: str) -> bool:
    if requests is None:
        return False
    url = st.secrets.get("SLACK_WEBHOOK_URL", None)
    if not url:
        return False
    try:
        payload = {"text": f":alarm_clock: *{title}*\n{body}"}
        r = requests.post(url, json=payload, timeout=5)
        return 200 <= r.status_code < 300
    except Exception:
        return False

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="자기계발 트래커 + 리마인더", page_icon="⏱️", layout="wide")

st.sidebar.title("⚙️ 설정 / 데이터")
cats = load_categories()
with st.sidebar:
    st.header("카테고리")
    st.write(", ".join(sorted(cats)) if cats else "(없음)")
    with st.form("cat_form", clear_on_submit=True):
        new_cat = st.text_input("카테고리 추가", "")
        rm_cat = st.multiselect("카테고리 삭제", options=sorted(cats))
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

    st.divider()
    st.header("데이터 백업")
    if os.path.exists(TRACKS_CSV):
        with open(TRACKS_CSV, "rb") as f:
            st.download_button("CSV 내보내기(트래킹)", f, file_name="tracks.csv", mime="text/csv")
    if os.path.exists(REMINDERS_CSV):
        with open(REMINDERS_CSV, "rb") as f:
            st.download_button("CSV 내보내기(리마인더)", f, file_name="reminders.csv", mime="text/csv")

    up1 = st.file_uploader("CSV 가져오기(트래킹)", type=["csv"], key="up_track")
    if up1 is not None:
        try:
            new_df = pd.read_csv(up1)
            needed = {"start_iso","end_iso","minutes","category","note"}
            if needed.issubset(set(new_df.columns)):
                if os.path.exists(TRACKS_CSV):
                    os.replace(TRACKS_CSV, TRACKS_CSV + ".bak")
                new_df.to_csv(TRACKS_CSV, index=False, encoding="utf-8")
                st.success("트래킹 CSV 가져오기 완료")
            else:
                st.error("트래킹 CSV 컬럼명이 맞지 않습니다.")
        except Exception as e:
            st.error(f"가져오기 실패: {e}")

    up2 = st.file_uploader("CSV 가져오기(리마인더)", type=["csv"], key="up_rem")
    if up2 is not None:
        try:
            new_df = pd.read_csv(up2)
            needed = {"id","title","category","note","due_iso","advance_minutes","repeat","active","last_fired_iso"}
            if needed.issubset(set(new_df.columns)):
                if os.path.exists(REMINDERS_CSV):
                    os.replace(REMINDERS_CSV, REMINDERS_CSV + ".bak")
                new_df.to_csv(REMINDERS_CSV, index=False, encoding="utf-8")
                st.success("리마인더 CSV 가져오기 완료")
            else:
                st.error("리마인더 CSV 컬럼명이 맞지 않습니다.")
        except Exception as e:
            st.error(f"가져오기 실패: {e}")

st.title("⏱️ 자기계발 시간 트래커 + 🔔 일정 리마인더")
st.caption("KST 기준 · CSV 영속 · 타이머/수동기록 · 요약/차트 · 리마인더(사전 알림, 반복, Slack 연동)")

# ---- 상단: 타이머/수동입력
if "running" not in st.session_state:
    st.session_state.running = read_state()

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
                    clear_state(); st.session_state.running = None
                    st.success(f"세션 종료: [{cat}] {minutes}분 기록")
                except Exception as e:
                    st.error(f"기록 실패: {e}")
        else:
            cats = load_categories()
            start_cat = st.selectbox("카테고리", options=sorted(cats) if cats else ["공부"])
            start_note = st.text_input("메모(옵션)", "")
            if st.button("▶️ 세션 시작"):
                state = {"category": start_cat, "start_iso": iso(now()), "note": start_note}
                write_state(state); st.session_state.running = state
                st.success(f"세션 시작: [{start_cat}] {state['start_iso']}")

with col2:
    st.subheader("수동 입력(분 단위)")
    with st.container(border=True):
        cats = load_categories()
        add_cat = st.selectbox("카테고리 선택", options=sorted(cats) if cats else ["공부"], key="add_cat")
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

# ---- 탭
df = read_all_tracks_df()
tabs = st.tabs(["📊 요약", "📜 로그", "📈 차트", "🔔 리마인더"])

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
            sum_df = (
                pd.DataFrame([{"category": k, "minutes": v} for k, v in by_cat.items()])
                .sort_values("minutes", ascending=False)
                .reset_index(drop_usecols=False)
            )
            sum_df["formatted"] = sum_df["minutes"].apply(lambda m: fmt_minutes(int(m)))
            st.dataframe(sum_df, use_container_width=True, hide_index=True)

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
        daily = df.copy()
        daily["date"] = pd.to_datetime(daily["start_iso"]).dt.tz_convert("Asia/Seoul").dt.date
        daily_sum = daily.groupby("date")["minutes"].sum().reset_index()
        fig2, ax2 = plt.subplots()
        ax2.bar(daily_sum["date"].astype(str), daily_sum["minutes"])
        ax2.set_xlabel("날짜"); ax2.set_ylabel("분"); ax2.set_title("일별 총합(분)")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig2)

        st.markdown("### 카테고리별 일별 추이(선)")
        cat_daily = daily.groupby(["date","category"])["minutes"].sum().reset_index()
        pivot = cat_daily.pivot(index="date", columns="category", values="minutes").fillna(0)
        fig3, ax3 = plt.subplots()
        pivot.plot(ax=ax3)
        ax3.set_xlabel("날짜"); ax3.set_ylabel("분"); ax3.set_title("카테고리별 일별 분")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig3)

# -----------------------------
# 🔔 리마인더 탭
# -----------------------------
with tabs[3]:
    st.markdown("### 리마인더 추가")
    rc1, rc2 = st.columns(2)
    with rc1:
        r_title = st.text_input("제목", placeholder="예: 오늘 독서 30분")
        r_cat = st.selectbox("관련 카테고리(옵션)", options=["(없음)"] + sorted(load_categories()))
        r_note = st.text_input("메모(옵션)")
    with rc2:
        today = now()
        r_date = st.date_input("기한 날짜", value=today.date())
        r_time = st.time_input("기한 시각", value=today.replace(second=0, microsecond=0).time())
        r_adv = st.number_input("사전 알림(분)", min_value=0, step=5, value=10)
        r_rep = st.selectbox("반복", REPEAT_CHOICES, index=0)

    if st.button("➕ 리마인더 생성"):
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
        view["due_local"] = view["due_iso"].dt.tz_convert("Asia/Seoul")
        view["last_fired_local"] = view["last_fired_iso"].dt.tz_convert("Asia/Seoul")
        view = view[[
            "id","active","title","category","note",
            "due_local","advance_minutes","repeat","last_fired_local"
        ]].sort_values(["active","due_local"], ascending=[False, True])
        st.dataframe(view, use_container_width=True, hide_index=True)

        st.markdown("#### 선택 항목 관리")
        sel = st.multiselect("리마인더 선택(ID)", options=view["id"].tolist())
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("선택 비활성화"):
                if sel:
                    rem_df.loc[rem_df["id"].isin(sel), "active"] = False
                    save_reminders_df(rem_df); st.success("비활성화 완료")
                else:
                    st.info("선택된 항목이 없습니다.")
        with c2:
            if st.button("선택 삭제"):
                if sel:
                    rem_df = rem_df[~rem_df["id"].isin(sel)]
                    save_reminders_df(rem_df); st.success("삭제 완료")
                else:
                    st.info("선택된 항목이 없습니다.")
        with c3:
            if st.button("선택 즉시 발송(테스트)"):
                now_dt = now()
                fired = 0
                for rid in sel:
                    row = rem_df.loc[rem_df["id"] == rid].iloc[0].to_dict()
                    title = row["title"]
                    due = row["due_iso"]
                    body = f"기한: {due}\n메모: {row.get('note','')}"
                    st.toast(f"🔔 {title}\n{body}")
                    if send_slack(f"[테스트] {title}", body):
                        st.info(f"Slack 전송: {title}")
                    rem_df = mark_fired(rem_df, rid, now_dt); fired += 1
                if fired:
                    save_reminders_df(rem_df); st.success(f"{fired}건 처리")

# -----------------------------
# 리마인더 감지 & 자동 새로고침
# -----------------------------
def scan_and_fire():
    rem_df = load_reminders_df()
    if rem_df.empty:
        return
    now_dt = now()
    fired_any = False
    for _, row in rem_df.iterrows():
        rowd = row.to_dict()
        if should_fire(rowd, now_dt):
            title = rowd["title"]
            due = rowd["due_iso"]
            adv = int(rowd.get("advance_minutes", 0))
            when = "마감 임박" if now_dt < due else "마감 도래"
            body = f"{when} · 기한: {due}\n사전알림: {adv}분\n메모: {rowd.get('note','')}"
            st.toast(f"🔔 {title}\n{body}")
            if send_slack(title, body):
                st.info(f"Slack 전송: {title}")
            rem_df = mark_fired(rem_df, rowd["id"], now_dt)
            fired_any = True
    if fired_any:
        save_reminders_df(rem_df)

# 실행 시마다 스캔 + JS로 1분마다 새로고침
scan_and_fire()
st.markdown(
    "<script>setTimeout(() => window.location.reload(), 60*1000);</script>",
    unsafe_allow_html=True
)
st.caption("💡 리마인더는 *앱이 열려 있을 때* 1분 간격으로 감지/발송됩니다. Slack 웹훅(SLACK_WEBHOOK_URL)을 설정하면 채널로도 알림을 보낼 수 있어요.")
