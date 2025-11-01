# app.py  -- Streamlit + Plotly + SQLite (Python 3.9+)
# Features:
# - Log study hours & question counts
# - Milestones with targets and status
# - Daywise (7/30/90) & Monthly charts (Plotly, dark theme)
# - Edit-once lock, delete row(s), CSV export
# - Optional email report (press a button) — fully-automatic via GitHub Actions (see below)

import os
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# ---------- Paths / DB ----------
def _base_dir():
    try:
        return Path(__file__).parent
    except NameError:
        return Path.cwd()

DB_PATH = _base_dir() / "study_web.db"

@st.cache_resource
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db():
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS study_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entry_date TEXT NOT NULL,
            subject TEXT NOT NULL,
            hours REAL NOT NULL CHECK(hours >= 0),
            notes TEXT,
            edited INTEGER NOT NULL DEFAULT 0
        );
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS question_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entry_date TEXT NOT NULL,
            topic TEXT NOT NULL,
            count INTEGER NOT NULL CHECK(count >= 0),
            notes TEXT,
            edited INTEGER NOT NULL DEFAULT 0
        );
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS milestones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            description TEXT,
            due_date TEXT NOT NULL,
            target_hours REAL,
            target_questions INTEGER,
            status TEXT NOT NULL DEFAULT 'Not started'
        );
    """)
    conn.commit()

# ---------- Helpers ----------
def to_iso(d):
    if isinstance(d, (datetime, date)):
        return d.strftime("%Y-%m-%d")
    return str(d)

def fetch_df(table: str) -> pd.DataFrame:
    df = pd.read_sql_query(f"SELECT * FROM {table}", get_conn(), parse_dates=["entry_date"] if "logs" in table else ["due_date"])
    return df

def upsert_study(entry_date, subject, hours, notes):
    con = get_conn()
    con.execute(
        "INSERT INTO study_logs(entry_date,subject,hours,notes) VALUES (?,?,?,?)",
        (to_iso(entry_date), subject.strip(), float(hours), notes.strip() or None),
    )
    con.commit()

def upsert_q(entry_date, topic, count, notes):
    con = get_conn()
    con.execute(
        "INSERT INTO question_logs(entry_date,topic,count,notes) VALUES (?,?,?,?)",
        (to_iso(entry_date), topic.strip(), int(count), notes.strip() or None),
    )
    con.commit()

def add_ms(title, due_date, thours, tqs, status, desc=""):
    con = get_conn()
    th = float(thours) if (thours is not None and str(thours).strip() != "") else None
    tq = int(float(tqs)) if (tqs is not None and str(tqs).strip() != "") else None
    con.execute(
        "INSERT INTO milestones(title,description,due_date,target_hours,target_questions,status) VALUES (?,?,?,?,?,?)",
        (title.strip(), desc.strip() if desc else "", to_iso(due_date), th, tq, status),
    )
    con.commit()

def mark_edited_once(table: str, row_id: int, mapping: dict):
    """Allow exactly one edit. If edited==1, block."""
    con = get_conn()
    cur = con.execute(f"SELECT edited FROM {table} WHERE id=?", (row_id,)).fetchone()
    if table.endswith("logs") and cur and cur[0] == 1:
        raise RuntimeError("This entry has already been edited once.")
    cols = ",".join([f"{k}=?" for k in mapping.keys()])
    vals = list(mapping.values())
    if table.endswith("logs"):
        cols += ", edited=1"
    con.execute(f"UPDATE {table} SET {cols} WHERE id=?", (*vals, row_id))
    con.commit()

def delete_rows(table: str, ids: list[int]):
    if not ids: return
    con = get_conn()
    q = ",".join(["?"] * len(ids))
    con.execute(f"DELETE FROM {table} WHERE id IN ({q})", ids)
    con.commit()

def export_csv(table: str) -> bytes:
    df = fetch_df(table)
    return df.to_csv(index=False).encode("utf-8")

# ---------- Charts ----------
def daily_agg(df: pd.DataFrame, value_col: str, last_n: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame({"entry_date": [], value_col: []})
    out = df.copy()
    out["entry_date"] = pd.to_datetime(out["entry_date"]).dt.date
    start = date.today() - timedelta(days=last_n - 1)
    out = out[out["entry_date"] >= start]
    g = out.groupby("entry_date", as_index=False)[value_col].sum()
    # ensure all days present
    idx = pd.date_range(start=start, end=date.today(), freq="D")
    g = g.set_index(pd.to_datetime(g["entry_date"])).reindex(idx, fill_value=0.0)
    g.index.name = "entry_date"
    g = g.reset_index().rename(columns={"index": "entry_date"})
    g["entry_date"] = g["entry_date"].dt.date
    return g[["entry_date", value_col]]

def monthly_agg(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame({"month": [], value_col: []})
    out = df.copy()
    out["entry_date"] = pd.to_datetime(out["entry_date"])
    out["month"] = out["entry_date"].dt.to_period("M").astype(str)
    g = out.groupby("month", as_index=False)[value_col].sum()
    return g

def kpi_card(label: str, value: str | int | float, col):
    with col:
        st.metric(label, value)

# ---------- UI ----------
st.set_page_config(page_title="Study Tracker", page_icon="📚", layout="wide", initial_sidebar_state="expanded")
st.markdown(
    """
    <style>
      .block-container {max-width: 1320px;}
      [data-testid="stMetricValue"] { font-size: 32px; }
    </style>
    """,
    unsafe_allow_html=True,
)

init_db()

with st.sidebar:
    st.title("📚 Study Tracker")
    page = st.radio("Navigate", ["Dashboard", "Log Study", "Log Questions", "Milestones"], index=0, label_visibility="collapsed")
    st.caption("Modern, fast, and simple.")

# --------- DASHBOARD ----------
if page == "Dashboard":
    st.subheader("Quick Actions")
    c1, c2, c3 = st.columns([1,1,3])
    with c1:
        st.page_link("app.py", label="Log Study", icon="📝")
    with c2:
        st.page_link("app.py", label="Log Questions", icon="❓")

    # KPIs
    st.divider()
    st.subheader("Overview")
    s_df = fetch_df("study_logs")
    q_df = fetch_df("question_logs")

    total_hours = float(s_df["hours"].sum()) if not s_df.empty else 0.0
    total_qs = int(q_df["count"].sum()) if not q_df.empty else 0
    last7_hours = daily_agg(s_df, "hours", 7)["hours"].sum() if not s_df.empty else 0.0

    k1, k2, k3, k4 = st.columns(4)
    kpi_card("Total Hours", f"{total_hours:.1f}", k1)
    kpi_card("Total Questions", total_qs, k2)
    kpi_card("Last 7 days (hours)", f"{last7_hours:.1f}", k3)

    # streak
    if not (s_df.empty and q_df.empty):
        active_days = set(pd.to_datetime(s_df["entry_date"]).dt.date[s_df["hours"] > 0]) if not s_df.empty else set()
        active_days |= set(pd.to_datetime(q_df["entry_date"]).dt.date[q_df["count"] > 0]) if not q_df.empty else set()
        s = 0; d = date.today()
        while d in active_days:
            s += 1; d = d - timedelta(days=1)
        kpi_card("Streak (days)", s, k4)
    else:
        kpi_card("Streak (days)", 0, k4)

    # Range selector
    st.divider()
    colA, colB = st.columns([1,3])
    with colA:
        view = st.radio("Chart View", ["7d", "30d", "90d", "Monthly"], horizontal=True)
    with colB:
        st.caption("Tip: Switch views to see daywise or monthwise trends.")

    # Charts
    if view == "Monthly":
        s_m = monthly_agg(s_df, "hours")
        q_m = monthly_agg(q_df, "count")
        fig1 = px.bar(s_m, x="month", y="hours", title="Monthly Study Hours", template="plotly_dark")
        fig2 = px.bar(q_m, x="month", y="count", title="Monthly Questions", template="plotly_dark")
        st.plotly_chart(fig1, use_container_width=True)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        span = 7 if view == "7d" else (30 if view == "30d" else 90)
        s_d = daily_agg(s_df, "hours", span)
        q_d = daily_agg(q_df, "count", span)

        fig1 = px.line(s_d, x="entry_date", y="hours", markers=True, title=f"Daily Study Hours — last {span} days", template="plotly_dark")
        fig2 = px.bar(q_d, x="entry_date", y="count", title=f"Daily Questions — last {span} days", template="plotly_dark")
        s_d["cum"] = s_d["hours"].cumsum()
        fig3 = px.line(s_d, x="entry_date", y="cum", markers=True, title="Cumulative Hours", template="plotly_dark")

        st.plotly_chart(fig1, use_container_width=True)
        st.plotly_chart(fig2, use_container_width=True)
        st.plotly_chart(fig3, use_container_width=True)

    # Export
    st.divider()
    st.subheader("Export")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button("Export Study CSV", data=export_csv("study_logs"), file_name="study_logs.csv", mime="text/csv")
    with c2:
        st.download_button("Export Questions CSV", data=export_csv("question_logs"), file_name="question_logs.csv", mime="text/csv")
    with c3:
        st.download_button("Export Milestones CSV", data=export_csv("milestones"), file_name="milestones.csv", mime="text/csv")

# --------- LOG STUDY ----------
elif page == "Log Study":
    st.subheader("Log Study Hours")
    with st.form("study_form", clear_on_submit=True):
        c1, c2, c3 = st.columns([1,2,1])
        f_date = c1.date_input("Date", value=date.today())
        f_subject = c2.text_input("Subject")
        f_hours = c3.number_input("Hours", min_value=0.0, step=0.5, value=1.0)
        f_notes = st.text_input("Notes (optional)")
        submitted = st.form_submit_button("Add")
        if submitted:
            if not f_subject.strip():
                st.error("Subject is required")
            else:
                upsert_study(f_date, f_subject, f_hours, f_notes)
                st.success("Logged ✔")

    # Table
    s_df = fetch_df("study_logs").sort_values(["entry_date", "id"], ascending=[False, False])
    st.dataframe(s_df, use_container_width=True, hide_index=True)

    # Edit once / Delete
    st.markdown("### Edit / Delete")
    c1, c2, c3 = st.columns([2,2,2])
    with c1:
        eid = st.number_input("Row ID to edit", min_value=0, step=1, value=0)
    with c2:
        new_hours = st.number_input("New Hours", min_value=0.0, step=0.5, value=1.0)
    with c3:
        new_subject = st.text_input("New Subject (optional)")
    new_notes = st.text_input("New Notes (optional)")
    if st.button("Apply Edit (allowed once)"):
        try:
            mapping = {"hours": float(new_hours)}
            if new_subject.strip(): mapping["subject"] = new_subject.strip()
            if new_notes.strip() or new_notes == "": mapping["notes"] = new_notes.strip()
            mark_edited_once("study_logs", int(eid), mapping)
            st.success("Edited ✔ (locked for further edits)")
        except Exception as e:
            st.error(str(e))

    del_ids = st.text_input("Delete IDs (comma-separated)")
    if st.button("Delete Selected"):
        try:
            ids = [int(x) for x in del_ids.split(",") if x.strip().isdigit()]
            delete_rows("study_logs", ids)
            st.success("Deleted ✔")
        except Exception as e:
            st.error(str(e))

# --------- LOG QUESTIONS ----------
elif page == "Log Questions":
    st.subheader("Log Questions Done")
    with st.form("q_form", clear_on_submit=True):
        c1, c2, c3 = st.columns([1,2,1])
        f_date = c1.date_input("Date", value=date.today(), key="q_date")
        f_topic = c2.text_input("Topic")
        f_count = c3.number_input("Count", min_value=0, step=1, value=10)
        f_notes = st.text_input("Notes (optional)", key="q_notes")
        submitted = st.form_submit_button("Add")
        if submitted:
            if not f_topic.strip():
                st.error("Topic is required")
            else:
                upsert_q(f_date, f_topic, f_count, f_notes)
                st.success("Logged ✔")

    q_df = fetch_df("question_logs").sort_values(["entry_date", "id"], ascending=[False, False])
    st.dataframe(q_df, use_container_width=True, hide_index=True)

    st.markdown("### Edit / Delete")
    c1, c2 = st.columns([2,2])
    with c1:
        eid = st.number_input("Row ID to edit", min_value=0, step=1, value=0, key="q_eid")
        new_count = st.number_input("New Count", min_value=0, step=1, value=10, key="q_new_count")
    with c2:
        new_topic = st.text_input("New Topic (optional)", key="q_new_topic")
        new_notes = st.text_input("New Notes (optional)", key="q_new_notes")
    if st.button("Apply Edit (allowed once)", key="q_edit_btn"):
        try:
            mapping = {"count": int(new_count)}
            if new_topic.strip(): mapping["topic"] = new_topic.strip()
            if new_notes.strip() or new_notes == "": mapping["notes"] = new_notes.strip()
            mark_edited_once("question_logs", int(eid), mapping)
            st.success("Edited ✔ (locked for further edits)")
        except Exception as e:
            st.error(str(e))

    del_ids = st.text_input("Delete IDs (comma-separated)", key="q_del_ids")
    if st.button("Delete Selected", key="q_del_btn"):
        try:
            ids = [int(x) for x in del_ids.split(",") if x.strip().isdigit()]
            delete_rows("question_logs", ids)
            st.success("Deleted ✔")
        except Exception as e:
            st.error(str(e))

# --------- MILESTONES ----------
elif page == "Milestones":
    st.subheader("Milestones")
    with st.form("ms_form", clear_on_submit=True):
        c1, c2, c3 = st.columns([2,1,1])
        title = c1.text_input("Title")
        due = c2.date_input("Due Date", value=date.today()+timedelta(days=7))
        th = c3.number_input("Target Hours (optional)", min_value=0.0, step=0.5)
        tq = c3.number_input("Target Questions (optional)", min_value=0, step=5, key="ms_tq")
        status = st.selectbox("Status", ["Not started", "In progress", "Done"])
        submitted = st.form_submit_button("Add")
        if submitted:
            if not title.strip():
                st.error("Title is required")
            else:
                add_ms(title, due, th, tq, status)
                st.success("Milestone added ✔")

    ms = fetch_df("milestones").sort_values(["due_date", "id"], ascending=[True, True])
    if not ms.empty:
        # live progress (based on totals)
        s_df = fetch_df("study_logs")
        q_df = fetch_df("question_logs")
        total_h = float(s_df["hours"].sum()) if not s_df.empty else 0.0
        total_q = int(q_df["count"].sum()) if not q_df.empty else 0
        def pct(row):
            parts = []
            if pd.notna(row["target_hours"]) and row["target_hours"] > 0:
                parts.append(total_h / float(row["target_hours"]) * 100)
            if pd.notna(row["target_questions"]) and row["target_questions"] > 0:
                parts.append(total_q / int(row["target_questions"]) * 100)
            return round(min(100, sum(parts)/len(parts)), 1) if parts else None
        ms["progress_%"] = ms.apply(pct, axis=1)
    st.dataframe(ms, use_container_width=True, hide_index=True)

    st.markdown("### Update / Delete")
    c1, c2 = st.columns([2,2])
    with c1:
        eid = st.number_input("Row ID to update", min_value=0, step=1, value=0, key="ms_eid")
        new_status = st.selectbox("New Status", ["Not started","In progress","Done"], key="ms_status_upd")
    with c2:
        new_due = st.date_input("New Due Date", value=date.today(), key="ms_due_upd")
    if st.button("Save Milestone Update"):
        try:
            con = get_conn()
            con.execute("UPDATE milestones SET status=?, due_date=? WHERE id=?", (new_status, to_iso(new_due), int(eid)))
            con.commit()
            st.success("Updated ✔")
        except Exception as e:
            st.error(str(e))

    del_ids = st.text_input("Delete IDs (comma-separated)", key="ms_del_ids")
    if st.button("Delete Selected", key="ms_del_btn"):
        try:
            ids = [int(x) for x in del_ids.split(",") if x.strip().isdigit()]
            delete_rows("milestones", ids)
            st.success("Deleted ✔")
        except Exception as e:
            st.error(str(e))

# --------- Optional: email report on-demand ----------
st.divider()
st.subheader("Email: Send today’s report (on-demand)")
sender = st.text_input("Gmail (sender)", value=os.environ.get("EMAIL_SENDER", ""))
pwd = st.text_input("Gmail App Password", type="password", value=os.environ.get("EMAIL_PASSWORD", ""))
to = st.text_input("Recipient", value=os.environ.get("EMAIL_RECIPIENT", ""))

def build_daily_report() -> str:
    s_df = fetch_df("study_logs")
    q_df = fetch_df("question_logs")
    today = pd.Timestamp(date.today())
    th = float(s_df[s_df["entry_date"]==today]["hours"].sum()) if not s_df.empty else 0.0
    tq = int(q_df[q_df["entry_date"]==today]["count"].sum()) if not q_df.empty else 0
    s7 = daily_agg(s_df, "hours", 7)["hours"].sum() if not s_df.empty else 0.0
    q7 = daily_agg(q_df, "count", 7)["count"].sum() if not q_df.empty else 0
    lines = [
        f"Daily Study Report — {date.today().isoformat()}",
        "",
        f"Today: {th:.2f} hours, {tq} questions",
        f"Last 7 days hours: {s7:.2f}",
        f"Last 7 days questions: {int(q7)}",
    ]
    return "\n".join(lines)

if st.button("Send report now"):
    if not (sender and pwd and to):
        st.error("Please fill sender, app password, and recipient.")
    else:
        import smtplib, ssl
        msg = f"From: {sender}\nTo: {to}\nSubject: Daily Study Report\n\n{build_daily_report()}"
        try:
            ctx = ssl.create_default_context()
            with smtplib.SMTP("smtp.gmail.com", 587) as s:
                s.starttls(context=ctx)
                s.login(sender, pwd)
                s.sendmail(sender, [to], msg.encode("utf-8"))
            st.success("Email sent ✔")
        except Exception as e:
            st.error(str(e))
