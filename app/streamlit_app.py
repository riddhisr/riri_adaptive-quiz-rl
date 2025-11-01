# # app/streamlit_app.py
# """
# Adaptive QuizRL+ — Clean, user-focused Streamlit app
# - Quiz: 5-question assessment, difficulty hidden (agent decides)
# - Q-Card: user chooses difficulty and must answer until correct; difficulty revealed
# - Dashboard: actionable insights (what to practice, which difficulty/area to focus)
# - Keeps logging and reporting (reportlab/fpdf) intact in app/reporting.py and app/utils_logging.py
# """

# import os, sys, time, html, random, json
# from pathlib import Path

# import numpy as np
# import pandas as pd
# import streamlit as st
# import matplotlib.pyplot as plt

# # ensure repo root on path
# PROJECT_ROOT = Path(__file__).resolve().parents[1]
# if str(PROJECT_ROOT) not in sys.path:
#     sys.path.insert(0, str(PROJECT_ROOT))

# # Local modules
# from agents.dqn_agent import DQNAgent
# from env.quiz_env import QuizEnv
# from utils_logging import start_session, log_interaction, log_transition, CSV_PATH
# from reporting import generate_session_report

# # constants
# DATA_DIR = PROJECT_ROOT / "data"
# os.makedirs(DATA_DIR, exist_ok=True)

# # UI config
# st.set_page_config(page_title="Adaptive QuizRL+", page_icon="🧠", layout="wide")
# st.title("Adaptive QuizRL+")
# st.caption("Short assessment quiz (5 Qs) + practice flashcards + clear insights")

# # Sidebar: minimal navigation
# page = st.sidebar.selectbox("Go to", ["Quiz (5 Qs)", "Q-Card (Practice)", "Dashboard & Report"])
# st.sidebar.markdown("Tip: run `python -m experiments.train_dqn` to train the adaptive policy (optional).")

# # Safe question fetcher
# import requests
# def get_question_safe(category, difficulty="medium"):
#     try:
#         url = f"https://opentdb.com/api.php?amount=1&type=multiple&category={category}&difficulty={difficulty}"
#         r = requests.get(url, timeout=5)
#         r.raise_for_status()
#         d = r.json()
#         if not d or d.get("response_code",1) != 0 or not d.get("results"):
#             raise ValueError("No result")
#         q = d["results"][0]
#         question = html.unescape(q["question"])
#         correct = html.unescape(q["correct_answer"])
#         options = [html.unescape(x) for x in q["incorrect_answers"]] + [correct]
#         random.shuffle(options)
#         return question, options, correct
#     except Exception:
#         # fallback
#         return "[Fallback] What is 2 + 2?", ["3","4","5","22"], "4"

# # Map action index to difficulty
# ACTION_TO_DIFF = {0: "easy", 1: "medium", 2: "hard"}

# # Small helper to reset quiz keys only
# def reset_assessment_state():
#     keys = ["session_meta","topic","topic_name","agent","q_no","score","start_time","state","last_action","current_question","max_q","agent_loaded"]
#     for k in keys:
#         if k in st.session_state:
#             del st.session_state[k]

# # ---------------------------
# # QUIZ: 5-question assessment (difficulty hidden)
# # ---------------------------
# if page == "Quiz (5 Qs)":
#     st.header("Assessment — 5 Questions")
#     st.write("This short 5-question assessment adapts to your performance. Difficulty is chosen by the system to estimate your level.")
#     # Setup (very minimal)
#     with st.form("quiz_start"):
#         cols = st.columns([2,2,2])
#         topics_map = {"General Knowledge":9, "Science & Nature":17, "Computers":18, "Mathematics":19, "Sports":21, "Geography":22, "History":23, "Art":25}
#         topic = cols[0].selectbox("Topic", list(topics_map.keys()), index=3)  # default Mathematics (user asked math earlier)
#         username = cols[1].text_input("Your name (optional)", value="You")
#         # max_q fixed to 5 per request
#         cols[2].markdown("Questions per session: **5**")
#         start = st.form_submit_button("Start Assessment")
#     if start:
#         reset_assessment_state()
#         st.session_state.session_meta = start_session(user_id=username or "anon")
#         st.session_state.topic = topics_map[topic]
#         st.session_state.topic_name = topic
#         st.session_state.max_q = 5  # fixed
#         # agent (DQN) only. If no trained model available, we still use DQNAgent object (untrained)
#         agent = DQNAgent()
#         model_path = PROJECT_ROOT / "models" / "dqn.pth"
#         agent_loaded = False
#         if model_path.exists():
#             try:
#                 agent.load(str(model_path))
#                 agent_loaded = True
#                 st.success("Adaptive policy loaded.")
#             except Exception:
#                 st.warning("Could not load adaptive policy; running with default policy.")
#         else:
#             st.info("No trained adaptive policy found; using default policy (not trained).")
#         st.session_state.agent = agent
#         st.session_state.agent_loaded = agent_loaded
#         # runtime
#         st.session_state.q_no = 1
#         st.session_state.score = 0
#         st.session_state.start_time = None
#         # initial state: last_result=0,time_norm=0.5,q_no_norm=0,diff_index=2
#         st.session_state.state = np.array([0.0, 0.5, 0.0, 2.0], dtype=np.float32)
#         # agent picks initial action/difficulty
#         try:
#             a = st.session_state.agent.select(st.session_state.state)
#         except Exception:
#             a = random.choice([0,1,2])
#         st.session_state.last_action = int(a)
#         diff = ACTION_TO_DIFF.get(a, "medium")
#         st.session_state.current_question = get_question_safe(st.session_state.topic, diff)
#         st.rerun()

#     # If session active, present question
#     if "session_meta" in st.session_state:
#         st.subheader(f"{st.session_state.topic_name} — Question {st.session_state.q_no}/5")
#         # For professional UX, do not display difficulty
#         q, opts, correct = st.session_state.current_question
#         st.write(q)
#         if st.session_state.start_time is None:
#             st.session_state.start_time = time.time()
#         ans = st.radio("Select an answer:", opts, key=f"assess_ans_{st.session_state.q_no}")
#         # Submit
#         if st.button("Submit Answer"):
#             time_taken = max(0.01, time.time() - st.session_state.start_time)
#             is_correct = (ans == correct)
#             if is_correct:
#                 st.success("Correct ✅")
#                 st.session_state.score += 1
#             else:
#                 st.error(f"Incorrect — the correct answer will appear in your detailed report.")
#             # compute reward (same as env's shaping)
#             a = st.session_state.last_action
#             base = 1.0 if is_correct else -1.0
#             diff_factor = {0:0.5,1:1.0,2:1.5}[a]
#             time_pen = min(time_taken/10.0, 1.0)
#             reward = base * diff_factor - 0.5 * time_pen
#             # Log interaction (difficulty hidden in UI but stored)
#             log_interaction(st.session_state.session_meta, st.session_state.q_no, ACTION_TO_DIFF.get(a,"medium"), q, ans, int(is_correct), time_taken, reward, agent_type="Adaptive", model_name="dqn.pth" if st.session_state.agent_loaded else "")
#             # store transition for retraining if agent supports store
#             s = st.session_state.state
#             ns = np.array([1.0 if is_correct else 0.0, min(time_taken/10.0,1.0), st.session_state.q_no/st.session_state.max_q, float(a+1)], dtype=np.float32)
#             done = (st.session_state.q_no >= st.session_state.max_q)
#             try:
#                 if hasattr(st.session_state.agent, "store"):
#                     st.session_state.agent.store(s, a, reward, ns, done)
#                     log_transition(s,a,reward,ns,done, st.session_state.session_meta["session_id"])
#             except Exception:
#                 pass
#             try:
#                 if hasattr(st.session_state.agent, "learn"):
#                     st.session_state.agent.learn()
#             except Exception:
#                 pass
#             # update and move to next or finish
#             st.session_state.state = ns
#             if done:
#                 # finished assessment: show high level result + link to Dashboard
#                 st.success(f"Assessment complete — Score: {st.session_state.score}/5")
#                 st.info("Your detailed per-question feedback and personalized recommendations are available in the Dashboard.")
#                 st.rerun()
#             else:
#                 st.session_state.q_no += 1
#                 # agent picks next action
#                 try:
#                     na = st.session_state.agent.select(st.session_state.state)
#                 except Exception:
#                     na = random.choice([0,1,2])
#                 st.session_state.last_action = int(na)
#                 next_diff = ACTION_TO_DIFF.get(na, "medium")
#                 st.session_state.current_question = get_question_safe(st.session_state.topic, next_diff)
#                 st.session_state.start_time = None
#                 st.rerun()

# # ---------------------------
# # Q-CARD (Practice) — user chooses difficulty, reveal it, repeat until correct
# # ---------------------------
# elif page == "Q-Card (Practice)":
#     st.header("Q-Card — Practice Mode")
#     st.write("Choose difficulty, see the level, and practice until you get it right (question repeats until correct).")
#     # choose difficulty
#     diff_choice = st.selectbox("Choose difficulty", ["easy", "medium", "hard"], index=1)
#     # choose topic
#     topics_map = {"General Knowledge":9, "Science & Nature":17, "Computers":18, "Mathematics":19, "Sports":21, "Geography":22, "History":23, "Art":25}
#     topic_choice = st.selectbox("Topic", list(topics_map.keys()), index=3)
#     st.write(f"Difficulty selected: **{diff_choice.title()}** (this will be shown each card)")

#     # initialize practice session
#     if "qcard_state" not in st.session_state:
#         st.session_state.qcard_state = {"q_no":1, "score":0, "streak":0}
#     s = st.session_state.qcard_state

#     # fetch card (or reuse until correct)
#     if "qcard_current" not in st.session_state or st.session_state.get("qcard_current", {}).get("answered_correctly", False):
#         q, opts, correct = get_question_safe(topics_map[topic_choice], diff_choice)
#         st.session_state.qcard_current = {"q": q, "opts": opts, "correct": correct, "answered_correctly": False, "start_time": time.time()}

#     cur = st.session_state.qcard_current
#     st.subheader(f"Card #{s['q_no']}")
#     st.write(f"Difficulty: **{diff_choice.title()}**")
#     st.write(cur["q"])
#     ans = st.radio("Choose an answer:", cur["opts"], key=f"qcard_ans_{s['q_no']}")
#     if st.button("Submit Answer"):
#         taken = max(0.01, time.time() - cur["start_time"])
#         if ans == cur["correct"]:
#             st.success("Correct — well done!")
#             s["score"] += 1
#             s["streak"] += 1
#             cur["answered_correctly"] = True
#             # log practice interaction (topic stored too)
#             log_interaction(start_session(user_id="practice"), s["q_no"], diff_choice, cur["q"], ans, True, taken, 1.0, agent_type="Practice")
#             # increment question number and prepare next card
#             s["q_no"] += 1
#             st.session_state.qcard_current = {"q": None, "opts": None, "correct": None, "answered_correctly": False}
#             st.experimental_rerun = getattr(st, "experimental_rerun", None)  # harmless
#             st.rerun()
#         else:
#             st.error("Incorrect — try again. This same question will repeat until you answer correctly.")
#             s["streak"] = 0
#             # log wrong attempt (reward negative)
#             log_interaction(start_session(user_id="practice"), s["q_no"], diff_choice, cur["q"], ans, False, taken, -0.5, agent_type="Practice")
#             # reset start_time so timer restarts
#             st.session_state.qcard_current["start_time"] = time.time()

#     st.markdown("---")
#     st.write(f"Practice Score: **{s['score']}**  •  Current streak: **{s['streak']}**")

# # ---------------------------
# # DASHBOARD & REPORT (actionable insights)
# # ---------------------------
# elif page == "Dashboard & Report":
#     st.header("Dashboard — Performance Insights")
#     if not os.path.exists(CSV_PATH):
#         st.info("No session logs available yet. Complete an assessment (Quiz) to generate data.")
#     else:
#         df = pd.read_csv(CSV_PATH, parse_dates=["ts"])
#         # Preprocess: ensure difficulty normalized
#         df["difficulty"] = df["difficulty"].astype(str)
#         # Overall KPIs
#         sessions = df["session_id"].nunique()
#         total_qs = len(df)
#         overall_acc = df["correct"].mean()
#         avg_time = df["time_taken"].mean()
#         col1, col2, col3, col4 = st.columns(4)
#         col1.metric("Assessment sessions", sessions)
#         col2.metric("Overall accuracy", f"{overall_acc*100:.2f}%")
#         col3.metric("Avg time / question (s)", f"{avg_time:.2f}")
#         col4.metric("Questions logged", total_qs)

#         st.markdown("### Accuracy by difficulty")
#         acc_by_diff = df.groupby("difficulty")["correct"].mean().reindex(["easy","medium","hard"]).fillna(0)
#         fig, ax = plt.subplots(figsize=(6,3))
#         ax.bar(acc_by_diff.index.str.title(), acc_by_diff.values*100)
#         ax.set_ylabel("Accuracy (%)")
#         for i,v in enumerate(acc_by_diff.values*100):
#             ax.text(i, v+1, f"{v:.1f}%", ha="center")
#         st.pyplot(fig)

#         st.markdown("### Avg reward by difficulty (which levels helped/hurt you)")
#         reward_by_diff = df.groupby("difficulty")["reward"].mean().reindex(["easy","medium","hard"]).fillna(0)
#         fig2, ax2 = plt.subplots(figsize=(6,3))
#         ax2.bar(reward_by_diff.index.str.title(), reward_by_diff.values)
#         ax2.axhline(0, color="black", linewidth=0.6)
#         ax2.set_ylabel("Avg reward")
#         for i,v in enumerate(reward_by_diff.values):
#             ax2.text(i, v + (0.05 if v>=0 else -0.15), f"{v:.2f}", ha="center")
#         st.pyplot(fig2)

#         st.markdown("### Topics to focus on (lowest accuracy)")
#         # We saved topic names as session topic_name in JSON - fallback to 'unknown' if not present in CSV
#         # derive topic accuracy by using the 'question' field and session jsons: fallback is to group by topic stored earlier
#         # Here, approximate by grouping by session's first question topic if available; else show per-question frequency of wrong answers by keywords
#         # Simpler: use counts of incorrect by topic_name present in session JSONs
#         # Build mapping session_id -> topic_name (if available)
#         topic_map = {}
#         sd = Path(PROJECT_ROOT / "data" / "session_details")
#         if sd.exists():
#             for f in sd.glob("*.json"):
#                 try:
#                     jd = json.loads(f.read_text(encoding="utf-8"))
#                     sid = jd.get("session_id")
#                     tname = jd.get("topic_name") or jd.get("topic_name", "Unknown")
#                     if sid:
#                         topic_map[sid] = tname
#                 except Exception:
#                     pass
#         # attach topic_name to df where possible
#         df["topic_name"] = df["session_id"].map(topic_map).fillna("Unknown")
#         topic_acc = df.groupby("topic_name")["correct"].mean().sort_values()
#         # show bottom 5 topics to focus on
#         if len(topic_acc) > 0:
#             to_focus = topic_acc.head(5)
#             st.table((to_focus*100).round(2).rename("Accuracy (%)").to_frame())
#             rec_topic = to_focus.idxmin()
#         else:
#             st.info("No topic metadata available yet.")
#             rec_topic = None

#         # heatmap correctness by question position
#         st.markdown("### Where do mistakes happen? (by question position in session)")
#         max_q = int(df["q_no"].max())
#         heat = np.zeros((max_q,))
#         counts = np.zeros((max_q,))
#         for _, row in df.iterrows():
#             pos = int(row["q_no"]) - 1
#             heat[pos] += int(row["correct"])
#             counts[pos] += 1
#         frac = np.divide(heat, np.maximum(1, counts))
#         fig3, ax3 = plt.subplots(figsize=(8,1.5))
#         ax3.imshow(frac.reshape(1,-1), aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
#         ax3.set_yticks([])
#         ax3.set_xticks(np.arange(max_q))
#         ax3.set_xlabel("Question position (1..N)")
#         st.pyplot(fig3)

#         # Personalized recommendations (simple rule-based from aggregated stats)
#         st.markdown("### Personalized recommendation")
#         recs = []
#         # If medium accuracy low, recommend medium; else recommend weakest difficulty
#         weakest_diff = acc_by_diff.idxmin()
#         weakest_diff_acc = acc_by_diff.min()
#         if weakest_diff_acc < 0.6:
#             recs.append(f"Focus on **{weakest_diff.title()}**-level questions (accuracy {weakest_diff_acc*100:.1f}%).")
#         if rec_topic and isinstance(rec_topic, str) and rec_topic != "Unknown":
#             recs.append(f"Practice more questions in **{rec_topic}** — your per-topic accuracy is low there.")
#         # which difficulty gave worst reward
#         worst_reward = reward_by_diff.idxmin()
#         recs.append(f"Avoid spending too much time on difficulties that give negative average reward (worst: **{worst_reward.title()}**).")
#         # show top 3 recs
#         for r in recs[:3]:
#             st.info(r)

#         # Session explorer & PDF download
#         st.markdown("---")
#         st.subheader("Session explorer & report")
#         sess = df.groupby("session_id").agg(accuracy=("correct","mean"), total_reward=("reward","sum"), questions=("q_no","count")).reset_index().sort_values("questions", ascending=False)
#         sid = st.selectbox("Select session to inspect", options=sess["session_id"].tolist())
#         if sid:
#             sdata = df[df["session_id"]==sid].sort_values("q_no")
#             st.table(sdata[["q_no","difficulty","question","chosen","correct","time_taken","reward"]].reset_index(drop=True))
#             if st.button("Generate PDF for this session"):
#                 try:
#                     pdf_path = generate_session_report(sid)
#                     with open(pdf_path, "rb") as f:
#                         st.download_button("Download PDF report", f.read(), file_name=os.path.basename(pdf_path))
#                 except Exception as e:
#                     st.error(f"PDF generation failed: {e}")

#         st.markdown("---")
#         st.caption("Tip: repeat the short assessment often (every 1–2 weeks) to track improvement. Use Q-Card for targeted practice.")

# app/streamlit_app.py
"""
Adaptive QuizRL+ — Unified, polished Streamlit app
- Quiz: 5-question assessment (difficulty HIDDEN from user)
- Q-Card: choose difficulty, difficulty SHOWN, repeat until correct
- Dashboard: actionable insights & PDF report generation
- TriviaNerd integration is optional via environment variables; otherwise falls back to OpenTDB then local CSV.
"""

import os
import sys
import time
import html
import random
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import requests

# ------------------------------
# Project setup
# ------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
SD_DIR = DATA_DIR / "session_details"
MODELS_DIR = PROJECT_ROOT / "models"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(SD_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Local helpers (optional)
try:
    from agents.dqn_agent import DQNAgent
except Exception:
    DQNAgent = None

try:
    from utils_logging import start_session, log_interaction, log_transition, CSV_PATH
except Exception:
    # minimal fallbacks so app runs even if utils_logging missing
    CSV_PATH = str(DATA_DIR / "interactions.csv")

    def start_session(user_id="anon"):
        sid = f"local-{int(time.time()*1000)}"
        return {"session_id": sid, "user_id": user_id, "ts": time.time(), "topic_name": None}

    def _append_row(row):
        df = pd.DataFrame([row])
        if os.path.exists(CSV_PATH):
            df.to_csv(CSV_PATH, mode="a", header=False, index=False)
        else:
            df.to_csv(CSV_PATH, index=False)

    def log_interaction(session_meta, q_no, difficulty, question, chosen, correct, time_taken, reward, agent_type="Adaptive", model_name=""):
        row = {
            "ts": pd.Timestamp.now(),
            "session_id": session_meta.get("session_id", "unknown"),
            "user_id": session_meta.get("user_id", "anon"),
            "q_no": q_no,
            "difficulty": difficulty,
            "question": question,
            "chosen": chosen,
            "correct": int(bool(correct)),
            "time_taken": float(time_taken),
            "reward": float(reward),
            "agent_type": agent_type,
            "model_name": model_name
        }
        _append_row(row)

    def log_transition(s, a, r, ns, done, session_id):
        # optional: no-op fallback
        pass

try:
    from reporting import generate_session_report
except Exception:
    def generate_session_report(session_id):
        raise RuntimeError("reporting.generate_session_report not available.")


# ------------------------------
# Streamlit UI config
# ------------------------------
st.set_page_config(page_title="Adaptive QuizRL+", page_icon="🧠", layout="wide")
st.title("Adaptive QuizRL+")
st.caption("Assessment (5 questions) — Practice Q-Cards — Dashboard & Reports")

page = st.sidebar.selectbox("Navigate", ["Quiz (5 Qs)", "Q-Card (Practice)", "Dashboard & Report"])
st.sidebar.markdown("---")
st.sidebar.write("Tip: if you trained a DQN, put the model at `models/dqn.pth` or run `python -m experiments.train_dqn`.")

# ------------------------------
# QUESTION PROVIDER: TriviaNerd optional -> OpenTDB -> local CSV fallback
# ------------------------------
TRIVIANERD_ENABLED = os.getenv("TRIVIANERD_ENABLED", "false").lower() in ("1", "true", "yes")
TRIVIANERD_URL = os.getenv("TRIVIANERD_URL", "").strip()
TRIVIANERD_KEY = os.getenv("TRIVIANERD_KEY", "").strip()
OPENTDB_BASE = "https://opentdb.com/api.php"
LOCAL_QCSV = DATA_DIR / "questions.csv"  # optional local dataset

def _get_from_trivianerd(category, difficulty):
    if not TRIVIANERD_ENABLED or not TRIVIANERD_URL:
        raise RuntimeError("TriviaNerd disabled or no URL set")
    headers = {}
    if TRIVIANERD_KEY:
        headers["x-api-key"] = TRIVIANERD_KEY
        headers["Authorization"] = f"Bearer {TRIVIANERD_KEY}"
    params = {"limit": 1, "categories": category, "difficulty": difficulty}
    resp = requests.get(TRIVIANERD_URL, params=params, headers=headers, timeout=6)
    resp.raise_for_status()
    data = resp.json()
    # attempt to parse common shapes
    item = None
    if isinstance(data, list) and data:
        item = data[0]
    elif isinstance(data, dict):
        if data.get("results"):
            item = data["results"][0]
        elif data.get("questions"):
            item = data["questions"][0]
        else:
            item = data
    if not item:
        raise ValueError("No question returned from TriviaNerd")
    q_text = item.get("question") or item.get("text") or item.get("title") or ""
    # options parsing
    if "incorrect_answers" in item and "correct_answer" in item:
        options = [html.unescape(x) for x in item.get("incorrect_answers", [])] + [html.unescape(item.get("correct_answer",""))]
        random.shuffle(options)
        correct = html.unescape(item.get("correct_answer",""))
    else:
        # try 'incorrectAnswers' or enumerated answers
        incorrect = item.get("incorrectAnswers") or item.get("incorrect_answers") or item.get("wrong_answers") or []
        correct = item.get("correct") or item.get("correctAnswer") or item.get("answer")
        options = [html.unescape(str(x)) for x in incorrect] + ([html.unescape(str(correct))] if correct else [])
        if not options:
            raise ValueError("No options parsed from TriviaNerd response")
        random.shuffle(options)
    return html.unescape(q_text), options, correct

def _get_from_opentdb(category, difficulty):
    url = f"{OPENTDB_BASE}?amount=1&type=multiple&category={category}&difficulty={difficulty}"
    r = requests.get(url, timeout=5)
    r.raise_for_status()
    d = r.json()
    if d.get("response_code", 1) != 0 or not d.get("results"):
        raise ValueError("OpenTDB returned no results")
    q = d["results"][0]
    question = html.unescape(q["question"])
    correct = html.unescape(q["correct_answer"])
    options = [html.unescape(x) for x in q["incorrect_answers"]] + [correct]
    random.shuffle(options)
    return question, options, correct

def _get_from_local(category, difficulty):
    if not LOCAL_QCSV.exists():
        raise FileNotFoundError("Local questions CSV not found")
    df = pd.read_csv(LOCAL_QCSV)
    # try filter by category/difficulty if present
    if "topic" in df.columns:
        df = df[df["topic"].astype(str).str.lower().str.contains(str(category).lower())]
    if "difficulty" in df.columns:
        df = df[df["difficulty"].astype(str).str.lower() == difficulty.lower()]
    if df.empty:
        df = pd.read_csv(LOCAL_QCSV)  # fallback to any
    row = df.sample(1).iloc[0]
    question = str(row.get("question", "[local] What is 2+2?"))
    options = []
    # try reading option columns
    for c in ["option_a","option_b","option_c","option_d","a","b","c","d"]:
        if c in row.index and pd.notna(row[c]):
            options.append(str(row[c]))
    if not options and "options" in row.index:
        opts = row["options"]
        if isinstance(opts, str):
            options = [x.strip() for x in opts.split("|") if x.strip()]
    if not options:
        options = ["3","4","5","22"]
        correct = "4"
    else:
        correct = str(row.get("correct", row.get("answer", options[0])))
    random.shuffle(options)
    return question, options, correct

def get_question_safe(category, difficulty="medium"):
    """
    Tries: TriviaNerd (optional) -> OpenTDB -> local CSV -> fallback.
    category may be numeric (OpenTDB) or string used for local mapping.
    """
    # 1) TriviaNerd (optional)
    if TRIVIANERD_ENABLED and TRIVIANERD_URL:
        try:
            return _get_from_trivianerd(category, difficulty)
        except Exception:
            # fail silently and fallback
            pass

    # 2) OpenTDB
    try:
        return _get_from_opentdb(category, difficulty)
    except Exception:
        pass

    # 3) local CSV
    try:
        return _get_from_local(category, difficulty)
    except Exception:
        pass

    # 4) guaranteed fallback
    return "[Fallback] What is 2 + 2?", ["3","4","5","22"], "4"

# ------------------------------
# Utility mapping & helpers
# ------------------------------
ACTION_TO_DIFF = {0: "easy", 1: "medium", 2: "hard"}

def reset_assessment_state():
    keys = ["session_meta","topic","topic_name","agent","q_no","score","start_time","state","last_action","current_question","max_q","agent_loaded"]
    for k in keys:
        if k in st.session_state:
            del st.session_state[k]

# ------------------------------
# QUIZ (5 questions) - difficulty hidden
# ------------------------------
if page == "Quiz (5 Qs)":
    st.header("Assessment — 5 Questions")
    st.write("Short adaptive assessment. Difficulty is chosen by the system (hidden).")
    with st.form("quiz_start"):
        cols = st.columns([2,2,2])
        topics_map = {"General Knowledge":9, "Science & Nature":17, "Computers":18, "Mathematics":19, "Sports":21, "Geography":22, "History":23, "Art":25}
        topic = cols[0].selectbox("Select topic", list(topics_map.keys()), index=3)
        username = cols[1].text_input("Your name (optional)", value="You")
        cols[2].markdown("Questions per session: **5**")
        start = st.form_submit_button("Start Assessment")
    if start:
        reset_assessment_state()
        st.session_state.session_meta = start_session(user_id=username or "anon")
        st.session_state.topic = topics_map[topic]
        st.session_state.topic_name = topic
        st.session_state.max_q = 5
        # load DQN agent if available (optional)
        agent = None
        if DQNAgent is not None:
            try:
                agent = DQNAgent()
                model_path = MODELS_DIR / "dqn.pth"
                if model_path.exists():
                    agent.load(str(model_path))
                    st.success("Adaptive policy loaded.")
                    st.session_state.agent_loaded = True
                else:
                    st.info("No trained DQN found; running with default policy.")
                    st.session_state.agent_loaded = False
            except Exception:
                agent = None
        st.session_state.agent = agent
        st.session_state.q_no = 1
        st.session_state.score = 0
        st.session_state.start_time = None
        st.session_state.state = np.array([0.0, 0.5, 0.0, 2.0], dtype=np.float32)
        # initial action selection
        try:
            a = st.session_state.agent.select(st.session_state.state) if st.session_state.agent else random.choice([0,1,2])
        except Exception:
            a = random.choice([0,1,2])
        st.session_state.last_action = int(a)
        diff = ACTION_TO_DIFF.get(a, "medium")
        st.session_state.current_question = get_question_safe(st.session_state.topic, diff)
        st.rerun()

    # active session
    if "session_meta" in st.session_state:
        st.subheader(f"{st.session_state.topic_name} — Question {st.session_state.q_no}/5")
        q, opts, correct = st.session_state.current_question
        st.write(q)
        if st.session_state.start_time is None:
            st.session_state.start_time = time.time()
        ans = st.radio("Select an answer:", opts, key=f"assess_ans_{st.session_state.q_no}")
        if st.button("Submit Answer"):
            time_taken = max(0.01, time.time() - st.session_state.start_time)
            is_correct = (ans == correct)
            if is_correct:
                st.success("Correct ✅")
                st.session_state.score += 1
            else:
                st.error("Incorrect — the correct answer will be visible in your detailed report.")
            a = st.session_state.last_action
            base = 1.0 if is_correct else -1.0
            diff_factor = {0:0.5,1:1.0,2:1.5}[a]
            time_pen = min(time_taken/10.0, 1.0)
            reward = base * diff_factor - 0.5 * time_pen
            # log
            log_interaction(st.session_state.session_meta, st.session_state.q_no, ACTION_TO_DIFF.get(a,"medium"), q, ans, int(is_correct), time_taken, reward, agent_type="Adaptive", model_name="dqn.pth" if st.session_state.get("agent_loaded") else "")
            # store transition if agent supports
            s = st.session_state.state
            ns = np.array([1.0 if is_correct else 0.0, min(time_taken/10.0,1.0), st.session_state.q_no/st.session_state.max_q, float(a+1)], dtype=np.float32)
            done = (st.session_state.q_no >= st.session_state.max_q)
            try:
                if st.session_state.agent and hasattr(st.session_state.agent, "store"):
                    st.session_state.agent.store(s, a, reward, ns, done)
                    log_transition(s,a,reward,ns,done, st.session_state.session_meta["session_id"])
            except Exception:
                pass
            try:
                if st.session_state.agent and hasattr(st.session_state.agent, "learn"):
                    st.session_state.agent.learn()
            except Exception:
                pass
            st.session_state.state = ns
            if done:
                st.success(f"Assessment complete — Score: {st.session_state.score}/5")
                st.info("Open Dashboard & Report for detailed feedback.")
                st.rerun()
            else:
                st.session_state.q_no += 1
                try:
                    na = st.session_state.agent.select(st.session_state.state) if st.session_state.agent else random.choice([0,1,2])
                except Exception:
                    na = random.choice([0,1,2])
                st.session_state.last_action = int(na)
                next_diff = ACTION_TO_DIFF.get(na, "medium")
                st.session_state.current_question = get_question_safe(st.session_state.topic, next_diff)
                st.session_state.start_time = None
                st.rerun()

# ------------------------------
# Q-CARD (Practice) - user chooses difficulty, repeat until correct
# ------------------------------
elif page == "Q-Card (Practice)":
    st.header("Q-Card — Practice Mode")
    st.write("Choose difficulty — the difficulty is shown. If you answer incorrectly, the same card repeats until you answer it correctly.")
    diff_choice = st.selectbox("Choose difficulty", ["easy","medium","hard"], index=1)
    topics_map = {"General Knowledge":9, "Science & Nature":17, "Computers":18, "Mathematics":19, "Sports":21, "Geography":22, "History":23, "Art":25}
    topic_choice = st.selectbox("Topic", list(topics_map.keys()), index=3)
    st.write(f"Difficulty: **{diff_choice.title()}**")

    if "qcard_state" not in st.session_state:
        st.session_state.qcard_state = {"q_no":1, "score":0, "streak":0}
    s = st.session_state.qcard_state

    # fetch if needed
    if "qcard_current" not in st.session_state or st.session_state.get("qcard_current", {}).get("answered_correctly", False):
        q, opts, correct = get_question_safe(topics_map[topic_choice], diff_choice)
        st.session_state.qcard_current = {"q": q, "opts": opts, "correct": correct, "answered_correctly": False, "start_time": time.time()}

    cur = st.session_state.qcard_current
    st.subheader(f"Card #{s['q_no']}")
    st.write(cur["q"])
    ans = st.radio("Choose an answer:", cur["opts"], key=f"qcard_ans_{s['q_no']}")
    if st.button("Submit Answer"):
        taken = max(0.01, time.time() - cur["start_time"])
        if ans == cur["correct"]:
            st.success("Correct — well done!")
            s["score"] += 1
            s["streak"] += 1
            cur["answered_correctly"] = True
            # log practice
            log_interaction(start_session(user_id="practice"), s["q_no"], diff_choice, cur["q"], ans, True, taken, 1.0, agent_type="Practice")
            s["q_no"] += 1
            st.session_state.qcard_current = {"q": None, "opts": None, "correct": None, "answered_correctly": False}
            st.rerun()
        else:
            st.error("Incorrect — this question will repeat until you answer it correctly.")
            s["streak"] = 0
            log_interaction(start_session(user_id="practice"), s["q_no"], diff_choice, cur["q"], ans, False, taken, -0.5, agent_type="Practice")
            st.session_state.qcard_current["start_time"] = time.time()

    st.markdown("---")
    st.write(f"Practice Score: **{s['score']}**  •  Current streak: **{s['streak']}**")

# ------------------------------
# DASHBOARD & REPORT
# ------------------------------
elif page == "Dashboard & Report":
    st.header("Dashboard — Performance Insights")
    if not os.path.exists(CSV_PATH):
        st.info("No session logs available yet. Play a Quiz to collect session logs.")
    else:
        df = pd.read_csv(CSV_PATH, parse_dates=["ts"])
        if df.empty:
            st.info("No data yet.")
        else:
            df["difficulty"] = df["difficulty"].astype(str)
            # KPIs
            sessions = df["session_id"].nunique()
            total_qs = len(df)
            overall_acc = df["correct"].mean()
            avg_time = df["time_taken"].mean()
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Assessment sessions", sessions)
            col2.metric("Overall accuracy", f"{overall_acc*100:.2f}%")
            col3.metric("Avg time / question (s)", f"{avg_time:.2f}")
            col4.metric("Questions logged", total_qs)

            # Accuracy by difficulty
            st.markdown("### Accuracy by difficulty")
            acc_by_diff = df.groupby("difficulty")["correct"].mean().reindex(["easy","medium","hard"]).fillna(0)
            fig, ax = plt.subplots(figsize=(6,3))
            ax.bar(acc_by_diff.index.str.title(), acc_by_diff.values*100)
            ax.set_ylabel("Accuracy (%)")
            for i,v in enumerate(acc_by_diff.values*100):
                ax.text(i, v+1, f"{v:.1f}%", ha="center")
            st.pyplot(fig)

            # Avg reward by difficulty
            st.markdown("### Avg reward by difficulty")
            reward_by_diff = df.groupby("difficulty")["reward"].mean().reindex(["easy","medium","hard"]).fillna(0)
            fig2, ax2 = plt.subplots(figsize=(6,3))
            ax2.bar(reward_by_diff.index.str.title(), reward_by_diff.values)
            ax2.axhline(0, color="black", linewidth=0.6)
            ax2.set_ylabel("Avg reward")
            for i,v in enumerate(reward_by_diff.values):
                ax2.text(i, v + (0.05 if v>=0 else -0.15), f"{v:.2f}", ha="center")
            st.pyplot(fig2)

            # Topic weaknesses (map session JSONs -> topic_name)
            topic_map = {}
            sd = SD_DIR
            if sd.exists():
                for f in sd.glob("*.json"):
                    try:
                        jd = json.loads(f.read_text(encoding="utf-8"))
                        sid = jd.get("session_id")
                        tname = jd.get("topic_name") or jd.get("topic") or "Unknown"
                        if sid:
                            topic_map[sid] = tname
                    except Exception:
                        pass
            df["topic_name"] = df["session_id"].map(topic_map).fillna("Unknown")
            topic_acc = df.groupby("topic_name")["correct"].mean().sort_values()
            if len(topic_acc) > 0:
                st.markdown("### Topics to focus on (lowest accuracy)")
                to_focus = topic_acc.head(5)
                st.table((to_focus*100).round(2).rename("Accuracy (%)").to_frame())
                rec_topic = to_focus.idxmin()
            else:
                rec_topic = None

            # Heatmap: correctness by question position
            st.markdown("### Where do mistakes happen? (by question position)")
            max_q = int(df["q_no"].max())
            heat = np.zeros((max_q,))
            counts = np.zeros((max_q,))
            for _, row in df.iterrows():
                pos = int(row["q_no"]) - 1
                if 0 <= pos < max_q:
                    heat[pos] += int(row["correct"])
                    counts[pos] += 1
            frac = np.divide(heat, np.maximum(1, counts))
            fig3, ax3 = plt.subplots(figsize=(8,1.2))
            ax3.imshow(frac.reshape(1,-1), aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
            ax3.set_yticks([])
            ax3.set_xticks(np.arange(max_q))
            ax3.set_xlabel("Question position (1..N)")
            st.pyplot(fig3)

            # Personalized recommendations
            st.markdown("### Personalized recommendation")
            recs = []
            weakest_diff = acc_by_diff.idxmin()
            weakest_diff_acc = acc_by_diff.min()
            if weakest_diff_acc < 0.6:
                recs.append(f"Focus on **{weakest_diff.title()}**-level questions (accuracy {weakest_diff_acc*100:.1f}%).")
            if rec_topic and rec_topic != "Unknown":
                recs.append(f"Practice more questions in **{rec_topic}** — your per-topic accuracy is low there.")
            worst_reward = reward_by_diff.idxmin()
            recs.append(f"Avoid spending too much time on difficulties that give negative average reward (worst: **{worst_reward.title()}**).")
            for r in recs[:3]:
                st.info(r)

            # Session explorer & PDF
            st.markdown("---")
            st.subheader("Session explorer & report")
            sess = df.groupby("session_id").agg(accuracy=("correct","mean"), total_reward=("reward","sum"), questions=("q_no","count")).reset_index().sort_values("questions", ascending=False)
            sid = st.selectbox("Select session to inspect", options=sess["session_id"].tolist())
            if sid:
                sdata = df[df["session_id"]==sid].sort_values("q_no")
                st.table(sdata[["q_no","difficulty","question","chosen","correct","time_taken","reward"]].reset_index(drop=True))
                if st.button("Generate PDF for this session"):
                    try:
                        pdf_path = generate_session_report(sid)
                        with open(pdf_path, "rb") as f:
                            st.download_button("Download PDF report", f.read(), file_name=os.path.basename(pdf_path))
                    except Exception as e:
                        st.error(f"PDF generation failed: {e}")

            st.markdown("---")
            st.caption("Tip: repeat the short assessment often to track improvement. Use Q-Card for targeted practice.")
