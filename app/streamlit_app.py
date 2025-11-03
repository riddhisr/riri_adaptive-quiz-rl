# app/streamlit_app.py
"""
Adaptive QuizRL+ — full app
- Robust agent selection via safe_agent_select(...)
- Deterministic session pools and per-session seen list to avoid repeats
- Q-Card practice mode (unchanged behavior)
- Dashboard with adaptiveness diagnostics and accuracy/time-by-position chart
- PDF generation using reportlab
- Debug toggle to inspect agent decisions and passed states
"""

import os
import sys
import time
import html
import random
import json
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import requests

# ------------------------------
# Project paths & setup
# ------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
SESSION_DETAILS_DIR = DATA_DIR / "session_details"
MODELS_DIR = PROJECT_ROOT / "models"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(SESSION_DETAILS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

INTERACTIONS_CSV = DATA_DIR / "interactions.csv"
SESSION_COUNTER = DATA_DIR / "session_counter.txt"

# Add a reports dir for generated PDFs
SESSION_REPORTS_DIR = DATA_DIR / "session_reports"
SESSION_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------
# Optional local agent imports (best-effort)
# ------------------------------
HeuristicAgent = None
DQNAgent = None
try:
    from agents.heuristic_cloned import HeuristicAgent as _HA
    HeuristicAgent = _HA
except Exception:
    try:
        from agents.heuristic import HeuristicAgent as _HA2
        HeuristicAgent = _HA2
    except Exception:
        HeuristicAgent = None

try:
    from agents.dqn_agent import DQNAgent as _DQN
    DQNAgent = _DQN
except Exception:
    DQNAgent = None

# ------------------------------
# Debug toggle in sidebar
# ------------------------------
DEBUG_AGENT = st.sidebar.checkbox("Show agent debug (actions & states)", value=False)

# ------------------------------
# Safe rerun helper
# ------------------------------
def safe_rerun():
    """
    Call streamlit's experimental_rerun if available and callable.
    Otherwise toggle a session_state sentinel to force a rerender.
    """
    rerun_attr = getattr(st, "experimental_rerun", None)
    if callable(rerun_attr):
        try:
            rerun_attr()
            return
        except Exception:
            pass
    st.session_state["_rerun_toggle"] = not st.session_state.get("_rerun_toggle", False)

# ------------------------------
# Robust agent selection helper
# ------------------------------
def safe_agent_select(agent, state, fallback=[0,1,2]):
    """
    Try several ways to call agent.select to handle variations across agent implementations.
    Returns an int action (0/1/2).
    """
    if agent is None:
        return random.choice(fallback)

    # normalize state to np.array float32
    try:
        s_arr = np.array(state, dtype=np.float32)
    except Exception:
        s_arr = state

    # Try common argument forms
    tries = [
        (s_arr,),                          # agent.select(state)
        (s_arr.reshape((1, -1)),),         # agent.select(1xN)
        (s_arr.tolist(),),                 # python list
    ]

    kw_tries = [
        {"eval_mode": True},
        {"training": False},
        {"train": False},
        {"deterministic": True},
        {"evaluation": True},
    ]

    # 1) try simple calls
    for args in tries:
        try:
            a = agent.select(*args)
            # handle numpy/scalar weirdness
            try:
                return int(np.asarray(a).item())
            except Exception:
                return int(a)
        except Exception:
            pass

    # 2) try calls with kwargs
    for args in tries:
        for kw in kw_tries:
            try:
                a = agent.select(*args, **kw)
                try:
                    return int(np.asarray(a).item())
                except Exception:
                    return int(a)
            except Exception:
                pass

    # 3) try attribute-based policy if exists
    try:
        if hasattr(agent, "policy"):
            try:
                a = agent.policy(s_arr)
                try:
                    return int(np.asarray(a).item())
                except Exception:
                    return int(a)
            except Exception:
                pass
    except Exception:
        pass

    # 4) try alternative method names
    for name in ("act","act_greedy","act_eval","action","forward"):
        if hasattr(agent, name):
            try:
                fn = getattr(agent, name)
                a = fn(s_arr)
                try:
                    return int(np.asarray(a).item())
                except Exception:
                    return int(a)
            except Exception:
                pass

    # 5) last resort: random
    return random.choice(fallback)

# ------------------------------
# Session & logging wrappers
# ------------------------------
def _ensure_session_counter():
    if not SESSION_COUNTER.exists():
        SESSION_COUNTER.write_text("0")

def _read_session_counter():
    _ensure_session_counter()
    try:
        return int(SESSION_COUNTER.read_text().strip() or "0")
    except Exception:
        return 0

def _increment_session_counter():
    n = _read_session_counter() + 1
    SESSION_COUNTER.write_text(str(n))
    return n

def start_session_local(user_id="anon", topic_name=None, session_type: Optional[str]=None):
    sid_num = _increment_session_counter()
    sid = str(sid_num)
    meta = {"session_id": sid, "user_id": user_id, "ts": time.time(), "topic_name": topic_name, "session_type": session_type}
    try:
        (SESSION_DETAILS_DIR / f"{sid}.json").write_text(json.dumps(meta), encoding="utf-8")
    except Exception:
        pass
    return meta

def log_interaction_local(session_meta, q_no, difficulty, question, chosen, correct, time_taken, reward, agent_type="Adaptive", model_name=""):
    row = {
        "ts": pd.Timestamp.now(),
        "session_id": session_meta.get("session_id", "0"),
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
    df_row = pd.DataFrame([row])
    if not Path(INTERACTIONS_CSV).exists():
        df_row.to_csv(INTERACTIONS_CSV, index=False)
    else:
        df_row.to_csv(INTERACTIONS_CSV, index=False, mode="a", header=False)

def log_transition_local(s, a, r, ns, done, session_id):
    return

# Attempt to import user's utils_logging; else fallback
try:
    from utils_logging import start_session as imported_start_session, log_interaction as imported_log_interaction, log_transition as imported_log_transition, CSV_PATH as imported_csv_path
    import inspect
    sig = inspect.signature(imported_start_session)
    if "topic_name" in sig.parameters:
        start_session = imported_start_session
    else:
        def start_session(user_id="anon", topic_name=None, session_type=None):
            return imported_start_session(user_id=user_id)
    log_interaction = imported_log_interaction
    log_transition = imported_log_transition
    CSV_PATH = str(imported_csv_path) if 'imported_csv_path' in globals() else INTERACTIONS_CSV
except Exception:
    start_session = start_session_local
    log_interaction = log_interaction_local
    log_transition = log_transition_local
    CSV_PATH = INTERACTIONS_CSV

# ------------------------------
# REPORTING: PDF generator (reportlab)
# ------------------------------
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import datetime

def generate_session_report(session_id: str, session_df: pd.DataFrame, session_kind: str = "Unknown", out_dir: Path = SESSION_REPORTS_DIR):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = out_dir / f"session_{session_id}_{timestamp}.pdf"

    preferred = ["q_no", "difficulty", "question", "chosen", "correct", "time_taken", "reward"]
    display_cols = [c for c in preferred if c in session_df.columns]
    display_cols += [c for c in session_df.columns if c not in display_cols]
    display_cols = ["Session Type"] + display_cols

    header = [col.capitalize().replace("_", " ") for col in display_cols]
    data = [header]

    def _cell(val):
        s = "" if pd.isna(val) else str(val)
        return (s[:280] + " …") if len(s) > 300 else s

    for _, row in session_df.iterrows():
        row_kind = None
        for c in ("session_type", "session_kind", "type", "mode", "source", "agent_type"):
            if c in session_df.columns:
                v = row.get(c)
                if pd.notna(v):
                    row_kind = str(v)
                    break
        if row_kind is None or row_kind == "nan":
            row_kind = session_kind
        row_cells = [row_kind] + [_cell(row[c]) if c in session_df.columns else "" for c in display_cols[1:]]
        data.append(row_cells)

    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4, rightMargin=18, leftMargin=18, topMargin=18, bottomMargin=18)
    styles = getSampleStyleSheet()
    flow = []
    flow.append(Paragraph(f"Session Report — {session_id}", styles["Title"]))
    flow.append(Spacer(1, 6))
    flow.append(Paragraph(f"Detected session type: <b>{session_kind}</b>", styles["Normal"]))
    flow.append(Spacer(1, 6))
    flow.append(Paragraph(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
    flow.append(Spacer(1, 12))
    try:
        total_q = len(session_df)
        if "correct" in session_df.columns:
            accuracy = session_df["correct"].mean()
            flow.append(Paragraph(f"Total questions: {total_q} — Accuracy: {accuracy:.3f}", styles["Normal"]))
        else:
            flow.append(Paragraph(f"Total questions: {total_q}", styles["Normal"]))
    except Exception:
        pass
    flow.append(Spacer(1, 12))
    table = Table(data, repeatRows=1)
    style = TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#e8e8e8")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.black),
        ("ALIGN", (0,0), (-1,-1), "LEFT"),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("INNERGRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("BOX", (0,0), (-1,-1), 0.5, colors.black),
    ])
    table.setStyle(style)
    flow.append(table)
    doc.build(flow)
    return str(pdf_path)

# ------------------------------
# Question provider
# ------------------------------
OPENTDB_BASE = "https://opentdb.com/api.php"
LOCAL_QCSV = DATA_DIR / "questions.csv"
FALLBACK_Q = ("[Fallback] What is 2 + 2?", ["3","4","5","22"], "4")

@st.cache_data(ttl=600)
def load_local_questions() -> pd.DataFrame:
    """Load local CSV into a DataFrame (cached)."""
    if LOCAL_QCSV.exists():
        try:
            df = pd.read_csv(LOCAL_QCSV)
            return df
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

def _fetch_opentdb(category, difficulty):
    url = f"{OPENTDB_BASE}?amount=1&type=multiple&category={category}&difficulty={difficulty}"
    r = requests.get(url, timeout=6)
    r.raise_for_status()
    d = r.json()
    if d.get("response_code",1) != 0 or not d.get("results"):
        raise ValueError("OpenTDB: no results")
    q = d["results"][0]
    question = html.unescape(q["question"])
    correct = html.unescape(q["correct_answer"])
    options = [html.unescape(x) for x in q["incorrect_answers"]] + [correct]
    random.shuffle(options)
    return question, options, correct

def _fetch_local_random(category, difficulty):
    df = load_local_questions()
    if df.empty:
        raise FileNotFoundError("Local CSV missing")
    if "topic" in df.columns and category is not None:
        df = df[df["topic"].astype(str).str.lower().str.contains(str(category).lower())]
    if "difficulty" in df.columns:
        dff = df[df["difficulty"].astype(str).str.lower() == difficulty.lower()]
        if not dff.empty:
            df = dff
    if df.empty:
        df = load_local_questions()
        if df.empty:
            raise FileNotFoundError("No local questions at all")
    row = df.sample(1).iloc[0]
    question = str(row.get("question", FALLBACK_Q[0]))
    opts = []
    for c in ["option_a","option_b","option_c","option_d","a","b","c","d"]:
        if c in row.index and pd.notna(row[c]):
            opts.append(str(row[c]))
    if not opts and "options" in row.index:
        opts = [x.strip() for x in str(row["options"]).split("|") if x.strip()]
    if not opts:
        return FALLBACK_Q
    correct = str(row.get("correct", row.get("answer", opts[0])))
    random.shuffle(opts)
    return question, opts, correct

def get_question_safe(category, difficulty="medium", retries=3):
    last_exc = None
    for _ in range(retries):
        try:
            return _fetch_local_random(category, difficulty)
        except Exception as e:
            last_exc = e
            try:
                return _fetch_opentdb(category, difficulty)
            except Exception:
                last_exc = e
                continue
    try:
        return _fetch_local_random(category, difficulty)
    except Exception:
        pass
    return FALLBACK_Q

# ------------------------------
# Build a session question pool (deterministic order, non-repeating)
# ------------------------------
def build_question_pool(topic: int, difficulty: str, pool_size: int = 20) -> List[Tuple[str, List[str], str]]:
    """
    Build a question pool for the session. Prefers local CSV questions; falls back to OpenTDB as needed.
    Returns list of (q, options, correct) in stable order (shuffled once).
    """
    pool = []
    df_local = load_local_questions()
    if not df_local.empty:
        try:
            ldf = df_local.copy()
            if "topic" in ldf.columns:
                ldf = ldf[ldf["topic"].astype(str).str.lower().str.contains(str(topic).lower())]
            if "difficulty" in ldf.columns:
                ldf = ldf[ldf["difficulty"].astype(str).str.lower() == difficulty.lower()]
            if not ldf.empty:
                sample_n = min(pool_size, len(ldf))
                sampled = ldf.sample(sample_n, random_state=42)
                for _, row in sampled.iterrows():
                    q = str(row.get("question", FALLBACK_Q[0]))
                    opts = []
                    for c in ["option_a","option_b","option_c","option_d","a","b","c","d"]:
                        if c in row.index and pd.notna(row[c]):
                            opts.append(str(row[c]))
                    if not opts and "options" in row.index:
                        opts = [x.strip() for x in str(row["options"]).split("|") if x.strip()]
                    if not opts:
                        continue
                    correct = str(row.get("correct", row.get("answer", opts[0])))
                    pool.append((q, opts, correct))
        except Exception:
            pass

    attempts = 0
    needed = pool_size - len(pool)
    while needed > 0 and attempts < pool_size * 2:
        attempts += 1
        try:
            q, opts, correct = _fetch_opentdb(topic, difficulty)
            if q not in [p[0] for p in pool]:
                pool.append((q, opts, correct))
                needed -= 1
        except Exception:
            break

    if not pool:
        pool.append(FALLBACK_Q)

    rng = random.Random(42)
    rng.shuffle(pool)
    return pool

# ------------------------------
# Pop next unique question from pool (session-level uniqueness)
# ------------------------------
def pop_next_question_from_pool(session_key: str, topic: int, difficulty: str, max_attempts: int = 8) -> Tuple[str, list, str]:
    """
    Pop next unique question from the per-session pool, skipping any question
    text already seen in this session across all difficulties.
    """
    pool_key = f"qpool_{session_key}"
    seen_key = f"seen_{session_key}"

    # Ensure seen list exists (fallback)
    if seen_key not in st.session_state:
        st.session_state[seen_key] = []

    # Ensure pool exists
    if pool_key not in st.session_state or not st.session_state[pool_key]:
        st.session_state[pool_key] = build_question_pool(topic, difficulty, pool_size=30)

    attempts = 0
    while attempts < max_attempts:
        attempts += 1
        if not st.session_state[pool_key]:
            st.session_state[pool_key] = build_question_pool(topic, difficulty, pool_size=30)
            if not st.session_state[pool_key]:
                break

        q, opts, correct = st.session_state[pool_key].pop(0)

        if q in st.session_state[seen_key]:
            # skip duplicates
            continue

        try:
            st.session_state[seen_key].append(q)
        except Exception:
            pass
        return q, opts, correct

    # fallback: try to fetch unique from safe source
    fallback_attempts = 0
    while fallback_attempts < 6:
        fallback_attempts += 1
        q, opts, correct = get_question_safe(topic, difficulty)
        if q not in st.session_state.get(seen_key, []):
            try:
                st.session_state[seen_key].append(q)
            except Exception:
                pass
            return q, opts, correct

    # absolute fallback: may repeat if truly nothing unique left
    if st.session_state.get(pool_key):
        return st.session_state[pool_key].pop(0)
    return FALLBACK_Q

# ------------------------------
# Helpers & constants
# ------------------------------
ACTION_TO_DIFF = {0: "easy", 1: "medium", 2: "hard"}
DIFF_TO_INDEX = {"easy": 0, "medium": 1, "hard": 2}

def reset_assessment_state():
    keys = [
        "session_meta","topic","topic_name","agent","agent_type","model_path","q_no","score","start_time",
        "state","last_action","current_question","max_q","agent_loaded","assessment_done","assessment_results",
    ]
    # Also clear any pool/seen keys for safety
    for k in list(st.session_state.keys()):
        if k.startswith("qpool_") or k.startswith("seen_"):
            try:
                del st.session_state[k]
            except Exception:
                pass
    for k in keys:
        if k in st.session_state:
            del st.session_state[k]

# ------------------------------
# Streamlit UI config
# ------------------------------
st.set_page_config(page_title="Adaptive QuizRL+", page_icon="🧠", layout="wide")
st.title("Adaptive QuizRL+")
st.caption("Assessment (5 questions) — Q-Card practice — Dashboard")

page = st.sidebar.selectbox("Navigate", ["Quiz (5 Qs)", "Q-Card (Practice)", "Dashboard & Report"])
st.sidebar.markdown("Tip: place .pth models in models/ or upload on the Quiz page.")

# ------------------------------
# QUIZ (5 questions) page
# ------------------------------
if page == "Quiz (5 Qs)":
    st.header("Assessment — 5 Questions (difficulty hidden)")
    with st.form("quiz_form"):
        cols = st.columns([2,2,2])
        topics_map = {"General Knowledge":9,"Science & Nature":17,"Computers":18,"Mathematics":19,"Sports":21,"Geography":22,"History":23,"Art":25}
        topic_name = cols[0].selectbox("Topic", list(topics_map.keys()), index=3)
        username = cols[1].text_input("Your name (optional)", value="You")
        cols[2].markdown("Questions per session: **5** (fixed)")
        model_options = ["Random"]
        if HeuristicAgent is not None:
            model_options.append("Heuristic")
        default_model = MODELS_DIR / "dqn.pth"
        if DQNAgent is not None and default_model.exists():
            model_options.append("DQN — models/dqn.pth")
        if DQNAgent is not None and any(MODELS_DIR.glob("*.pth")):
            model_options.append("DQN — choose from models/")
        if DQNAgent is not None:
            model_options.append("DQN — upload .pth")
        model_choice = cols[1].selectbox("Agent / Model", model_options, index=0)
        start = st.form_submit_button("Start Assessment")
    # model upload/choose
    if model_choice == "DQN — upload .pth":
        uploaded = st.file_uploader("Upload .pth model (saved to models/)", type=["pth","pt"])
        if uploaded:
            save_path = MODELS_DIR / uploaded.name
            with open(save_path,"wb") as f:
                f.write(uploaded.getbuffer())
            st.success(f"Saved uploaded model to {save_path.name}")
            st.session_state["_uploaded_model"] = str(save_path)
    if model_choice == "DQN — choose from models/":
        available = [p.name for p in MODELS_DIR.glob("*.pth")]
        if available:
            chosen_model_name = st.selectbox("Choose model file", available)
            if chosen_model_name:
                st.session_state["_chosen_model"] = str(MODELS_DIR / chosen_model_name)
        else:
            st.info("No .pth files in models/ yet.")

    if start:
        reset_assessment_state()
        st.session_state.session_meta = start_session(user_id=(username or "anon"), topic_name=topic_name, session_type="Quiz")
        st.session_state.topic = topics_map[topic_name]
        st.session_state.topic_name = topic_name
        st.session_state.max_q = 5
        st.session_state.q_no = 1
        st.session_state.score = 0
        st.session_state.start_time = None
        st.session_state.assessment_done = False
        st.session_state.assessment_results = []

        # prepare pool key using session id and seen list
        sid = st.session_state.session_meta["session_id"]
        pkey = f"qpool_{sid}"
        if pkey in st.session_state:
            del st.session_state[pkey]
        st.session_state[f"seen_{sid}"] = []

        # load agent/model
        agent = None
        st.session_state.agent_type = "Random"
        st.session_state.model_path = ""
        if model_choice == "Random":
            agent = None
            st.session_state.agent_type = "Random"
        elif model_choice == "Heuristic" and HeuristicAgent is not None:
            try:
                agent = HeuristicAgent()
                st.session_state.agent_type = "Heuristic"
            except Exception as e:
                st.warning(f"Could not init HeuristicAgent: {e}")
                agent = None
        elif model_choice == "DQN — models/dqn.pth":
            if DQNAgent is None:
                st.error("DQNAgent class not found.")
            else:
                mp = MODELS_DIR / "dqn.pth"
                if mp.exists():
                    try:
                        agent = DQNAgent()
                        agent.load(str(mp))
                        st.session_state.agent_type = "DQN"
                        st.session_state.model_path = str(mp)
                    except Exception as e:
                        st.error(f"Failed to load model: {e}")
                else:
                    st.error("models/dqn.pth not found.")
        elif model_choice == "DQN — choose from models/":
            mpath = st.session_state.get("_chosen_model", None)
            if mpath and DQNAgent is not None:
                try:
                    agent = DQNAgent()
                    agent.load(mpath)
                    st.session_state.agent_type = "DQN"
                    st.session_state.model_path = mpath
                except Exception as e:
                    st.error(f"Failed to load chosen model: {e}")
            else:
                st.warning("No model chosen.")
        elif model_choice == "DQN — upload .pth":
            mpath = st.session_state.get("_uploaded_model", None)
            if mpath and DQNAgent is not None:
                try:
                    agent = DQNAgent()
                    agent.load(mpath)
                    st.session_state.agent_type = "DQN"
                    st.session_state.model_path = mpath
                except Exception as e:
                    st.error(f"Failed to load uploaded model: {e}")
            else:
                st.warning("No uploaded model yet.")
        st.session_state.agent = agent
        st.session_state.agent_loaded = bool(agent)

        # initial state
        st.session_state.state = np.array([0.0, 0.5, 0.0, float(DIFF_TO_INDEX["medium"])], dtype=np.float32)

        # use safe agent selection helper
        a = safe_agent_select(agent, st.session_state.state)
        st.session_state.last_action = int(a)

        if DEBUG_AGENT:
            try:
                st.sidebar.write("AGENT DEBUG — initial selection:", st.session_state.get("agent_type"), "action:", int(a))
                st.sidebar.write("state:", st.session_state.get("state"))
            except Exception:
                pass

        diff = ACTION_TO_DIFF.get(a, "medium")

        # fetch first question deterministically from pool
        q, opts, correct = pop_next_question_from_pool(st.session_state.session_meta["session_id"], st.session_state.topic, diff)
        st.session_state.current_question = (q, opts, correct)

    # active assessment rendering
    if "session_meta" in st.session_state and not st.session_state.get("assessment_done", False):
        st.subheader(f"{st.session_state.topic_name} — Q{st.session_state.q_no}/{st.session_state.max_q}")
        st.write(f"Agent: **{st.session_state.get('agent_type','Random')}** — Model: `{Path(st.session_state.get('model_path','')).name or 'n/a'}`")

        if "current_question" not in st.session_state or not st.session_state.current_question:
            a = st.session_state.get("last_action", random.choice([0,1,2]))
            diff = ACTION_TO_DIFF.get(a, "medium")
            q, opts, correct = pop_next_question_from_pool(st.session_state.session_meta["session_id"], st.session_state.topic, diff)
            st.session_state.current_question = (q, opts, correct)

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
                st.error(f"Incorrect — correct answer: **{correct}**")
            a = st.session_state.last_action
            base = 1.0 if is_correct else -1.0
            diff_factor = {0:0.5,1:1.0,2:1.5}.get(a,1.0)
            time_pen = min(time_taken/10.0, 1.0)
            reward = base * diff_factor - 0.5 * time_pen

            try:
                log_interaction(st.session_state.session_meta, st.session_state.q_no, ACTION_TO_DIFF.get(a,"medium"), q, ans, int(is_correct), time_taken, reward, agent_type=st.session_state.get("agent_type","Random"), model_name=Path(st.session_state.get("model_path","")).name)
            except Exception:
                log_interaction_local(st.session_state.session_meta, st.session_state.q_no, ACTION_TO_DIFF.get(a,"medium"), q, ans, int(is_correct), time_taken, reward, agent_type=st.session_state.get("agent_type","Random"), model_name=Path(st.session_state.get("model_path","")).name)

            s = st.session_state.state
            ns = np.array([1.0 if is_correct else 0.0, min(time_taken/10.0,1.0), st.session_state.q_no / st.session_state.max_q, float(a)], dtype=np.float32)
            done = (st.session_state.q_no >= st.session_state.max_q)
            agent = st.session_state.get("agent", None)
            try:
                if agent and hasattr(agent,"store"):
                    agent.store(s,a,reward,ns,done)
                    log_transition(s,a,reward,ns,done, st.session_state.session_meta["session_id"])
                if agent and hasattr(agent,"learn"):
                    agent.learn()
            except Exception:
                pass

            st.session_state.assessment_results.append({
                "q_no": st.session_state.q_no,
                "question": q,
                "difficulty": ACTION_TO_DIFF.get(a,"medium"),
                "chosen": ans,
                "correct": correct,
                "is_correct": is_correct,
                "time_taken": time_taken,
                "reward": reward
            })
            st.session_state.state = ns

            if done:
                st.session_state.assessment_done = True
            else:
                st.session_state.q_no += 1
                # choose next action using safe helper
                na = safe_agent_select(agent, st.session_state.state)
                st.session_state.last_action = int(na)

                if DEBUG_AGENT:
                    try:
                        st.sidebar.write("AGENT DEBUG — after submit action:", st.session_state.get("agent_type"), "action:", int(na))
                        st.sidebar.write("state:", st.session_state.get("state"))
                    except Exception:
                        pass

                next_diff = ACTION_TO_DIFF.get(na, "medium")
                qn, optsn, correctn = pop_next_question_from_pool(st.session_state.session_meta["session_id"], st.session_state.topic, next_diff)
                st.session_state.current_question = (qn, optsn, correctn)
                st.session_state.start_time = None

    if st.session_state.get("assessment_done", False):
        st.success(f"Assessment complete — Score: {st.session_state.get('score',0)}/{st.session_state.get('max_q',5)}")
        df_res = pd.DataFrame(st.session_state.get("assessment_results", []))
        if not df_res.empty:
            st.table(df_res[["q_no","question","difficulty","chosen","correct","is_correct","time_taken","reward"]])
        if st.button("Go to Dashboard"):
            st.info("Switch to 'Dashboard & Report' from the left sidebar to view the report.")
            safe_rerun()

# ------------------------------
# Q-CARD (Practice) page
# ------------------------------
elif page == "Q-Card (Practice)":
    st.header("Q-Card — Practice Mode (wrong cards re-queued)")
    topics_map = {"General Knowledge":9,"Science & Nature":17,"Computers":18,"Mathematics":19,"Sports":21,"Geography":22,"History":23,"Art":25}

    if "qcard_state" not in st.session_state:
        st.session_state.qcard_state = {"mode":"Classic","score":0.0,"streak":0,"round":0,"lives":3,"help_used":0,"skips":2,"hints":2,"target":50,"time_limit":20}
    if "qcard_queue" not in st.session_state:
        st.session_state.qcard_queue = []

    with st.form("qcard_setup"):
        col_a, col_b, col_c = st.columns([2,2,2])
        diff_choice = col_a.selectbox("Choose difficulty", ["easy","medium","hard"], index=1)
        topic_choice = col_b.selectbox("Topic", list(topics_map.keys()), index=2)
        mode_choice = col_c.selectbox("Mode", ["Classic","Timed","Competitive"], index=["Classic","Timed","Competitive"].index(st.session_state.qcard_state.get("mode","Classic")))
        n_cards = st.number_input("Distinct cards to load", min_value=3, max_value=30, value=10, step=1)
        start_practice = st.form_submit_button("Start Practice")
    st.session_state.qcard_state["mode"] = mode_choice
    if mode_choice == "Timed":
        st.session_state.qcard_state["time_limit"] = st.slider("Time per card (seconds)", min_value=5, max_value=90, value=st.session_state.qcard_state.get("time_limit",20))
    if mode_choice == "Competitive":
        st.session_state.qcard_state["target"] = st.number_input("Target score to reach", min_value=10, max_value=1000, value=st.session_state.qcard_state.get("target",50))

    if start_practice:
        st.session_state.qcard_queue = []
        attempts = 0
        while len(st.session_state.qcard_queue) < n_cards and attempts < n_cards * 6:
            q, opts, correct = get_question_safe(topics_map[topic_choice], diff_choice)
            if not any(q == item["q"] for item in st.session_state.qcard_queue):
                st.session_state.qcard_queue.append({"q":q,"opts":opts,"correct":correct,"difficulty":diff_choice,"topic":topic_choice})
            attempts += 1
        st.session_state.qcard_state.update({"score":0.0,"streak":0,"round":0,"lives":3,"help_used":0,"skips":2,"hints":2})
        st.session_state.current_card = None
        st.success(f"Loaded {len(st.session_state.qcard_queue)} cards for practice (Topic: {topic_choice}, Difficulty: {diff_choice})")
        safe_rerun()

    if not st.session_state.qcard_queue:
        st.info("No cards loaded. Use the settings above and press 'Start Practice' to load questions.")
    else:
        if "current_card" not in st.session_state or st.session_state.current_card is None:
            st.session_state.current_card = st.session_state.qcard_queue[0]
        card = st.session_state.current_card
        state = st.session_state.qcard_state

        st.subheader(f"Card #{state['round']+1} — Difficulty: {card['difficulty'].title()} • Topic: {card['topic']}")
        st.write(card["q"])
        ans_key = f"qcard_choice_{state['round']}"
        opts = card.get("opts", [])
        ans = st.radio("Pick an option:", opts, key=ans_key)

        col1, col2, col3, col4 = st.columns([1,1,1,2])
        if col1.button("Help (use hint)"):
            if state["hints"] > 0:
                state["hints"] -= 1
                state["help_used"] += 1
                wrongs = [o for o in opts if o != card["correct"]]
                if wrongs:
                    remove = random.choice(wrongs)
                    new_opts = [o for o in opts if o != remove]
                    st.session_state.current_card["opts"] = new_opts
                    st.warning("Hint used: one wrong option removed. Correct answers with help get reduced points.")
                else:
                    st.info("No hint available for this card.")
                safe_rerun()
            else:
                st.warning("No hints left.")
        if col2.button("Next"):
            state["round"] += 1
            st.session_state.qcard_queue = st.session_state.qcard_queue[1:] + [st.session_state.qcard_queue[0]]
            st.session_state.current_card = None
            safe_rerun()
        if col3.button("Skip"):
            if state["skips"] > 0:
                state["skips"] -= 1
                state["streak"] = 0
                queue = st.session_state.qcard_queue
                card = queue.pop(0)
                pos = random.randint(0, max(0, len(queue)))
                queue.insert(pos, card)
                st.session_state.qcard_queue = queue
                st.session_state.current_card = None
                st.warning("Card skipped (minor penalty).")
                state["round"] += 1
                safe_rerun()
            else:
                st.warning("No skips left.")
        col4.markdown(f"Score: **{state['score']:.1f}**  •  Streak: **{state['streak']}**  •  Lives: **{state['lives']}**  •  Hints: **{state['hints']}**  •  Skips: **{state['skips']}**")

        if st.button("Submit Answer"):
            is_correct = (ans == card["correct"])
            help_used = (state["help_used"] > 0)
            if is_correct:
                base_points = 1.0 if card["difficulty"]=="easy" else 1.5 if card["difficulty"]=="medium" else 2.0
                points = base_points * (0.5 if help_used else 1.0)
                state["score"] += points
                state["streak"] += 1
                st.success(f"Correct! +{points:.1f} pts")
                st.session_state.qcard_queue.pop(0)
                st.session_state.current_card = None
                state["help_used"] = 0
            else:
                state["lives"] -= 1
                state["streak"] = 0
                st.error(f"Incorrect — correct answer: **{card['correct']}**. Card will be requeued.")
                queue = st.session_state.qcard_queue
                card = queue.pop(0)
                pos = random.randint(0, max(0, len(queue)))
                queue.insert(pos, card)
                st.session_state.qcard_queue = queue
                st.session_state.current_card = None
                state["help_used"] = 0

            try:
                log_interaction(start_session(user_id="practice", topic_name=card["topic"], session_type="Q-Card"), state["round"], card["difficulty"], card["q"], ans, int(is_correct), 0.0, 1.0 if is_correct else -0.5, agent_type="Practice")
            except Exception:
                log_interaction_local(start_session_local(user_id="practice", topic_name=card["topic"], session_type="Q-Card"), state["round"], card["difficulty"], card["q"], ans, int(is_correct), 0.0, 1.0 if is_correct else -0.5, agent_type="Practice")

            state["round"] += 1

            if state["lives"] <= 0:
                st.balloons()
                st.success(f"Game over — final score: {state['score']:.1f}")
                st.session_state.qcard_queue = []
                st.session_state.current_card = None
                st.session_state.qcard_state = {"mode":"Classic","score":0.0,"streak":0,"round":0,"lives":3,"help_used":0,"skips":2,"hints":2,"target":50}
                safe_rerun()
            if mode_choice == "Competitive" and state["score"] >= state.get("target",50):
                st.balloons()
                st.success(f"Target achieved! Score: {state['score']:.1f}")
                st.session_state.qcard_queue = []
                st.session_state.current_card = None
                st.session_state.qcard_state = {"mode":"Classic","score":0.0,"streak":0,"round":0,"lives":3,"help_used":0,"skips":2,"hints":2,"target":50}
                safe_rerun()

        st.markdown("---")
        st.write(f"Cards remaining: {len(st.session_state.qcard_queue)}")

# ------------------------------
# DASHBOARD & REPORT page (optimized)
# ------------------------------
elif page == "Dashboard & Report":
    st.header("Dashboard — Performance Insights")

    @st.cache_data(ttl=60)
    def load_interactions(csvpath: str):
        try:
            df = pd.read_csv(csvpath, parse_dates=["ts"])
            return df
        except Exception:
            return pd.DataFrame()

    csv_exists = Path(CSV_PATH).exists() if "CSV_PATH" in globals() else False
    interactions_exists = Path(INTERACTIONS_CSV).exists() if "INTERACTIONS_CSV" in globals() else False

    if not csv_exists and not interactions_exists:
        st.info("No session logs yet. Run a Quiz or Q-Card practice to generate data.")
    else:
        csvpath = CSV_PATH if csv_exists else INTERACTIONS_CSV
        df = load_interactions(str(csvpath))

        if df.empty:
            st.info("No data yet.")
        else:
            if "session_index" not in df.columns:
                try:
                    df["session_index"] = df["session_id"].astype(int)
                except Exception:
                    uniq = sorted(df["session_id"].unique(), key=lambda x: str(x))
                    mapping = {sid: i + 1 for i, sid in enumerate(uniq)}
                    df["session_index"] = df["session_id"].map(mapping)

            # ADAPTIVENESS DIAGNOSTICS
            diff_map = {"easy": 0, "medium": 1, "hard": 2}
            def to_index(d):
                try:
                    return diff_map.get(str(d).lower(), np.nan)
                except Exception:
                    return np.nan

            _diag_df = df.copy()
            if "difficulty" not in _diag_df.columns:
                st.info("No 'difficulty' column found — cannot compute adaptiveness diagnostics.")
            else:
                _diag_df["diff_idx"] = _diag_df["difficulty"].apply(to_index)
                sort_key = ["session_index"]
                if "q_no" in _diag_df.columns:
                    sort_key.append("q_no")
                else:
                    sort_key.append("ts")
                _diag_df = _diag_df.sort_values(sort_key)

                # next-difficulty per-session (shifted)
                _diag_df["next_diff_idx"] = _diag_df.groupby("session_index")["diff_idx"].shift(-1)
                _diag_df["prev_correct"] = _diag_df["correct"]
                _diag_df["diff_change"] = _diag_df["next_diff_idx"] - _diag_df["diff_idx"]

                # Filter rows that have a next question (drop the last row of each session)
                valid = _diag_df.dropna(subset=["next_diff_idx", "diff_idx", "prev_correct"])

                def adapt_metrics(df_sub):
                    wrongs = df_sub[df_sub["prev_correct"].astype(int) == 0]
                    rights = df_sub[df_sub["prev_correct"].astype(int) == 1]
                    def frac_makes_easier(rows):
                        if len(rows) == 0:
                            return np.nan
                        return (rows["diff_change"] < 0).sum() / len(rows)
                    def frac_makes_harder(rows):
                        if len(rows) == 0:
                            return np.nan
                        return (rows["diff_change"] > 0).sum() / len(rows)
                    return {
                        "count_rows": len(df_sub),
                        "when_wrong_make_easier": frac_makes_easier(wrongs),
                        "when_correct_make_harder": frac_makes_harder(rights),
                        "avg_diff_change": df_sub["diff_change"].mean()
                    }

                session_metrics = []
                for sid, g in valid.groupby("session_index"):
                    km = adapt_metrics(g)
                    km["session_index"] = sid
                    kind = "Unknown"
                    if "agent_type" in g.columns and pd.notna(g["agent_type"].iloc[0]):
                        kind = str(g["agent_type"].iloc[0])
                    else:
                        try:
                            jd = json.loads(Path(SESSION_DETAILS_DIR / f"{sid}.json").read_text(encoding="utf-8"))
                            kind = jd.get("session_type") or jd.get("mode") or kind
                        except Exception:
                            pass
                    km["session_kind"] = kind
                    session_metrics.append(km)
                sess_metrics_df = pd.DataFrame(session_metrics).set_index("session_index").sort_index(ascending=False)

                st.markdown("#### Adaptiveness diagnostics (per session)")
                if not sess_metrics_df.empty:
                    display = sess_metrics_df.copy()
                    display["when_wrong_make_easier"] = display["when_wrong_make_easier"].apply(lambda x: f"{(x*100):.1f}%" if pd.notna(x) else "n/a")
                    display["when_correct_make_harder"] = display["when_correct_make_harder"].apply(lambda x: f"{(x*100):.1f}%" if pd.notna(x) else "n/a")
                    display["avg_diff_change"] = display["avg_diff_change"].round(3)
                    st.table(display[["session_kind","count_rows","when_wrong_make_easier","when_correct_make_harder","avg_diff_change"]])
                else:
                    st.info("Not enough data to compute adaptiveness diagnostics.")

            # Core metrics
            sessions = int(df["session_index"].nunique())
            total_qs = int(len(df))
            overall_acc = float(df["correct"].mean()) if "correct" in df.columns else 0.0
            avg_time = float(df["time_taken"].mean()) if "time_taken" in df.columns else 0.0
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Sessions", sessions)
            col2.metric("Overall accuracy", f"{overall_acc*100:.2f}%")
            col3.metric("Avg time / question (s)", f"{avg_time:.2f}")
            col4.metric("Questions logged", total_qs)

            # Accuracy by difficulty
            st.markdown("### Accuracy by difficulty")
            if "difficulty" in df.columns and "correct" in df.columns:
                acc_by_diff = df.groupby("difficulty")["correct"].mean().reindex(["easy", "medium", "hard"]).fillna(0)
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.bar(acc_by_diff.index.str.title(), acc_by_diff.values * 100)
                ax.set_ylabel("Accuracy (%)")
                for i, v in enumerate(acc_by_diff.values * 100):
                    ax.text(i, v + 1, f"{v:.1f}%", ha="center")
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("Need 'difficulty' and 'correct' columns to plot accuracy by difficulty.")

            # Avg reward by difficulty
            st.markdown("### Avg reward by difficulty")
            if "difficulty" in df.columns and "reward" in df.columns:
                reward_by_diff = df.groupby("difficulty")["reward"].mean().reindex(["easy", "medium", "hard"]).fillna(0)
                fig2, ax2 = plt.subplots(figsize=(6, 3))
                ax2.bar(reward_by_diff.index.str.title(), reward_by_diff.values)
                ax2.axhline(0, color="black", linewidth=0.6)
                ax2.set_ylabel("Avg reward")
                for i, v in enumerate(reward_by_diff.values):
                    ax2.text(i, v + (0.05 if v >= 0 else -0.15), f"{v:.2f}", ha="center")
                st.pyplot(fig2)
                plt.close(fig2)
            else:
                st.info("Need 'difficulty' and 'reward' columns to plot rewards by difficulty.")

            # Topics mapping by reading session JSON files (if available)
            st.markdown("### Topics to focus (lowest accuracy)")
            topic_map = {}
            try:
                if "SESSION_DETAILS_DIR" in globals():
                    for f in Path(SESSION_DETAILS_DIR).glob("*.json"):
                        try:
                            jd = json.loads(f.read_text(encoding="utf-8"))
                            topic_map[str(jd.get("session_id"))] = jd.get("topic_name", "Unknown")
                        except Exception:
                            pass
                df["topic_name"] = df["session_id"].astype(str).map(topic_map).fillna("Unknown")
                if "topic_name" in df.columns and "correct" in df.columns:
                    topic_acc = df.groupby("topic_name")["correct"].mean().sort_values()
                    if not topic_acc.empty:
                        st.table((topic_acc.head(6) * 100).round(2).rename("Accuracy (%)").to_frame())
            except Exception as e:
                st.write("Could not compute topic map:", e)

            # ------------------------------
            # REPLACED: Mistakes by question position (heatmap)
            # New: Accuracy (%) and Avg time (s) by question position (1..N)
            # ------------------------------
            st.markdown("### Accuracy & Avg time by question position")
            if "q_no" in df.columns and "correct" in df.columns and "time_taken" in df.columns:
                max_q = int(df["q_no"].max())
                pos_acc = []
                pos_time = []
                pos_counts = []
                for pos in range(1, max_q+1):
                    sub = df[df["q_no"] == pos]
                    if len(sub) == 0:
                        pos_acc.append(np.nan)
                        pos_time.append(np.nan)
                        pos_counts.append(0)
                    else:
                        pos_acc.append(sub["correct"].mean() * 100)
                        pos_time.append(sub["time_taken"].mean())
                        pos_counts.append(len(sub))
                x = list(range(1, max_q+1))
                fig4, ax4 = plt.subplots(figsize=(8, 3))
                ax4.plot(x, pos_acc, marker="o", linewidth=1.5, label="Accuracy (%)")
                ax4.set_xlabel("Question position (1..N)")
                ax4.set_ylabel("Accuracy (%)")
                ax4.set_xticks(x)
                ax4.set_ylim(0, 100)
                ax4.grid(axis="y", linestyle="--", linewidth=0.4)

                ax5 = ax4.twinx()
                ax5.bar(x, pos_time, alpha=0.25, label="Avg time (s)")
                ax5.set_ylabel("Avg time (s)")

                # annotate counts lightly
                for xi, c in zip(x, pos_counts):
                    ax4.text(xi, 2, f"n={c}", ha="center", va="bottom", fontsize=8, alpha=0.7)

                lines, labels = ax4.get_legend_handles_labels()
                lines2, labels2 = ax5.get_legend_handles_labels()
                ax4.legend(lines + lines2, labels + labels2, loc="upper right", fontsize="small")
                st.pyplot(fig4)
                plt.close(fig4)
            else:
                st.info("Need 'q_no', 'correct' and 'time_taken' columns to show position-wise accuracy/time.")

            # Recommendations
            st.markdown("### Recommendations")
            recs = []
            if "difficulty" in df.columns and "correct" in df.columns:
                weakest_diff = acc_by_diff.idxmin()
                weakest_acc = acc_by_diff.min()
                if weakest_acc < 0.6:
                    recs.append(f"Practice more **{weakest_diff.title()}**-level questions (accuracy {weakest_acc*100:.1f}%).")
            if "topic_name" in df.columns and "correct" in df.columns:
                topic_acc = df.groupby("topic_name")["correct"].mean().sort_values()
                if not topic_acc.empty:
                    recs.append(f"Focus on topics: {list(topic_acc.head(3).index)}.")
            if "difficulty" in df.columns and "reward" in df.columns:
                worst_reward = reward_by_diff.idxmin()
                recs.append(f"Reduce time on difficulties that give negative avg reward (worst: **{worst_reward.title()}**).")
            for r in recs[:4]:
                st.info(r)

            st.markdown("---")
            st.subheader("Session explorer & report (all sessions)")

            sessions_df = df.groupby("session_index").agg(
                first_ts=("ts", "min"),
                accuracy=("correct", "mean"),
                total_reward=("reward", "sum"),
                questions=("q_no", "count")
            ).reset_index().sort_values("session_index", ascending=False)

            st.write("Sessions summary:")
            st.table(sessions_df.rename(columns={
                "session_index": "Session Index",
                "first_ts": "First Seen",
                "accuracy": "Accuracy",
                "total_reward": "Total Reward",
                "questions": "Questions"
            }).assign(**{"Accuracy": lambda dfx: (dfx["Accuracy"]*100).round(2).astype(str) + "%"}))

            if st.button("Generate PDF for ALL sessions (combined)"):
                try:
                    all_df = df.copy()
                    if "session_index" not in all_df.columns:
                        all_df["session_index"] = all_df["session_id"].astype(str)
                    cols = ["session_index"] + [c for c in all_df.columns if c != "session_index"]
                    all_df = all_df[cols]
                    pdf_path = generate_session_report("all_sessions", all_df, "Mixed")
                    with open(pdf_path, "rb") as f:
                        pdf_bytes = f.read()
                    st.success(f"Combined PDF generated: {Path(pdf_path).name}")
                    st.download_button("Download combined PDF", data=pdf_bytes, file_name=Path(pdf_path).name, mime="application/pdf")
                except Exception as e:
                    st.error(f"Failed to generate combined PDF: {e}")

            # Per-session view
            for sid in sessions_df["session_index"].tolist():
                with st.expander(f"Session {sid} — {int(sessions_df[sessions_df['session_index']==sid]['questions'].values[0])} questions — first seen: {sessions_df[sessions_df['session_index']==sid]['first_ts'].values[0]}"):
                    sdata = df[df["session_index"] == sid].sort_values("q_no" if "q_no" in df.columns else df.columns[0])
                    def detect_session_kind(session_id: str, sample_df: pd.DataFrame):
                        for col in ("session_type", "session_kind", "type", "mode", "source", "agent_type"):
                            if col in sample_df.columns:
                                vals = sample_df[col].dropna().astype(str).unique()
                                if len(vals) > 0:
                                    v = vals[0].lower()
                                    if "qcard" in v or "q-card" in v or "q_card" in v or "q card" in v:
                                        return "Q-Card"
                                    if "quiz" in v:
                                        return "Quiz"
                                    return vals[0]
                        json_path = Path(SESSION_DETAILS_DIR) / f"{session_id}.json"
                        if json_path.exists():
                            try:
                                jd = json.loads(json_path.read_text(encoding="utf-8"))
                                for key in ("session_type", "session_kind", "mode", "type", "is_qcard", "kind"):
                                    if key in jd:
                                        val = str(jd.get(key))
                                        v = val.lower()
                                        if "qcard" in v or "q-card" in v or "q card" in v:
                                            return "Q-Card"
                                        if "quiz" in v:
                                            return "Quiz"
                                        return val
                            except Exception:
                                pass
                        text_blob = " ".join(sample_df.astype(str).values.flatten()).lower()
                        if "q-card" in text_blob or "qcard" in text_blob:
                            return "Q-Card"
                        return "Unknown"

                    session_kind = detect_session_kind(str(sid), sdata)
                    show_cols = [c for c in ["q_no", "difficulty", "question", "chosen", "correct", "time_taken", "reward", "agent_type", "model_name", "topic_name"] if c in sdata.columns]
                    st.write(f"Detected type: **{session_kind}**")
                    st.table(sdata[show_cols].reset_index(drop=True))

                    if st.button(f"Generate PDF for session {sid}", key=f"pdf_sess_{sid}"):
                        try:
                            pdf_path = generate_session_report(str(sid), sdata, session_kind)
                            with open(pdf_path, "rb") as f:
                                pdf_bytes = f.read()
                            st.success(f"PDF generated for session {sid}: {Path(pdf_path).name}")
                            st.download_button(f"Download PDF (session {sid})", data=pdf_bytes, file_name=Path(pdf_path).name, mime="application/pdf", key=f"dl_{sid}")
                        except Exception as e:
                            st.error(f"Failed to generate PDF for session {sid}: {e}")

# EOF
