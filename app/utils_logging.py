# app/utils_logging.py
import csv, os, json, uuid
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CSV_PATH = DATA_DIR / "sessions.csv"
JSON_DIR = DATA_DIR / "session_details"
LEADERBOARD_PATH = DATA_DIR / "leaderboard.csv"
TRANSITIONS_PATH = DATA_DIR / "transitions.csv"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(JSON_DIR, exist_ok=True)

def start_session(user_id="anon"):
    sid = str(uuid.uuid4())
    meta = {"session_id": sid, "user_id": user_id, "start_time": datetime.utcnow().isoformat()}
    jpath = JSON_DIR / f"{sid}.json"
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(meta, f)
    return meta

def log_interaction(session_meta, q_no, difficulty, question, chosen, correct, time_taken, reward, agent_type="heuristic", model_name=None):
    row = [
        session_meta["session_id"],
        session_meta.get("user_id", "anon"),
        datetime.utcnow().isoformat(),
        q_no,
        difficulty,
        question,
        chosen,
        int(bool(correct)),
        float(time_taken),
        float(reward),
        agent_type,
        model_name or ""
    ]
    header = ["session_id","user_id","ts","q_no","difficulty","question","chosen","correct","time_taken","reward","agent_type","model_name"]
    write_header = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow(row)
    # update json
    jpath = JSON_DIR / f"{session_meta['session_id']}.json"
    if jpath.exists():
        with open(jpath,"r",encoding="utf-8") as jf:
            existing = json.load(jf)
    else:
        existing = session_meta
    existing.setdefault("interactions", []).append({
        "ts": datetime.utcnow().isoformat(),
        "q_no": q_no,
        "difficulty": difficulty,
        "question": question,
        "chosen": chosen,
        "correct": bool(correct),
        "time_taken": float(time_taken),
        "reward": float(reward),
        "agent_type": agent_type,
        "model_name": model_name
    })
    with open(jpath, "w", encoding="utf-8") as jf:
        json.dump(existing, jf, indent=2)

def save_leaderboard_entry(user, session_id, score, accuracy):
    write_header = not LEADERBOARD_PATH.exists()
    with open(LEADERBOARD_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["user","session_id","score","accuracy","ts"])
        w.writerow([user, session_id, score, accuracy, datetime.utcnow().isoformat()])

def log_transition(s, a, r, ns, done, session_id=None):
    write_header = not TRANSITIONS_PATH.exists()
    row = list(map(str, list(s))) + [str(a), str(r)] + list(map(str, list(ns))) + [str(int(done)), session_id or ""]
    with open(TRANSITIONS_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            header = ["s0","s1","s2","s3","a","r","ns0","ns1","ns2","ns3","done","session_id"]
            w.writerow(header)
        w.writerow(row)
