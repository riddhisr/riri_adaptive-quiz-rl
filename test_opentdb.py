# test_opentdb.py
import requests, html, random, time
OPENTDB_BASE = "https://opentdb.com/api.php"

def fetch_one(cat="18", diff="medium"):
    url = f"{OPENTDB_BASE}?amount=1&type=multiple&category={cat}&difficulty={diff}"
    r = requests.get(url, timeout=8)
    r.raise_for_status()
    d = r.json()
    print("response_code:", d.get("response_code"))
    if not d.get("results"):
        print("no results")
        return
    q = d["results"][0]
    print("question:", html.unescape(q.get("question","")))
    opts = [html.unescape(x) for x in q.get("incorrect_answers",[])] + [html.unescape(q.get("correct_answer",""))]
    random.shuffle(opts)
    print("options:", opts)
    print("correct:", html.unescape(q.get("correct_answer","")))

if __name__ == "__main__":
    try:
        fetch_one("18","medium")
    except Exception as e:
        print("error:", e)
