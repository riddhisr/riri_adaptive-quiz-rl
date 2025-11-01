# Adaptive QuizRL+

Adaptive Quiz platform with:
- Adaptive 5-question assessment (RL-driven)
- Q-Card (flashcards) practice mode
- Dashboard with analytics & PDF reporting

## Quick start (macOS)

```bash
# 1. clone, create venv & activate
git clone git@github.com:<your-username>/<repo>.git
cd adaptive-quiz-rl
python -m venv venv
source venv/bin/activate

# 2. upgrade pip & install
pip install --upgrade pip
pip install -r requirements.txt

# 3. run app
streamlit run app/streamlit_app.py
