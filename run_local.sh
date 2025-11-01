#!/usr/bin/env bash
# run training (optional) and start streamlit app
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# optional: train a small DQN (you can skip and use heuristic)
# python3 experiments/train_dqn.py --episodes 2000 --save models/dqn_small.pth
streamlit run app/streamlit_app.py
