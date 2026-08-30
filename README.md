# Student Performance Prediction

An end-to-end Flask application that predicts a student's maths score from
demographic information and reading and writing scores. It is heavily inspired
by Krish Naik's end-to-end ML deployment video series.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000/predictdata`. The saved model artifacts are loaded
from the repository regardless of the directory from which the application is
started.
