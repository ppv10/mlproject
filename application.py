"""Elastic Beanstalk entry point.

The Flask routes live in ``app.py``; re-exporting the application prevents the
two entry-point modules from drifting apart.
"""

from app import app, application


if __name__ == "__main__":
    app.run(host="0.0.0.0")
