"""
wsgi.py
-------
Used by PythonAnywhere to start the Flask app.
In the PythonAnywhere Web tab, the WSGI configuration file should contain
the code below (with your own username in the path).
"""
import os
import sys

# Path of the project folder on PythonAnywhere
project_home = "/home/YOUR_USERNAME/churn_app"
if project_home not in sys.path:
    sys.path.insert(0, project_home)
os.chdir(project_home)

from app import app as application  # noqa: E402
