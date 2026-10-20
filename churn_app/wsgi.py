
import os
import sys

project_home = "/home/70171972/churn_app"
if project_home not in sys.path:
    sys.path.insert(0, project_home)
os.chdir(project_home)

from app import app as application  
