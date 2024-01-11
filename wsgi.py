# wsgi.py
import sys

# Add your project directory to the sys.path
project_home = '/home/sandobreezez/storm-safe'
### test github
if project_home not in sys.path:
    sys.path = [project_home] + sys.path

# Import your Dash application instance from your main app file
from dash_app import app as application  # Assumes your main file is named dash_app.py and your Dash instance is named 'app'

# Assign the Flask server instance to the 'application' variable
application = application.server
