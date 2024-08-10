from controller import create_app
import logging
from logging.handlers import RotatingFileHandler
import os

# Create logs directory if it doesn't exist
if not os.path.exists('logs'):
  os.makedirs('logs')

# Set up logging
handler = RotatingFileHandler('logs/app.log', maxBytes=10000, backupCount=3)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("[%(asctime)s] %(levelname)s in %(module)s: %(message)s")
handler.setFormatter(formatter)

app = create_app()
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)

if __name__ == "__main__":
  app.run()