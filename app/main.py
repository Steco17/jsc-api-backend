"""ASGI entry point with validated environment-backed configuration."""

from app.application import create_app
from app.config import Settings
from app.environment import load_settings_file

load_settings_file()
settings = Settings.from_environment()
app = create_app(settings)
