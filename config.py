import os
import yaml
from pathlib import Path

# Load config.yaml
def load_config():
    config_path = Path(__file__).parent / 'config.yaml'
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

config = load_config()

# Configuration getters
def get_host():
    return os.getenv('HOST', '0.0.0.0')

def get_port():
    return int(os.getenv('PORT', 9001))

def get_log_file_path():
    return Path('logs/server.log')

def get_output_path():
    return Path('outputs')

def get_reference_audio_path():
    return Path('reference_audio')

def get_predefined_voices_path():
    return Path('chatterbox-predefined-voices.json')

def get_ui_title():
    return config.get('ui', {}).get('title', 'Chatterbox TTS')

def get_gen_default_temperature():
    return config.get('generation', {}).get('default_temperature', 0.7)

# Create a configuration manager class
class ConfigManager:
    def __init__(self):
        self.config = config
    
    def get_config(self):
        return self.config
    
    def reload(self):
        self.config = load_config()

config_manager = ConfigManager()
