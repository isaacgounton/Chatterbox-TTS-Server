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
    return config.get('generation_defaults', {}).get('temperature', 0.8)

def get_gen_default_exaggeration():
    return config.get('generation_defaults', {}).get('exaggeration', 1.3)

def get_gen_default_cfg_weight():
    return config.get('generation_defaults', {}).get('cfg_weight', 0.5)

def get_gen_default_seed():
    return config.get('generation_defaults', {}).get('seed', 0)

def get_gen_default_speed_factor():
    return config.get('generation_defaults', {}).get('speed_factor', 1.0)

def get_gen_default_language():
    return config.get('generation_defaults', {}).get('language', 'en')

def get_audio_sample_rate():
    return config.get('audio_output', {}).get('sample_rate', 24000)

def get_audio_output_format():
    return config.get('audio_output', {}).get('format', 'wav')

def get_full_config_for_template():
    return {
        'ui': config.get('ui', {}),
        'generation_defaults': config.get('generation_defaults', {}),
        'audio_output': config.get('audio_output', {}),
        'ui_state': config.get('ui_state', {})
    }

# Create a configuration manager class
class ConfigManager:
    def __init__(self):
        self.config = config
    
    def get_config(self):
        return self.config
    
    def reload(self):
        self.config = load_config()

config_manager = ConfigManager()
