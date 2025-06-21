import os
import yaml
import logging
from pathlib import Path
from typing import Any, Dict, Union, Optional

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

def get_auth_username():
    return os.getenv('AUTH_USERNAME', config.get('server', {}).get('auth_username', 'user'))

def get_auth_password():
    return os.getenv('AUTH_PASSWORD', config.get('server', {}).get('auth_password', 'password'))

def get_log_file_path(ensure_absolute: bool = False):
    path = Path('logs/server.log')
    return path.absolute() if ensure_absolute else path

def get_output_path(ensure_absolute: bool = False):
    path = Path('outputs')
    return path.absolute() if ensure_absolute else path

def get_reference_audio_path(ensure_absolute: bool = False):
    path = Path('reference_audio')
    return path.absolute() if ensure_absolute else path

def get_predefined_voices_path(ensure_absolute: bool = False):
    path = Path('voices')
    return path.absolute() if ensure_absolute else path

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
    
    def _get_nested_value(self, path, default=None):
        """Get a nested value from config using dot notation"""
        current = self.config
        for part in path.split('.'):
            if isinstance(current, dict):
                current = current.get(part, default)
            else:
                return default
        return current

    def get_int(self, path, default=0) -> int:
        """Get an integer value from config"""
        value = self._get_nested_value(path, default)
        if value is None:
            return default
        try:
            if isinstance(value, bool):  # Handle bool separately as it's a subclass of int
                return 1 if value else 0
            if isinstance(value, dict):
                return default
            if isinstance(value, (int, float, str)):
                return int(value)
            return default
        except (TypeError, ValueError):
            return default
    
    def get_float(self, path, default=0.0) -> float:
        """Get a float value from config"""
        value = self._get_nested_value(path, default)
        if value is None:
            return default
        try:
            if isinstance(value, bool):  # Handle bool separately
                return 1.0 if value else 0.0
            if isinstance(value, dict):
                return default
            if isinstance(value, (int, float, str)):
                return float(value)
            return default
        except (TypeError, ValueError):
            return default
    
    def get_str(self, path, default=""):
        """Get a string value from config"""
        value = self._get_nested_value(path, default)
        return str(value) if value is not None else default

    def get_string(self, path, default=""):
        """Alias for get_str for compatibility"""
        return self.get_str(path, default)
    
    def get_bool(self, path, default=False):
        """Get a boolean value from config"""
        value = self._get_nested_value(path, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ('true', '1', 'yes', 'on')
        return bool(value)
    
    def get(self, path, default=None):
        """Get a value from config with no type conversion"""
        return self._get_nested_value(path, default)
    
    def get_path(self, path, default="", ensure_absolute=False):
        """Get a path value from config"""
        value = self._get_nested_value(path, default)
        if value is None:
            return Path(default)
        path_obj = Path(str(value))
        return path_obj.absolute() if ensure_absolute else path_obj
    
    def update_and_save(self, update_data: Dict[str, Any]) -> bool:
        """Updates configuration with new values and saves to file"""
        try:
            def deep_update(current: dict, updates: dict):
                for key, value in updates.items():
                    if key in current and isinstance(current[key], dict) and isinstance(value, dict):
                        deep_update(current[key], value)
                    else:
                        current[key] = value

            deep_update(self.config, update_data)
            config_path = Path(__file__).parent / 'config.yaml'
            
            with open(config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            
            return True
        except Exception as e:
            logging.error(f"Failed to update and save configuration: {e}")
            return False
    
    def reset_and_save(self) -> bool:
        """Resets configuration to defaults and saves to file"""
        try:
            # Define default configuration
            default_config = {
                "ui": {
                    "title": "Chatterbox TTS",
                },
                "generation_defaults": {
                    "temperature": 0.8,
                    "exaggeration": 1.3,
                    "cfg_weight": 0.5,
                    "seed": 0,
                    "speed_factor": 1.0,
                    "language": "en"
                },
                "audio_output": {
                    "sample_rate": 24000,
                    "format": "wav",
                    "max_reference_duration_sec": 30
                },
                "audio_processing": {
                    "enable_silence_trimming": False,
                    "enable_internal_silence_fix": False,
                    "enable_unvoiced_removal": False
                },
                "server": {
                    "enable_performance_monitor": False,
                    "log_file_max_size_mb": 10,
                    "log_file_backup_count": 5
                },
                "paths": {
                    "model_cache": "./model_cache"
                }
            }
            
            self.config = default_config
            config_path = Path(__file__).parent / 'config.yaml'
            
            with open(config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            
            return True
        except Exception as e:
            logging.error(f"Failed to reset and save configuration: {e}")
            return False

config_manager = ConfigManager()
