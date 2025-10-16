"""
Configuration management for face detection project.

This module handles loading and managing configuration settings from YAML files.
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class DetectionConfig:
    """Configuration for face detection."""
    method: str = "opencv"
    confidence_threshold: float = 0.5
    min_face_size: int = 30
    scale_factor: float = 1.1
    min_neighbors: int = 5


@dataclass
class UIConfig:
    """Configuration for user interface."""
    title: str = "Face Detection Demo"
    theme: str = "light"
    sidebar_width: int = 300
    max_image_size: int = 800


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: Optional[str] = None


@dataclass
class AppConfig:
    """Main application configuration."""
    detection: DetectionConfig
    ui: UIConfig
    logging: LoggingConfig
    data_dir: str = "data"
    models_dir: str = "models"
    output_dir: str = "output"


class ConfigManager:
    """Manages application configuration."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path or "config/config.yaml"
        self.logger = logging.getLogger(__name__)
        self._config: Optional[AppConfig] = None
    
    def load_config(self) -> AppConfig:
        """Load configuration from file."""
        config_file = Path(self.config_path)
        
        if not config_file.exists():
            self.logger.warning(f"Config file {self.config_path} not found, using defaults")
            return self._create_default_config()
        
        try:
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            
            return self._parse_config(config_data)
        
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            return self._create_default_config()
    
    def _parse_config(self, config_data: Dict[str, Any]) -> AppConfig:
        """Parse configuration data into AppConfig object."""
        detection_config = DetectionConfig(**config_data.get('detection', {}))
        ui_config = UIConfig(**config_data.get('ui', {}))
        logging_config = LoggingConfig(**config_data.get('logging', {}))
        
        return AppConfig(
            detection=detection_config,
            ui=ui_config,
            logging=logging_config,
            data_dir=config_data.get('data_dir', 'data'),
            models_dir=config_data.get('models_dir', 'models'),
            output_dir=config_data.get('output_dir', 'output')
        )
    
    def _create_default_config(self) -> AppConfig:
        """Create default configuration."""
        return AppConfig(
            detection=DetectionConfig(),
            ui=UIConfig(),
            logging=LoggingConfig()
        )
    
    def save_config(self, config: AppConfig) -> None:
        """Save configuration to file."""
        config_file = Path(self.config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        config_data = {
            'detection': {
                'method': config.detection.method,
                'confidence_threshold': config.detection.confidence_threshold,
                'min_face_size': config.detection.min_face_size,
                'scale_factor': config.detection.scale_factor,
                'min_neighbors': config.detection.min_neighbors
            },
            'ui': {
                'title': config.ui.title,
                'theme': config.ui.theme,
                'sidebar_width': config.ui.sidebar_width,
                'max_image_size': config.ui.max_image_size
            },
            'logging': {
                'level': config.logging.level,
                'format': config.logging.format,
                'file': config.logging.file
            },
            'data_dir': config.data_dir,
            'models_dir': config.models_dir,
            'output_dir': config.output_dir
        }
        
        try:
            with open(config_file, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False)
            self.logger.info(f"Configuration saved to {self.config_path}")
        
        except Exception as e:
            self.logger.error(f"Error saving config: {e}")
    
    @property
    def config(self) -> AppConfig:
        """Get current configuration."""
        if self._config is None:
            self._config = self.load_config()
        return self._config


# Global configuration manager instance
config_manager = ConfigManager()
