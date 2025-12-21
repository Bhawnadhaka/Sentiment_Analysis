"""Configuration management utilities."""

import yaml
from pathlib import Path
from typing import Dict, Any
from loguru import logger


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        logger.warning(f"Config file not found: {config_path}")
        return {}
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded config from {config_path}")
    return config


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries.
    Later configs override earlier ones.
    
    Args:
        *configs: Variable number of config dictionaries
        
    Returns:
        Merged configuration
    """
    merged = {}
    
    for config in configs:
        merged.update(config)
    
    return merged


def load_all_configs(config_dir: str = "configs") -> Dict[str, Any]:
    """
    Load all configuration files from directory.
    
    Args:
        config_dir: Directory containing config files
        
    Returns:
        Combined configuration
    """
    config_path = Path(config_dir)
    
    if not config_path.exists():
        logger.warning(f"Config directory not found: {config_dir}")
        return {}
    
    all_configs = {}
    
    for config_file in config_path.glob("*.yaml"):
        config_name = config_file.stem
        config_data = load_config(str(config_file))
        all_configs[config_name] = config_data
    
    return all_configs


if __name__ == "__main__":
    # Test configuration loading
    configs = load_all_configs()
    
    print("Loaded configurations:")
    for name, config in configs.items():
        print(f"\n{name}:")
        print(yaml.dump(config, default_flow_style=False))
