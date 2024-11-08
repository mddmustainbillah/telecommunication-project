import yaml
from pathlib import Path
from typing import Dict, Any
from telecommunication.config import PROJ_ROOT


def validate_params(params: Dict[str, Any]) -> None:
    """Validate parameters from yaml file"""
    required_params = {
        'split': ['test_size', 'random_state'],
        # 'model': ['alpha', 'max_iter', 'tol'],
        # Add other required parameter groups
    }
    
    for group, params_list in required_params.items():
        if group not in params:
            raise ValueError(f"Missing parameter group: {group}")
        for param in params_list:
            if param not in params[group]:
                raise ValueError(f"Missing parameter: {param} in group {group}")

def load_params(params_path: str = "params.yaml") -> dict:
    """Load parameters from yaml file with validation"""
    with open(Path(PROJ_ROOT, params_path)) as f:
        params = yaml.safe_load(f)
    
    validate_params(params)
    return params

