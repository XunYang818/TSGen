import importlib

VERSIONS = ['full', 'core']

_current_version = 'full'


def set_version(version: str):
    """Set global prompt version"""
    global _current_version
    if version not in VERSIONS:
        raise ValueError(f"Invalid version '{version}', must be one of {VERSIONS}")
    _current_version = version
    print(f"Prompt version set to {version}")


def get_version() -> str:
    return _current_version


def _load_module(version: str = None):
    """Load prompt module by version"""
    v = version or _current_version
    return importlib.import_module(f'.prompts_{v}', package='agents.prompts')


def get_planning_prompts(version: str = None):
    """Get planning prompts"""
    return _load_module(version).get_planning_prompts()


def get_event_prompts(version: str = None):
    """Get event prompts"""
    return _load_module(version).get_event_prompts()


def get_action_step_prompts(version: str = None):
    """Get action step prompts"""
    return _load_module(version).get_action_step_prompts()


def get_requirements(version: str = None):
    """Get all requirement constants"""
    m = _load_module(version)
    return {
        'activity': m.ACTIVITY_PLANNING_REQUIREMENTS,
        'event': m.EVENT_REQUIREMENTS,
        'action_step': m.ACTION_STEP_REQUIREMENTS
    }


# Expose constants from current version (for backward compatibility)
def __getattr__(name):
    """Dynamic attribute access for backward compatibility"""
    if name in ['ACTIVITY_PLANNING_REQUIREMENTS', 'EVENT_REQUIREMENTS', 'ACTION_STEP_REQUIREMENTS']:
        return getattr(_load_module(), name)
    raise AttributeError(f"module 'prompts' has no attribute '{name}'")
