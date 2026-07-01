from typing import List, Optional, TypedDict, Literal, Any
from pydantic import BaseModel, Field

class ValidationResult(BaseModel):
    """Validation result"""
    correction_content: str = Field(description="Correction content", default="")
    is_valid: bool = Field(description="Whether validation passed")
    
class ActivityPlan(BaseModel):
    """Activity planning data model"""
    activity_id: str = Field(description="Unique incremental ID of activity, starting from 1, such as 1, 2, 3, ...", pattern=r'^\d+$')
    activity_name: str = Field(description="Activity name, such as breakfast, work, watching TV, etc.")
    start_time: str = Field(description="Start time, format YYYY-MM-DD HH:MM:SS, such as 2025-01-01 07:00:00", pattern=r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$')
    end_time: str = Field(description="End time, format YYYY-MM-DD HH:MM:SS, such as 2025-01-01 08:00:00", pattern=r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$')
    rooms: List[str] = Field(description="List of main rooms involved (must be rooms that exist in house layout, or empty list if going out)")
    description: str = Field(description="Detailed description of activity")
    purpose: str = Field(description="Purpose of activity")
    related_objects: List[str] = Field(description="Related objects")
    activity_type: str = Field(description="Activity type")

class Event(BaseModel):
    """Event data model"""
    event_id: str = Field(description="Unique incremental ID of event, starting from 1, such as 1, 2, 3, ...", pattern=r'^\d+$')
    activity_id: str = Field(description="ID of the activity this belongs to", pattern=r'^\d+$')
    event_name: str = Field(description="Event name, such as going to refrigerator to get bread, cutting vegetables, cleaning dishes, etc.")
    start_time: str = Field(description="Start time, format YYYY-MM-DD HH:MM:SS, such as 2025-01-01 07:00:00", pattern=r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$')
    end_time: str = Field(description="End time, format YYYY-MM-DD HH:MM:SS, such as 2025-01-01 08:00:00", pattern=r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$')
    room: str = Field(description="Room where event occurs, must be a room that exists in house layout, or 'outdoor' for outdoor activities")
    description: str = Field(description="Detailed description of event, explaining what exactly is being done")
    purpose: str = Field(description="Purpose of event, explaining why this is being done")
    related_objects: List[str] = Field(description="Related objects")
    event_type: str = Field(description="Event type")

class Position(BaseModel):
    """Coordinate position"""
    x: float = Field(description="X coordinate (decimeters)")
    y: float = Field(description="Y coordinate (decimeters)")

class MovementDetail(BaseModel):
    """Movement action details"""
    path_description: str = Field(description="Description of movement path, such as moving from bed center to bed edge, from refrigerator edge to kitchen sink edge.")
    start_position: Position = Field(description="Start position coordinates")
    end_position: Position = Field(description="End position coordinates")
    

class StationaryDetail(BaseModel):
    """Stationary action details"""
    activity_description: str = Field(description="Description of activity during stationary period, such as operating rice cooker in kitchen, watching TV in living room, etc.")
    center_position: Position = Field(description="Position coordinates")
    fluctuation_radius: float = Field(description="Fluctuation radius (meters)")
    

class AbsentDetail(BaseModel):
    """Absent action details"""
    reason: str = Field(description="Reason for leaving")
    location_description: str = Field(description="Description of outdoor location")

class ActionStep(BaseModel):
    """Action step data model"""
    action_step_id: str = Field(description="Unique incremental ID of action step, starting from 1, such as 1, 2, 3, ...", pattern=r'^\d+$')
    event_id: str = Field(description="ID of the event this belongs to", pattern=r'^\d+$')
    action_step_name: str = Field(description="Action step name, such as getting up from bed, going to refrigerator to get bread, cutting vegetables, cleaning dishes, etc.")
    start_time: str = Field(description="Start time, format YYYY-MM-DD HH:MM:SS, such as 2025-01-01 07:00:00", pattern=r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$')
    end_time: str = Field(description="End time, format YYYY-MM-DD HH:MM:SS, such as 2025-01-01 08:00:00", pattern=r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$')
    action_type: Literal['movement', 'stationary', 'absent'] = Field(description="Action type (movement/stationary/absent)")
    room: str = Field(description="Room where action occurs, must be a room that exists in house layout, or 'outdoor' for outdoor activities")
    description: str = Field(description="Action description, detailed explanation of what is being done")
    movement_detail: Optional[MovementDetail] = Field(description="Movement action details, only fill when action type is movement", default=None)
    stationary_detail: Optional[StationaryDetail] = Field(description="Stationary action details, only fill when action type is stationary", default=None)
    absent_detail: Optional[AbsentDetail] = Field(description="Absent action details, only fill when action type is absent", default=None)

class TrajectoryPoint(BaseModel):
    """Trajectory point data model"""
    x: float = Field(description="X coordinate")
    y: float = Field(description="Y coordinate")
    timestamp: str = Field(description="Timestamp, format YYYY-MM-DD HH:MM:SS, such as 2025-01-01 07:00:00", pattern=r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$')

class Trajectory(BaseModel):
    """Trajectory data model"""
    trajectory_id: str = Field(description="Unique incremental ID of trajectory, starting from 1, such as 1, 2, 3, ...", pattern=r'^\d+$')
    action_step_id: str = Field(description="ID of the action step this belongs to", pattern=r'^\d+$')
    points: List[TrajectoryPoint] = Field(description="List of trajectory points")

class SimulationResult(BaseModel):
    """Simulation result data model"""
    simulation_id: str = Field(description="Simulation ID")
    activity_plans: List[ActivityPlan] = Field(description="Activity planning list")
    events: List[Event] = Field(description="Event list")
    action_steps: List[ActionStep] = Field(description="Action step list")
    trajectories: List[Trajectory] = Field(description="Trajectory list")
    created_at: str = Field(description="Creation time")
    updated_at: Optional[str] = Field(description="Update time", default=None)

# ==================== State Models ====================

class SimulationState(TypedDict, total=False):
    """Simulation workflow state"""
    # Basic information
    simulation_id: str
    output_filepath: str
    resident_profile: str
    house_layout_yaml: str
    
    # Generated data
    activity_plans: List[ActivityPlan]
    events: List[Event]
    current_event_index: int
    action_steps: List[ActionStep]
    current_action_step_index: int
    trajectories: List[Trajectory]
    
    # Batch processing state
    current_batch_start: int
    current_batch_end: int
    
    # Validation results
    activity_validation: ValidationResult
    event_validation: ValidationResult
    event_rule_validation: ValidationResult
    action_validation: ValidationResult
    action_rule_validation: ValidationResult
    trajectory_validation: ValidationResult
    
    # Control information
    status: str
    error_message: str
    retry_count: int
    activity_retry_count: int
    event_retry_count: int
    action_retry_count: int
    
    # Flag indicating if current batch is freshly generated (for pass-once statistics)
    is_fresh_batch: bool
    
    # Usage statistics
    usage_stats: Any  # UsageStats object

def create_initial_state(simulation_id: str, output_filepath: str, 
                        resident_profile: str, house_layout_yaml: str,
                        usage_stats: Any = None) -> SimulationState:
    """Create initial simulation state"""
    return SimulationState(
        simulation_id=simulation_id,
        output_filepath=output_filepath,
        resident_profile=resident_profile,
        house_layout_yaml=house_layout_yaml,
        current_event_index=0,
        status="initialized",
        retry_count=0,
        activity_retry_count=0,
        event_retry_count=0,
        action_retry_count=0,
        usage_stats=usage_stats
    )
