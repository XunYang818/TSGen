from typing import List, Dict, Any
from datetime import datetime, timedelta
from agents.data_models import Trajectory, TrajectoryPoint, SimulationState
from utils.navigation import Navigation
import logging

logger = logging.getLogger(__name__)

class TrajectoryAgent:
    """Trajectory generation agent"""
    
    def __init__(self, speed_range: tuple = (0.6, 0.8), dt: float = 1.0, 
                 stationary_dt: float = 5.0, unit: str = 'dm'):
        """
        Initialize trajectory generation agent
        
        Args:
            speed_range: Movement speed range (meters/second)
            dt: Movement event sampling interval (seconds)
            stationary_dt: Stationary event sampling interval (seconds)
            unit: Coordinate unit, 'dm' for decimeters, 'm' for meters
        """
        self.speed_range = speed_range
        self.dt = dt
        self.stationary_dt = stationary_dt
        self.unit = unit
        
    def generate_trajectories_for_action_steps(self, state: SimulationState) -> SimulationState:
        """
        Generate trajectory sequences for all action steps
        """
        try:
            action_steps = state.get("action_steps", [])
            house_layout_yaml = state.get("house_layout_yaml")
              
            logger.info(f"Starting to generate trajectories for {len(action_steps)} action steps...")
            
            # Initialize navigation system
            nav = Navigation(house_layout_yaml)
            
            # Copy action steps list to allow time modifications
            adjusted_action_steps = [step.model_copy() for step in action_steps]
            trajectories = []
            
            # Process action steps one by one, adjusting time offset for all subsequent actions each time
            for i, action_step in enumerate(adjusted_action_steps):
                logger.info(f"Processing action step {i+1}/{len(adjusted_action_steps)}: {action_step.action_step_name}")
                
                # Get action basic information
                action_type = action_step.action_type
                start_time_str = action_step.start_time
                end_time_str = action_step.end_time
                
                try:
                    if action_type == "movement":
                        # Generate movement trajectory
                        trajectory_result = self._generate_movement_trajectory(
                            action_step, nav, start_time_str
                        )
                        
                        if trajectory_result["success"]:
                            trajectory_obj = trajectory_result["trajectory"]
                            trajectories.append(trajectory_obj)

                            planned_duration = (datetime.strptime(end_time_str, "%Y-%m-%d %H:%M:%S") - datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")).total_seconds()
                            actual_duration = trajectory_result["actual_duration"]

                            time_offset = actual_duration - planned_duration

                            logger.debug(f"Movement trajectory generated successfully, planned duration: {planned_duration:.1f}s, actual duration: {actual_duration:.1f}s, offset: {time_offset:.1f}s")

                            # If there's time offset, adjust current action end time and all subsequent action times
                            if abs(time_offset) > 0.1:
                                self._adjust_current_and_subsequent_steps(adjusted_action_steps, i, time_offset)
                            
                        else:
                            error_msg = f"Movement trajectory generation failed: {trajectory_result['error_message']}"
                            raise ValueError(error_msg)
                    
                    elif action_type == "stationary":
                        # Generate stationary trajectory (maintain planned time)
                        planned_duration = (datetime.strptime(end_time_str, "%Y-%m-%d %H:%M:%S") - datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")).total_seconds()
                        trajectory_result = self._generate_stationary_trajectory_with_duration(
                            action_step, nav, start_time_str, planned_duration
                        )
                        
                        if trajectory_result["success"]:
                            trajectory_obj = trajectory_result["trajectory"]
                            trajectories.append(trajectory_obj)

                            logger.debug(f"Stationary trajectory generated successfully, duration: {trajectory_result['actual_duration']:.1f}s")
                        else:
                            error_msg = f"Stationary trajectory generation failed: {trajectory_result['error_message']}"
                            raise ValueError(error_msg)
                    
                    elif action_type == "absent":
                        # Skip trajectory generation for absent type, but maintain time continuity
                        logger.debug("Skipping absent type action step (no trajectory), maintaining time continuity")
                        # Don't generate trajectory, but maintain original time of action step, no time offset
                    
                    else:
                        error_msg = f"Unknown action type: {action_type}"
                        raise ValueError(error_msg)
                        
                except Exception as e:
                    error_msg = f"   - Trajectory generation exception: {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    raise ValueError(error_msg)
            
            # Update trajectories and adjusted action steps in state
            state["trajectories"] = trajectories
            state["action_steps"] = adjusted_action_steps
            
            # Update event and activity planning times based on adjusted action steps
            logger.info("Adjusting event and activity planning times based on action steps...")
            state = self._update_time_hierarchy(state)
            
            logger.info(f"Trajectory generation completed! Total generated {len(trajectories)} trajectories")
            
            return state
            
        except Exception as e:
            error_msg = f"Trajectory generation failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ValueError(error_msg)

    def _generate_movement_trajectory(self, action_step, nav: Navigation, start_time: str) -> Dict[str, Any]:
        """Generate trajectory for movement action"""
        try:
            # Get movement details
            movement_detail = action_step.movement_detail
            if not movement_detail:
                return {"success": False, "error_message": "Missing movement action coordinate information"}
            
            # Extract start and end positions
            start_pos = (movement_detail.start_position.x, movement_detail.start_position.y)
            end_pos = (movement_detail.end_position.x, movement_detail.end_position.y)
            
            # Generate navigation trajectory
            nav_result = nav.generate_navigation_trajectory(
                start=start_pos,
                end=end_pos,
                start_time=start_time,
                speed=self.speed_range,
                dt=self.dt,
                potential=True,
                potential_k=1.0,
                potential_decay=1,
                potential_rand=True,
                smooth=True,
                speed_start=0.5,
                speed_end=0.5,
                unit=self.unit
            )
            
            if not nav_result["success"]:
                return {"success": False, "error_message": nav_result["error_message"]}
            
            # Convert to trajectory data model
            trajectory = Trajectory(
                trajectory_id=action_step.action_step_id,
                action_step_id=action_step.action_step_id,
                points=[TrajectoryPoint(x=x, y=y, timestamp=timestamp) for x, y, timestamp in nav_result["trajectory"]]
            )

            return {"success": True, "trajectory": trajectory, "actual_duration": nav_result["total_duration"]}
            
        except Exception as e:
            error_msg = f"   - Movement trajectory generation exception: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise   

    def _generate_stationary_trajectory_with_duration(self, action_step, nav: Navigation, start_time: str, duration: float) -> Dict[str, Any]:
        """Generate trajectory for stationary action (using specified duration)"""
        try:
            # Get stationary details
            stationary_detail = action_step.stationary_detail
            if not stationary_detail:
                return {"success": False, "error_message": "Missing stationary action coordinate information"}
            
            # Extract center position and fluctuation radius
            center_pos = (stationary_detail.center_position.x, stationary_detail.center_position.y)
            fluctuation_radius = stationary_detail.fluctuation_radius
            
            # Generate stationary trajectory
            stationary_result = nav.generate_stationary_trajectory(
                center_position=center_pos,
                fluctuation_radius=fluctuation_radius,
                start_time=start_time,
                duration=duration,
                dt=self.stationary_dt,
                unit=self.unit
            )
            
            # Convert to trajectory data model
            trajectory = Trajectory(
                trajectory_id=action_step.action_step_id,
                action_step_id=action_step.action_step_id,
                points=[TrajectoryPoint(x=x, y=y, timestamp=timestamp) for x, y, timestamp in stationary_result["trajectory"]]
            )
            
            return {"success": True, "trajectory": trajectory, "actual_duration": stationary_result["total_duration"]}
            
        except Exception as e:
            error_msg = f"   - Stationary trajectory generation exception: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise Exception(error_msg)   

    def _adjust_current_and_subsequent_steps(self, action_steps: List, current_index: int, time_offset: float):
        """Adjust current action end time and time offset for all subsequent actions"""
        # Adjust current action's end time
        current_step = action_steps[current_index]
        
        try:
            current_end_dt = datetime.strptime(current_step.end_time, "%Y-%m-%d %H:%M:%S")
            new_end_dt = current_end_dt + timedelta(seconds=time_offset)
            current_step.end_time = new_end_dt.strftime("%Y-%m-%d %H:%M:%S")
            
            # Adjust times for all subsequent actions (both start and end times are offset)
            for i in range(current_index + 1, len(action_steps)):
                subsequent_step = action_steps[i]
                
                # Adjust start time
                start_dt = datetime.strptime(subsequent_step.start_time, "%Y-%m-%d %H:%M:%S")
                new_start_dt = start_dt + timedelta(seconds=time_offset)
                subsequent_step.start_time = new_start_dt.strftime("%Y-%m-%d %H:%M:%S")
                
                # Adjust end time
                end_dt = datetime.strptime(subsequent_step.end_time, "%Y-%m-%d %H:%M:%S")
                new_end_dt = end_dt + timedelta(seconds=time_offset)
                subsequent_step.end_time = new_end_dt.strftime("%Y-%m-%d %H:%M:%S")
                
        except Exception as e:
            error_msg = f"Time adjustment exception: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise Exception(error_msg)

    def _update_time_hierarchy(self, state: SimulationState) -> SimulationState:
        """Update event and activity planning time hierarchy based on action steps"""
        try:
            action_steps = state.get("action_steps", [])
            events = state.get("events", [])
            activity_plans = state.get("activity_plans", [])
            
            if not action_steps or not events or not activity_plans:
                return state
            
            # 1. Update event times based on action steps
            updated_events = []
            for event in events:
                # Find all action steps corresponding to this event
                related_action_steps = [
                    step for step in action_steps 
                    if step.event_id == event.event_id
                ]
                
                if related_action_steps:
                    # Use action step time range
                    start_times = [step.start_time for step in related_action_steps]
                    end_times = [step.end_time for step in related_action_steps]
                    
                    # Update event times
                    updated_event = event.model_copy()
                    updated_event.start_time = min(start_times)
                    updated_event.end_time = max(end_times)
                    updated_events.append(updated_event)
                else:
                    updated_events.append(event)
            
            # 2. Update activity planning times based on events
            updated_activity_plans = []
            for activity_plan in activity_plans:
                # Find all events corresponding to this activity
                related_events = [
                    e for e in updated_events 
                    if e.activity_id == activity_plan.activity_id
                ]
                
                if related_events:
                    # Use event time range
                    start_times = [e.start_time for e in related_events]
                    end_times = [e.end_time for e in related_events]
                    
                    # Update activity planning times
                    updated_activity_plan = activity_plan.model_copy()
                    updated_activity_plan.start_time = min(start_times)
                    updated_activity_plan.end_time = max(end_times)
                    updated_activity_plans.append(updated_activity_plan)
                else:
                    updated_activity_plans.append(activity_plan)
            
            # Update state
            state["events"] = updated_events
            state["activity_plans"] = updated_activity_plans
            
            logger.info("Time hierarchy update completed")
            return state
            
        except Exception as e:
            error_msg = f"Time hierarchy update exception: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise Exception(error_msg)
