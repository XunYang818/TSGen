from typing import List
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from agents.data_models import ActionStep, ValidationResult, SimulationState
from utils.tools import convert_to_json
from utils.stats import extract_usage
import logging
import math
from .prompts import get_action_step_prompts, get_requirements

logger = logging.getLogger(__name__)

# Action step output model
class ActionStepOutput(BaseModel):
    """Action step output model"""
    action_steps: List[ActionStep] = Field(description="Action step list")

class ActionStepAgent:
    """Action step agent"""
    
    def __init__(self, generation_model: str = "gpt-4o-2024-11-20", validation_model: str = "gpt-4o-2024-11-20", correction_model: str = "gpt-4o-2024-11-20",
                 generation_t: float = 0.8, validation_t: float = 0.3, correction_t: float = 0.7,
                 batch_size: int = 3, context_size: int = 5):
        """
        Initialize action step agent
        
        Args:
            batch_size: Batch size for events processing, default 3
            context_size: Action step context size, default 5
        """
 
        self.batch_size = batch_size
        self.context_size = context_size
        self.requirements = get_requirements()['action_step']
        # Build processing chains (include_raw=True to get usage_metadata)
        prompts = get_action_step_prompts()
        self.generation_chain = prompts['generation'] | ChatOpenAI(model=generation_model, temperature=generation_t, max_retries=5).with_structured_output(ActionStepOutput, include_raw=True)
        self.validation_chain = prompts['validation'] | ChatOpenAI(model=validation_model, temperature=validation_t, max_retries=5).with_structured_output(ValidationResult, include_raw=True)
        self.correction_chain = prompts['correction'] | ChatOpenAI(model=correction_model, temperature=correction_t, max_retries=5).with_structured_output(ActionStepOutput, include_raw=True)
    
    def generate_action_steps(self, state: SimulationState) -> SimulationState:
        """
        Generate action steps for current event group
        """
        try:
            events = state.get("events", [])
            batch_start = state.get("current_batch_start", 0)
            batch_end = state.get("current_batch_end", 0)
            current_batch = events[batch_start:batch_end]
            
            action_steps_context = self._prepare_action_steps_context(state)

            result = self.generation_chain.invoke({
                "action_step_requirements": self.requirements,
                "resident_profile": state["resident_profile"],
                "house_layout_yaml": state["house_layout_yaml"],
                "events_batch": convert_to_json(current_batch),
                "previous_action_steps_context": action_steps_context,
                "context_size": self.context_size
            })
            
            # Collect statistics
            inp, out = extract_usage(result)
            if state.get("usage_stats"):
                state["usage_stats"].action_gen.add(inp, out)
            
            new_steps = result["parsed"].action_steps
            logger.debug(f"Generated {len(new_steps)} new action steps")
            
            if not state.get("action_steps") or state["action_steps"] is None:
                state["action_steps"] = []
            
            state["action_steps"].extend(new_steps)
            logger.debug(f"Action steps list updated, total: {len(state['action_steps'])}")
            return state
            
        except Exception as e:
            logger.error(f"Action step generation failed: {str(e)}", exc_info=True)
            raise

    def validate_action_steps(self, state: SimulationState) -> SimulationState:
        """
        Validate action steps for current event group
        """
        try:
            events = state.get("events", [])
            batch_start = state.get("current_batch_start", 0)
            batch_end = state.get("current_batch_end", 0)
            action_steps = state.get("action_steps", [])

            current_batch = events[batch_start:batch_end]
            batch_event_ids = [event.event_id for event in current_batch]
            
            current_batch_action_steps = [
                step for step in action_steps 
                if step.event_id in batch_event_ids
            ]
            
            action_steps_context = self._prepare_action_steps_context(state, exclude_event_ids=batch_event_ids)
            
            result = self.validation_chain.invoke({
                "action_step_requirements": self.requirements,
                "resident_profile": state["resident_profile"],
                "house_layout_yaml": state["house_layout_yaml"],
                "events_batch": convert_to_json(current_batch),
                "action_steps_json": convert_to_json(current_batch_action_steps),
                "previous_action_steps_context": action_steps_context,
                "context_size": self.context_size
            })
            
            # Collect statistics
            inp, out = extract_usage(result)
            if state.get("usage_stats"):
                state["usage_stats"].action_val.add(inp, out)
            
            parsed = result["parsed"]
            state["action_validation"] = {
                "is_valid": parsed.is_valid,
                "correction_content": parsed.correction_content
            }
            
            if parsed.is_valid:
                state["status"] = "action_steps_validated"
            else:
                state["status"] = "action_steps_correction"
            
            return state
            
        except Exception as e:
            error_msg = f"Action step validation failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise

    def correct_action_steps(self, state: SimulationState) -> SimulationState:
        """
        Correct action steps for current event group
        """
        try:
            events = state.get("events", [])
            batch_start = state.get("current_batch_start", 0)
            batch_end = state.get("current_batch_end", 0)
            action_steps = state.get("action_steps", [])
            
            rule_validation_result = state.get("action_rule_validation")
            model_validation_result = state.get("action_validation")
            
            if rule_validation_result and not rule_validation_result["is_valid"]:
                validation_result = rule_validation_result
            elif model_validation_result and not model_validation_result["is_valid"]:
                validation_result = model_validation_result

            current_batch = events[batch_start:batch_end]
            batch_event_ids = [event.event_id for event in current_batch]
            
            current_batch_action_steps = [
                step for step in action_steps 
                if step.event_id in batch_event_ids
            ]
            
            previous_action_steps_context = self._prepare_action_steps_context(state, exclude_event_ids=batch_event_ids)
            
            result = self.correction_chain.invoke({
                "action_step_requirements": self.requirements,
                "resident_profile": state["resident_profile"],
                "house_layout_yaml": state["house_layout_yaml"],
                "events_batch": convert_to_json(current_batch),
                "original_action_steps_json": convert_to_json(current_batch_action_steps),
                "correction_content": validation_result["correction_content"],
                "previous_action_steps_context": previous_action_steps_context,
                "context_size": self.context_size
            })
            
            # Collect statistics
            inp, out = extract_usage(result)
            if state.get("usage_stats"):
                state["usage_stats"].action_cor.add(inp, out)
            
            corrected_steps = result["parsed"].action_steps
            state["action_steps"] = [
                step for step in action_steps 
                if step.event_id not in batch_event_ids
            ]
            state["action_steps"].extend(corrected_steps)
            state["status"] = "action_steps_corrected"
            state["action_validation"] = None
            state["action_rule_validation"] = None
            return state
            
        except Exception as e:
            error_msg = f"Action step correction failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise



    def _prepare_action_steps_context(self, state: SimulationState, exclude_event_ids: List[str] = None) -> str:
        """
        Prepare previously generated action steps context, including position connection information
        """
        action_steps = state.get("action_steps", [])
        
        if not action_steps:
            return "[]"
        
        # Determine event IDs to exclude
        exclude_ids = exclude_event_ids or []
        
        # Filter action steps (exclude action steps from specified events)
        filtered_steps = []
        for step in action_steps:
            if step.event_id not in exclude_ids:
                filtered_steps.append(step)
        
        if not filtered_steps:
            return "[]"
        
        # Take the most recent context_size action steps as context
        recent_steps = filtered_steps[-self.context_size:] if len(filtered_steps) > self.context_size else filtered_steps
        return convert_to_json(recent_steps)
    

    def rule_validate_action_steps(self, state: SimulationState) -> SimulationState:
        """Rule-based validation of action steps for current event group"""
        try:
            # Get current batch data
            batch_start = state.get("current_batch_start", 0)
            batch_end = state.get("current_batch_end", 0)
            events = state.get("events", [])
            action_steps = state.get("action_steps", [])
            
            if not events or batch_start >= len(events):
                error_msg = "Current batch events not found"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Get current batch event IDs and action steps
            batch_event_ids = [event.event_id for event in events[batch_start:batch_end]]
            current_batch_steps = [step for step in action_steps if step.event_id in batch_event_ids]
            
            if not current_batch_steps:
                error_msg = "Current batch action steps not found"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Execute all validations
            validation_errors = []
            
            # 0. ID validation (format + incrementing)
            validation_errors.extend(self._validate_action_step_ids(action_steps, batch_event_ids, current_batch_steps))
            
            # 1. Continuity validation (between batches + within batch)
            validation_errors.extend(self._validate_continuity(action_steps, batch_event_ids, current_batch_steps))
            
            # 2. Single-step validation (action_type and detail field consistency + fluctuation radius + navigation feasibility)
            for step in current_batch_steps:
                validation_errors.extend([
                    error for error in [
                        self._validate_action_detail_consistency(step),
                        self._validate_fluctuation_radius(step),
                        self._validate_navigation_feasibility(step, state["house_layout_yaml"])
                    ] if error
                ])
            
            # Set validation result
            state["action_rule_validation"] = {
                "is_valid": not validation_errors,
                "correction_content": "Rule validation failed:\n" + "\n".join(validation_errors) if validation_errors else ""
            }
            
            state["status"] = "action_steps_rule_validated" if not validation_errors else "action_steps_correction"
            return state
            
        except Exception as e:
            error_msg = f"Rule validation failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise

    def _validate_action_step_ids(self, all_steps, batch_event_ids, current_batch_steps) -> List[str]:
        """
        Validate action step IDs:
        1. ID must be numeric string format
        2. ID must be incrementing (both within batch and compared to previous batch)
        """
        errors = []
        
        # 1. Validate ID format (must be numeric string)
        for step in current_batch_steps:
            if not step.action_step_id.isdigit():
                errors.append(f"Action step ID format error: Action step '{step.action_step_name}' has invalid ID '{step.action_step_id}', must be numeric string (e.g., '1', '2', '3')")
        
        # If format validation failed, return early
        if errors:
            return errors
        
        # 2. Validate ID incrementing within current batch
        current_batch_sorted = sorted(current_batch_steps, key=lambda s: int(s.action_step_id))
        for i in range(1, len(current_batch_sorted)):
            prev_id = int(current_batch_sorted[i-1].action_step_id)
            curr_id = int(current_batch_sorted[i].action_step_id)
            if curr_id <= prev_id:
                errors.append(f"Action step ID order error: Action step '{current_batch_sorted[i].action_step_name}' ID '{curr_id}' is not greater than previous action step ID '{prev_id}' within current batch")
            if curr_id != prev_id + 1:
                errors.append(f"Action step ID continuity error: Action step '{current_batch_sorted[i].action_step_name}' ID '{curr_id}' should be '{prev_id + 1}' (IDs must be consecutive)")
        
        # 3. Check for duplicate IDs within current batch
        current_ids = [step.action_step_id for step in current_batch_steps]
        seen_ids = set()
        for step_id in current_ids:
            if step_id in seen_ids:
                errors.append(f"Action step ID duplicate error: ID '{step_id}' appears multiple times in current batch")
            seen_ids.add(step_id)
        
        # 4. Validate continuity with previous batch
        # Get action steps from previous batches (not in current batch)
        previous_steps = [step for step in all_steps if step.event_id not in batch_event_ids]
        
        if previous_steps:
            # Find the maximum ID from previous batches
            try:
                max_previous_id = max(int(step.action_step_id) for step in previous_steps if step.action_step_id.isdigit())
                min_current_id = min(int(step.action_step_id) for step in current_batch_steps)
                
                if min_current_id <= max_previous_id:
                    errors.append(f"Action step ID continuity error: Current batch minimum ID '{min_current_id}' is not greater than previous batch maximum ID '{max_previous_id}'. IDs must be strictly incrementing across batches.")
                
                if min_current_id != max_previous_id + 1:
                    errors.append(f"Action step ID continuity error: Current batch should start with ID '{max_previous_id + 1}', but starts with '{min_current_id}'. IDs must be consecutive across batches.")
            except ValueError:
                pass  # If conversion fails, format validation already caught this
        else:
            # First batch: should start from 1
            min_current_id = min(int(step.action_step_id) for step in current_batch_steps)
            if min_current_id != 1:
                errors.append(f"Action step ID start error: First batch should start with ID '1', but starts with '{min_current_id}'")
        
        return errors

    def _validate_continuity(self, all_steps, batch_event_ids, current_batch_steps):
        """Validate continuity (between batches + within batch)"""
        errors = []
        
        # Find the position of the first action step in current batch
        batch_start_index = next(
            (i for i, step in enumerate(all_steps) if step.event_id in batch_event_ids), 
            None
        )
        
        # Validate continuity between batches
        if batch_start_index and batch_start_index > 0:
            prev_step = all_steps[batch_start_index-1]
            current_step = all_steps[batch_start_index]
            errors.extend([error for error in [
                self._validate_time_continuity(prev_step, current_step),
                self._validate_space_continuity(prev_step, current_step)
            ] if error])
        
        # Validate continuity within batch
        for i in range(1, len(current_batch_steps)):
            prev_step = current_batch_steps[i-1]
            current_step = current_batch_steps[i]
            errors.extend([error for error in [
                self._validate_time_continuity(prev_step, current_step),
                self._validate_space_continuity(prev_step, current_step)
            ] if error])
        
        return errors

    def _validate_time_continuity(self, prev_step, current_step) -> str:
        """Validate time continuity"""
        try:
            # Time must match exactly
            if prev_step.end_time != current_step.start_time:
                return f"Time continuity error: Action step '{current_step.action_step_id}' start time ({current_step.start_time}) does not match previous action step '{prev_step.action_step_id}' end time ({prev_step.end_time}); possible time discontinuity or missing action steps"
            return ""
        except Exception as e:
            return f"Time continuity validation exception: {str(e)}"

    def _validate_space_continuity(self, prev_step, current_step) -> str:
        """Validate space continuity"""
        try:
            prev_end_position = self._get_step_position(prev_step, 'end')
            current_start_position = self._get_step_position(current_step, 'start')
            
            # Skip space validation if either position is None (e.g., absent type)
            if prev_end_position is None or current_start_position is None:
                return ""
            
            # Calculate distance (meters)
            dx = current_start_position[0] - prev_end_position[0]
            dy = current_start_position[1] - prev_end_position[1]
            distance = math.sqrt(dx*dx + dy*dy) / 10  # Convert to meters (original unit is decimeters)
            
            if distance > 0.5:
                prev_id = prev_step.action_step_id
                current_id = current_step.action_step_id
                return f"Space continuity error: Action step '{current_id}' start position {current_start_position} is too far from previous action step '{prev_id}' end position {prev_end_position}, distance is {distance:.2f} meters; possible position discontinuity or missing action steps"
            
            return ""
        except Exception as e:
            return f"Space continuity validation exception: {str(e)}"

    def _validate_action_detail_consistency(self, step) -> str:
        """Validate consistency between action_type and corresponding detail field"""
        try:
            if step.action_type == 'movement':
                if not step.movement_detail:
                    return f"Action type consistency error: Action step '{step.action_step_id}' has action_type 'movement', but movement_detail is empty"
            
            elif step.action_type == 'stationary':
                if not step.stationary_detail:
                    return f"Action type consistency error: Action step '{step.action_step_id}' has action_type 'stationary', but stationary_detail is empty"
            
            elif step.action_type == 'absent':
                if not step.absent_detail:
                    return f"Action type consistency error: Action step '{step.action_step_id}' has action_type 'absent', but absent_detail is empty"
            
            return ""
        except Exception as e:
            return f"Action type consistency validation exception: {str(e)}"

    def _validate_fluctuation_radius(self, step) -> str:
        """Validate fluctuation radius for stationary actions"""
        try:
            if step.action_type != 'stationary':
                return ""
            
            # Get fluctuation radius
            if not step.stationary_detail:
                return ""
            
            fluctuation_radius = step.stationary_detail.fluctuation_radius
            if fluctuation_radius is None:
                return ""
            
            if not (0.01 <= fluctuation_radius <= 0.5):
                return f"Fluctuation radius error: Action step '{step.action_step_id}' fluctuation radius {fluctuation_radius} meters is not within reasonable range (should be between 0.01-0.5 meters)"
            
            return ""
        except Exception as e:
            return f"Fluctuation radius validation exception: {str(e)}"

    def _validate_navigation_feasibility(self, step, house_layout_yaml) -> str:
        """Validate navigation feasibility for movement actions"""
        try:
            if step.action_type != 'movement':
                return ""
            
            if not step.movement_detail:
                return ""
            
            # Extract start and end coordinates
            start_pos = (step.movement_detail.start_position.x, step.movement_detail.start_position.y)
            end_pos = (step.movement_detail.end_position.x, step.movement_detail.end_position.y)
            
            # Navigation validation
            from utils.navigation import Navigation
            nav = Navigation(house_layout_yaml)
            validation_result = nav.validate_navigation_path(start_pos, end_pos)
            
            if not validation_result["is_valid"]:
                return f"Navigation feasibility error: Action step '{step.action_step_id}' path is not feasible, {validation_result['error_message']}"
            
            return ""
        except Exception as e:
            return f"Navigation feasibility validation exception: {str(e)}"

    def _get_step_position(self, step, position_type):
        """Get action step position (start or end)"""
        try:
            if step.action_type == 'movement':
                if step.movement_detail:
                    if position_type == 'start':
                        pos = step.movement_detail.start_position
                    else:  # 'end'
                        pos = step.movement_detail.end_position
                    return (pos.x, pos.y)
                        
            elif step.action_type == 'stationary':
                if step.stationary_detail:
                    pos = step.stationary_detail.center_position
                    return (pos.x, pos.y)
                        
            elif step.action_type == 'absent':
                return None
                
        except Exception:
            pass
        
        return None