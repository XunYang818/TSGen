from typing import List
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from agents.data_models import Event, ValidationResult, SimulationState
from utils.tools import convert_to_json
from utils.stats import extract_usage
import logging
from .prompts import get_event_prompts, get_requirements

logger = logging.getLogger(__name__)

# Event output model
class EventOutput(BaseModel):
    """Event output model"""
    events: List[Event] = Field(description="Event list")

class EventAgent:
    """Event agent"""
    
    def __init__(self, generation_model: str = "gpt-4o-2024-11-20", validation_model: str = "gpt-4o-2024-11-20", correction_model: str = "gpt-4o-2024-11-20",
                 generation_t: float = 0.8, validation_t: float = 0.3, correction_t: float = 0.7,
                 batch_size: int = 5, context_size: int = 5):
        """
        Initialize event agent
        
        Args:
            batch_size: Batch size for activities processing, default 5
            context_size: Event context size, default 5
        """

        self.batch_size = batch_size
        self.context_size = context_size
        self.requirements = get_requirements()['event']
        # Build processing chains (include_raw=True to get usage_metadata)
        prompts = get_event_prompts()
        self.generation_chain = prompts['generation'] | ChatOpenAI(model=generation_model, temperature=generation_t, max_retries=5).with_structured_output(EventOutput, method="function_calling", include_raw=True)
        self.validation_chain = prompts['validation'] | ChatOpenAI(model=validation_model, temperature=validation_t, max_retries=5).with_structured_output(ValidationResult, method="function_calling", include_raw=True)
        self.correction_chain = prompts['correction'] | ChatOpenAI(model=correction_model, temperature=correction_t, max_retries=5).with_structured_output(EventOutput, method="function_calling", include_raw=True)
    
    def generate_events(self, state: SimulationState) -> SimulationState:
        """
        Generate events for current activity group
        """
        try:
            activity_plans = state.get("activity_plans", [])
            current_index = batch_start = state.get("current_batch_start", 0)
            batch_end = state.get("current_batch_end", 0)
            assert current_index == batch_start and current_index < len(activity_plans)
            
            current_batch = activity_plans[batch_start:batch_end]
            previous_events_context = self._prepare_events_context(state)
            
            result = self.generation_chain.invoke({
                "event_requirements": self.requirements,
                "resident_profile": state["resident_profile"],
                "house_layout_yaml": state["house_layout_yaml"],
                "activities_batch": convert_to_json(current_batch),
                "previous_events_context": previous_events_context,
                "context_size": self.context_size
            })
            
            # Collect statistics
            inp, out = extract_usage(result)
            if state.get("usage_stats"):
                state["usage_stats"].event_gen.add(inp, out)
            
            new_events = result["parsed"].events
            logger.debug(f"Generated {len(new_events)} new events")
            
            if "events" not in state or state["events"] is None:
                state["events"] = []
            
            state["events"].extend(new_events)
            state["status"] = "events_generated"
            logger.debug(f"Current batch completed, more activities to process")
            return state
            
        except Exception as e:
            error_msg = f"Event generation failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise
 


    def validate_events(self, state: SimulationState) -> SimulationState:
        """
        Validate events for current activity group
        """
        try:
            activity_plans = state.get("activity_plans", [])
            batch_start = state.get("current_batch_start", 0)
            batch_end = state.get("current_batch_end", 0)
            events = state.get("events", [])
            assert activity_plans and batch_start < len(activity_plans)       
            
            current_batch = activity_plans[batch_start:batch_end]
            activities_batch = convert_to_json(current_batch)
            batch_activity_ids = [activity.activity_id for activity in current_batch]
            
            current_batch_events = [
                event for event in events 
                if event.activity_id in batch_activity_ids
            ]
            
            if not current_batch_events:
                error_msg = "No events found for current batch to validate"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            events_json = convert_to_json(current_batch_events)
            previous_events_context = self._prepare_events_context(state, exclude_activity_ids=batch_activity_ids)
            
            result = self.validation_chain.invoke({
                "event_requirements": self.requirements,
                "resident_profile": state["resident_profile"],
                "house_layout_yaml": state["house_layout_yaml"],
                "activities_batch": activities_batch,
                "events_json": events_json,
                "previous_events_context": previous_events_context,
                "context_size": self.context_size
            })
            
            # Collect statistics
            inp, out = extract_usage(result)
            if state.get("usage_stats"):
                state["usage_stats"].event_val.add(inp, out)
            
            parsed = result["parsed"]
            state["event_validation"] = {
                "is_valid": parsed.is_valid,
                "correction_content": parsed.correction_content
            }
            
            if parsed.is_valid:
                state["status"] = "events_validated"
            else:
                state["status"] = "events_correction"
                
            return state
            
        except Exception as e:
            error_msg = f"Event validation failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ValueError(error_msg)



    def correct_events(self, state: SimulationState) -> SimulationState:
        """
        Correct events for current activity group
        """
        try:
            activity_plans = state.get("activity_plans", [])
            batch_start = state.get("current_batch_start", 0)
            batch_end = state.get("current_batch_end", 0)
            events = state.get("events", [])
            
            rule_validation_result = state.get("event_rule_validation")
            model_validation_result = state.get("event_validation")
            
            if rule_validation_result and not rule_validation_result["is_valid"]:
                validation_result = rule_validation_result
            elif model_validation_result and not model_validation_result["is_valid"]:
                validation_result = model_validation_result
            else:
                error_msg = "No validation result to correct"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            current_batch = activity_plans[batch_start:batch_end]
            activities_batch = convert_to_json(current_batch)
            batch_activity_ids = [activity.activity_id for activity in current_batch]
            
            current_batch_events = [
                event for event in events 
                if event.activity_id in batch_activity_ids
            ]
            
            previous_events_context = self._prepare_events_context(state, exclude_activity_ids=batch_activity_ids)
            
            result = self.correction_chain.invoke({
                "event_requirements": self.requirements,
                "resident_profile": state["resident_profile"],
                "house_layout_yaml": state["house_layout_yaml"],
                "activities_batch": activities_batch,
                "original_events_json": convert_to_json(current_batch_events),
                "correction_content": validation_result["correction_content"],
                "previous_events_context": previous_events_context,
                "context_size": self.context_size
            })
            
            # Collect statistics
            inp, out = extract_usage(result)
            if state.get("usage_stats"):
                state["usage_stats"].event_cor.add(inp, out)
            
            corrected_events = result["parsed"].events
            state["events"] = [
                event for event in events 
                if event.activity_id not in batch_activity_ids
            ]
            state["events"].extend(corrected_events)
            state["status"] = "events_corrected"
            state["event_validation"] = None
            state["event_rule_validation"] = None
            return state
            
        except Exception as e:
            error_msg = f"Event correction failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ValueError(error_msg)


    def _prepare_events_context(self, state: SimulationState, exclude_activity_ids: List[str] = None) -> str:
        """
        Prepare previously generated event context
        """
        events = state.get("events", [])
        if not events:
            return "[]"
        
        # Filter events (exclude events from specified activities)
        if exclude_activity_ids:
            filtered_events = [event for event in events if event.activity_id not in exclude_activity_ids]
        else:
            filtered_events = events
        
        if not filtered_events:
            return "[]"
        
        # Only take the most recent context_size events as context
        recent_events = filtered_events[-self.context_size:] if len(filtered_events) > self.context_size else filtered_events
        
        return convert_to_json(recent_events)
    
    def rule_validate_events(self, state: SimulationState) -> SimulationState:
        """Rule-based validation of events for current activity group"""
        try:
            # Get current batch data
            batch_start = state.get("current_batch_start", 0)
            batch_end = state.get("current_batch_end", 0)
            activity_plans = state.get("activity_plans", [])
            events = state.get("events", [])
            
            if not activity_plans or batch_start >= len(activity_plans):
                error_msg = "Current batch activities not found"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Get current batch activity IDs and events
            batch_activity_ids = [activity.activity_id for activity in activity_plans[batch_start:batch_end]]
            current_batch_events = [event for event in events if event.activity_id in batch_activity_ids]
            
            if not current_batch_events:
                error_msg = "Current batch events not found"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Execute ID validations
            validation_errors = []
            validation_errors.extend(self._validate_event_ids(events, batch_activity_ids, current_batch_events))
            
            # Set validation result
            state["event_rule_validation"] = {
                "is_valid": not validation_errors,
                "correction_content": "Rule validation failed:\n" + "\n".join(validation_errors) if validation_errors else ""
            }
            
            state["status"] = "events_rule_validated" if not validation_errors else "events_correction"
            return state
            
        except Exception as e:
            error_msg = f"Event rule validation failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise
    
    def _validate_event_ids(self, all_events: List, batch_activity_ids: List[str], current_batch_events: List) -> List[str]:
        """
        Validate event IDs:
        1. ID must be numeric string format
        2. ID must be incrementing (both within batch and compared to previous batch)
        """
        errors = []
        
        # 1. Validate ID format (must be numeric string)
        for event in current_batch_events:
            if not event.event_id.isdigit():
                errors.append(f"Event ID format error: Event '{event.event_name}' has invalid ID '{event.event_id}', must be numeric string (e.g., '1', '2', '3')")
        
        # If format validation failed, return early
        if errors:
            return errors
        
        # 2. Validate ID incrementing within current batch
        current_batch_sorted = sorted(current_batch_events, key=lambda e: int(e.event_id))
        for i in range(1, len(current_batch_sorted)):
            prev_id = int(current_batch_sorted[i-1].event_id)
            curr_id = int(current_batch_sorted[i].event_id)
            if curr_id <= prev_id:
                errors.append(f"Event ID order error: Event '{current_batch_sorted[i].event_name}' ID '{curr_id}' is not greater than previous event ID '{prev_id}' within current batch")
            if curr_id != prev_id + 1:
                errors.append(f"Event ID continuity error: Event '{current_batch_sorted[i].event_name}' ID '{curr_id}' should be '{prev_id + 1}' (IDs must be consecutive)")
        
        # 3. Check for duplicate IDs within current batch
        current_ids = [event.event_id for event in current_batch_events]
        seen_ids = set()
        for event_id in current_ids:
            if event_id in seen_ids:
                errors.append(f"Event ID duplicate error: ID '{event_id}' appears multiple times in current batch")
            seen_ids.add(event_id)
        
        # 4. Validate continuity with previous batch
        # Get events from previous batches (not in current batch)
        previous_events = [event for event in all_events if event.activity_id not in batch_activity_ids]
        
        if previous_events:
            # Find the maximum ID from previous batches
            try:
                max_previous_id = max(int(event.event_id) for event in previous_events if event.event_id.isdigit())
                min_current_id = min(int(event.event_id) for event in current_batch_events)
                
                if min_current_id <= max_previous_id:
                    errors.append(f"Event ID continuity error: Current batch minimum ID '{min_current_id}' is not greater than previous batch maximum ID '{max_previous_id}'. IDs must be strictly incrementing across batches.")
                
                if min_current_id != max_previous_id + 1:
                    errors.append(f"Event ID continuity error: Current batch should start with ID '{max_previous_id + 1}', but starts with '{min_current_id}'. IDs must be consecutive across batches.")
            except ValueError:
                pass  # If conversion fails, format validation already caught this
        else:
            # First batch: should start from 1
            min_current_id = min(int(event.event_id) for event in current_batch_events)
            if min_current_id != 1:
                errors.append(f"Event ID start error: First batch should start with ID '1', but starts with '{min_current_id}'")
        
        return errors
