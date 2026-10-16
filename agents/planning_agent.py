from typing import List
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from agents.data_models import ActivityPlan, ValidationResult, SimulationState
from utils.tools import convert_to_json
from utils.stats import extract_usage
import logging
from .prompts import get_planning_prompts, get_requirements

logger = logging.getLogger(__name__)

# Activity planning output model
class ActivityPlanOutput(BaseModel):
    """Activity planning output model"""
    activity_plans: List[ActivityPlan] = Field(description="Activity planning list")

class PlanningAgent:
    """Overall planning agent"""
    
    def __init__(self, planning_model: str = "gpt-4o", validation_model: str = "gpt-4o", correction_model: str = "gpt-4o", 
                 planning_t: float = 0.7, validation_t: float = 0.7, correction_t: float = 0.7):
        """Initialize planning agent"""
        self.requirements = get_requirements()['activity']
        prompts = get_planning_prompts()
        self.planning_chain = prompts['planning'] | ChatOpenAI(model=planning_model, temperature=planning_t, max_retries=5).with_structured_output(ActivityPlanOutput, method="function_calling", include_raw=True)
        self.validation_chain = prompts['validation'] | ChatOpenAI(model=validation_model, temperature=validation_t, max_retries=5).with_structured_output(ValidationResult, method="function_calling", include_raw=True)
        self.correction_chain = prompts['correction'] | ChatOpenAI(model=correction_model, temperature=correction_t, max_retries=5).with_structured_output(ActivityPlanOutput, method="function_calling", include_raw=True)
        
    def generate_plans(self, state: SimulationState) -> SimulationState:
        """
        Generate activity plans
        """
        try:
            result = self.planning_chain.invoke({
                "activity_planning_requirements": self.requirements,
                "resident_profile": state["resident_profile"],
                "house_layout_yaml": state["house_layout_yaml"]
            })
            
            # Collect statistics (include_raw=True returns {"raw": AIMessage, "parsed": ...})
            inp, out = extract_usage(result)
            if state.get("usage_stats"):
                state["usage_stats"].plan_gen.add(inp, out)
            
            state["activity_plans"] = result["parsed"].activity_plans
            state["status"] = "plans_generated"
            return state
            
        except Exception as e:
            error_msg = f"Activity plan generation failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ValueError(error_msg)
    
    def validate_plans(self, state: SimulationState) -> SimulationState:
        """
        Validate activity plans
        """
        try:
            result = self.validation_chain.invoke({
                "activity_planning_requirements": self.requirements,
                "resident_profile": state["resident_profile"],
                "house_layout_yaml": state["house_layout_yaml"],
                "activity_plans_json": convert_to_json(state["activity_plans"])
            })
            
            # Collect statistics
            inp, out = extract_usage(result)
            if state.get("usage_stats"):
                state["usage_stats"].plan_val.add(inp, out)
            
            parsed = result["parsed"]
            state["activity_validation"] = {
                "is_valid": parsed.is_valid,
                "correction_content": parsed.correction_content
            }
            
            if parsed.is_valid:
                state["status"] = "plans_validated"
            else:
                state["status"] = "plans_correction"
            
            return state
            
        except Exception as e:
            error_msg = f"Activity plan validation failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ValueError(error_msg)
    
    def correct_plans(self, state: SimulationState) -> SimulationState:
        """
        Correct activity plans
        """
        try:
            assert state.get("activity_plans") and state.get("activity_validation") and not state.get("activity_validation")["is_valid"]
            
            result = self.correction_chain.invoke({
                "activity_planning_requirements": self.requirements,
                "resident_profile": state["resident_profile"],
                "house_layout_yaml": state["house_layout_yaml"],
                "original_activity_plans_json": convert_to_json(state["activity_plans"]),
                "correction_content": state["activity_validation"]["correction_content"]
            })
            
            # Collect statistics
            inp, out = extract_usage(result)
            if state.get("usage_stats"):
                state["usage_stats"].plan_cor.add(inp, out)
            
            state["activity_plans"] = result["parsed"].activity_plans
            state["status"] = "plans_corrected"
            state["activity_validation"] = None
            return state
            
        except Exception as e:
            error_msg = f"Activity plan correction failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ValueError(error_msg)
