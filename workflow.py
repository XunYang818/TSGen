import os
import uuid
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError

from agents.data_models import SimulationState, create_initial_state
from agents.planning_agent import PlanningAgent
from agents.event_agent import EventAgent
from agents.action_step_agent import ActionStepAgent
from agents.trajectory_agent import TrajectoryAgent
from utils.stats import UsageStats

logger = logging.getLogger(__name__)

class TrajectorySimulationWorkflow:
    """Trajectory simulation workflow"""
    
    def __init__(self, recursion_limit: int = 1000, 
                 max_retry_count: int = 3,
                 agent_configs: Optional[Dict[str, Dict[str, Any]]] = None):
        """
        Initialize workflow
        
        Args:
            recursion_limit: Recursion limit, default 200
            agent_configs: Agent configuration dictionary containing parameters for each agent
        """
        self.recursion_limit = recursion_limit
        self.max_retry_count = max_retry_count
        self.agent_configs = agent_configs
        
        # Build workflow graph
        self.planning_agent = PlanningAgent(**self.agent_configs["planning"])
        self.event_agent = EventAgent(**self.agent_configs["event"])
        self.action_step_agent = ActionStepAgent(**self.agent_configs["action_step"])
        self.trajectory_agent = TrajectoryAgent(**self.agent_configs["trajectory"])
        self.workflow: StateGraph = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build LangGraph workflow"""
        workflow = StateGraph(SimulationState)
    
        # Activity planning nodes
        workflow.add_node("generate_plans", self._generate_plans_node)
        workflow.add_node("validate_plans", self._validate_plans_node)
        workflow.add_node("correct_plans", self._correct_plans_node)
        
        # Event nodes
        workflow.add_node("generate_events", self._generate_events_node)
        workflow.add_node("rule_validate_events", self._rule_validate_events_node)
        workflow.add_node("validate_events", self._validate_events_node)
        workflow.add_node("correct_events", self._correct_events_node)
        
        # Action step nodes
        workflow.add_node("generate_action_steps", self._generate_action_steps_node)
        workflow.add_node("rule_validate_action_steps", self._rule_validate_action_steps_node) 
        workflow.add_node("validate_action_steps", self._validate_action_steps_node)
        workflow.add_node("correct_action_steps", self._correct_action_steps_node)
        
        # Trajectory generation nodes
        workflow.add_node("generate_trajectories", self._generate_trajectories_node)
    
        # Set entry point
        workflow.set_entry_point("generate_plans")
        
        # Activity planning workflow
        workflow.add_edge("generate_plans", "validate_plans")
        workflow.add_edge("correct_plans", "validate_plans")
        workflow.add_conditional_edges(
            "validate_plans",
            lambda state: state.get("status"),
            {"plans_validated": "generate_events", "plans_correction": "correct_plans"}
        )
        
        # Event workflow
        workflow.add_edge("generate_events", "rule_validate_events")
        workflow.add_conditional_edges(
            "rule_validate_events",
            lambda state: state.get("status"),
            {"events_rule_validated": "validate_events", "events_correction": "correct_events"}
        )
        workflow.add_edge("correct_events", "rule_validate_events")  # Re-validate after correction
        workflow.add_conditional_edges(
            "validate_events",
            lambda state: state.get("status"),
            {"events_validated": "generate_events", "events_correction": "correct_events", "all_generated": "generate_action_steps"}
        )
        
        # Action step workflow
        workflow.add_edge("generate_action_steps", "rule_validate_action_steps")
        workflow.add_conditional_edges(
            "rule_validate_action_steps",
            lambda state: state.get("status"),
            {"action_steps_rule_validated": "validate_action_steps", "action_steps_correction": "correct_action_steps"}
        )
        workflow.add_edge("correct_action_steps", "rule_validate_action_steps")  # Re-validate after correction
        workflow.add_conditional_edges(
            "validate_action_steps",
            lambda state: state.get("status"),
            {"action_steps_validated": "generate_action_steps", "action_steps_correction": "correct_action_steps", "all_generated": "generate_trajectories"}
        )
        
        # Trajectory generation workflow
        workflow.add_edge("generate_trajectories", END)
        
        return workflow.compile(checkpointer=MemorySaver())
    
    def _generate_plans_node(self, state: SimulationState) -> SimulationState:
        logger.info("Generating activity plans...")
        state["is_fresh_batch"] = True
        state = self.planning_agent.generate_plans(state)
        return state

    def _validate_plans_node(self, state: SimulationState) -> SimulationState:
        state = self.planning_agent.validate_plans(state)
        if state["status"] == "plans_validated":
            # Count pass-once (generated content passes validation without correction)
            if state.get("is_fresh_batch") and state.get("usage_stats"):
                state["usage_stats"].plan_pass_once += 1
        else:
            logger.info(f"Activity plans validation failed: {state['activity_validation']['correction_content']}")
        return state
    
    def _correct_plans_node(self, state: SimulationState) -> SimulationState:
        state["is_fresh_batch"] = False
        state = self.planning_agent.correct_plans(state)
        return state
    
    def _generate_events_node(self, state: SimulationState) -> SimulationState:
        """Generate events node"""
        if state.get("current_batch_end", 0) == 0:
            state["current_batch_start"] = 0
            state["current_batch_end"] = self.agent_configs["event"]["batch_size"]
        
        state["is_fresh_batch"] = True
        logger.info(f"Generating events ({state['current_batch_start'] + 1}-{state['current_batch_end']}/{len(state['activity_plans'])})...")
        return self.event_agent.generate_events(state)
    
    def _validate_events_node(self, state: SimulationState) -> SimulationState:
        """Validate events node"""
        state = self.event_agent.validate_events(state)
        
        if state["status"] == "events_validated":
            # Count pass-once (generated content passes validation without correction)
            if state.get("is_fresh_batch") and state.get("usage_stats"):
                state["usage_stats"].event_pass_once += 1

            activity_plans = state.get("activity_plans", [])
            batch_size = self.agent_configs["event"]["batch_size"]
            
            if state["current_batch_end"] >= len(activity_plans):
                state["current_batch_start"] = 0
                state["current_batch_end"] = 0
                state["status"] = "all_generated"
                return state
            state["current_batch_start"] = state["current_batch_end"]
            state["current_batch_end"] = min(state["current_batch_end"] + batch_size, len(activity_plans)) 
        else:
            correction = state.get("event_validation", {}).get("correction_content", "")
            logger.info(f"Events validation failed: {correction}")
        
        return state
    
    def _rule_validate_events_node(self, state: SimulationState) -> SimulationState:
        """Rule-based validation of events node"""
        state = self.event_agent.rule_validate_events(state)
        if state["status"] == "events_correction":
            logger.info(f"Events rule validation failed: {state['event_rule_validation']['correction_content']}")
        return state

    def _correct_events_node(self, state: SimulationState) -> SimulationState:
        """Correct events node"""
        state["is_fresh_batch"] = False
        logger.info("Correcting events...")
        state = self.event_agent.correct_events(state)            
        return state

    def _generate_action_steps_node(self, state: SimulationState) -> SimulationState:
        """Generate action steps node"""
        if state.get("current_batch_end", 0) == 0:
            state["current_batch_start"] = 0
            state["current_batch_end"] = self.agent_configs["action_step"]["batch_size"]
        
        state["is_fresh_batch"] = True
        logger.info(f"Generating action steps ({state['current_batch_start'] + 1}-{state['current_batch_end']}/{len(state['events'])})...")
        
        state = self.action_step_agent.generate_action_steps(state)
        
        if state.get("status") == "error":
            logger.error(f"Action step generation error: {state.get('error_message', 'unknown error')}")
        return state
    
    def _rule_validate_action_steps_node(self, state: SimulationState) -> SimulationState:
        """Rule-based validation of action steps node"""
        state = self.action_step_agent.rule_validate_action_steps(state)
        if state["status"] == "action_steps_correction":
            logger.info(f"Action steps rule validation failed: {state['action_rule_validation']['correction_content']}")
        return state
    
    def _validate_action_steps_node(self, state: SimulationState) -> SimulationState:
        """Validate action steps node"""
        state = self.action_step_agent.validate_action_steps(state)
        
        if state["status"] == "action_steps_validated":
            # Count pass-once (generated content passes validation without correction)
            if state.get("is_fresh_batch") and state.get("usage_stats"):
                state["usage_stats"].action_pass_once += 1
            
            if state["current_batch_end"] >= len(state["events"]):
                state["status"] = "all_generated" 
                state["current_batch_start"] = 0
                state["current_batch_end"] = 0
                return state

            state["current_batch_start"] = state["current_batch_end"]
            state["current_batch_end"] = min(
                state["current_batch_end"] + self.agent_configs["action_step"]["batch_size"],
                len(state["events"]) 
            )
        else:
            correction = state.get("action_validation", {}).get("correction_content", "")
            logger.info(f"Action steps validation failed: {correction}")

        return state
    
    def _correct_action_steps_node(self, state: SimulationState) -> SimulationState:
        """Correct action steps node"""
        state["is_fresh_batch"] = False
        logger.info("Correcting action steps...")
        state = self.action_step_agent.correct_action_steps(state)
        if state.get("status") == "action_steps_corrected":
            logger.info("Action steps correction completed")
        return state
    
    def _generate_trajectories_node(self, state: SimulationState) -> SimulationState:
        """Generate trajectories node"""
        logger.info("Starting trajectory sequence generation...")
        
        try:
            # Call trajectory generation agent
            state = self.trajectory_agent.generate_trajectories_for_action_steps(state)
            
            # Set trajectory generation completion status
            if state.get("trajectories"):
                state["status"] = "completed"
                # self._save_results(state)
                logger.info(f"Trajectory generation completed: {len(state['trajectories'])} trajectories")
            
        except Exception as e:
            logger.error(f"Trajectory generation error: {str(e)}", exc_info=True)
            raise e
        
        return state

    def _save_results(self, state: SimulationState, error_mode: bool = False, error_message: str = None):
        """Save simulation results to JSON file
        
        Args:
            state: Simulation state
            error_mode: If True, save to error subdirectory
            error_message: Error message to include in output (only when error_mode=True)
        """
        try:
            output_filepath = state["output_filepath"]
            output_dir = os.path.dirname(output_filepath)
            filename = os.path.basename(output_filepath)
            
            # If error mode, save to error subdirectory
            if error_mode:
                output_dir = os.path.join(output_dir, "error")
            
            # Ensure output directory exists
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            final_filepath = os.path.join(output_dir, filename)
            
            # Convert Pydantic models to dicts
            def to_dict(item):
                return item.model_dump() if hasattr(item, 'model_dump') else item
            
            # Build info with statistics
            plans = state.get("activity_plans") or []
            events = state.get("events") or []
            actions = state.get("action_steps") or []
            
            info = {
                "counts": {
                    "plans": len(plans),
                    "events": len(events),
                    "actions": len(actions)
                }
            }
            
            # Add usage statistics if available
            usage_stats = state.get("usage_stats")
            if usage_stats:
                info["usage"] = usage_stats.to_dict()
            
            # Add error info if in error mode
            if error_mode:
                info["error"] = {
                    "type": "recursion_limit",
                    "message": error_message or "Recursion limit reached",
                    "status": state.get("status", "unknown")
                }
            
            # Build result data structure with info at the beginning
            result_data = {
                "info": info,
                "simulation_id": state["simulation_id"],
                "activity_plans": [to_dict(item) for item in plans],
                "events": [to_dict(item) for item in events],
                "action_steps": [to_dict(item) for item in actions],
                "trajectories": [to_dict(item) for item in (state.get("trajectories") or [])],
                "created_at": datetime.now().isoformat()
            }
            
            # Save to file
            with open(final_filepath, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)
            
            if error_mode:
                logger.info(f"Error state saved to: {final_filepath}")
                
        except Exception as e:
            logger.error(f"Failed to save results: {str(e)}", exc_info=True)
            raise


    def run_simulation(self, profile_path: str, house_layout_path: str, output_filepath: str) -> Dict[str, Any]:
        """
        Run trajectory simulation
        """

        with open(profile_path, 'r', encoding='utf-8') as f:
            resident_profile = f.read()
        with open(house_layout_path, 'r', encoding='utf-8') as f:
            house_layout_yaml = f.read()

        # Initialize state with usage stats
        simulation_id = str(uuid.uuid4())[:8]
        usage_stats = UsageStats()
        initial_state = create_initial_state(
            simulation_id=simulation_id,
            output_filepath=output_filepath,
            resident_profile=resident_profile,
            house_layout_yaml=house_layout_yaml,
            usage_stats=usage_stats
        )
        
        
        logger.info(f"Starting trajectory simulation (ID: {simulation_id})")
        logger.info(f"Output file: {output_filepath}")
        logger.info("Agent configurations:")
        for agent_name, config in self.agent_configs.items():
            logger.info(f"   {agent_name}:")
            for key, value in config.items():
                logger.info(f"      - {key}: {value}")
        logger.info(f"Limits: Recursion limit={self.recursion_limit}")
        

        filename = os.path.basename(output_filepath)
        logger.info(f"Initialized simulation result file: {filename}")
        
        config = {
            "configurable": {"thread_id": simulation_id},
            "recursion_limit": self.recursion_limit
        }
        
        try:
            # Run workflow, passing recursion limit configuration
            final_state = self.workflow.invoke(initial_state, config)
            assert final_state["status"] == "completed"
            self._save_results(final_state)
               
        except GraphRecursionError as e:
            # Handle recursion limit error: save current state to error directory
            logger.error(f"Recursion limit reached: {str(e)}")
            
            # Try to get the last state from checkpointer
            try:
                checkpoint = self.workflow.get_state(config)
                if checkpoint and checkpoint.values:
                    last_state = checkpoint.values
                    # Ensure output_filepath is set
                    if "output_filepath" not in last_state:
                        last_state["output_filepath"] = output_filepath
                    if "simulation_id" not in last_state:
                        last_state["simulation_id"] = simulation_id
                    
                    self._save_results(last_state, error_mode=True, error_message=str(e))
                    logger.info("Partial state saved to error directory")
                else:
                    # If no checkpoint available, save initial state with error info
                    initial_state["status"] = "recursion_limit_error"
                    self._save_results(initial_state, error_mode=True, error_message=str(e))
                    logger.info("Initial state with error info saved to error directory")
            except Exception as save_error:
                logger.error(f"Failed to save error state: {str(save_error)}")
            
            raise e
               
        except Exception as e:
            logger.error(f"Workflow execution error: {str(e)}", exc_info=True)
            raise e
