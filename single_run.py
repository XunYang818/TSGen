import os
import argparse
from utils.logger import setup_logging, set_task_context
from agents.prompts import set_version as set_prompt_version
from dotenv import load_dotenv

load_dotenv()

# Prompt version: 'full' (with constraints) or 'core' (without constraints)
PROMPT_VERSION = "full"
MODEL_NAME = os.getenv("TSGEN_MODEL")


def configured_model(default: str) -> str:
    """Use TSGEN_MODEL when set, otherwise preserve each original default."""
    return MODEL_NAME or default


# Agent configurations (default)
DEFAULT_AGENT_CONFIGS = {
    "planning": {
        "planning_model": configured_model("o3-2025-04-16"),
        "validation_model": configured_model("gpt-4o-2024-11-20"),
        "correction_model": configured_model("gpt-4o-2024-11-20"),
        "planning_t": 0.5,        
        "validation_t": 0.3,      
        "correction_t": 0.7
    },
    "event": {
        "generation_model": configured_model("gpt-4o-2024-11-20"),
        "validation_model": configured_model("gpt-4o-2024-11-20"),
        "correction_model": configured_model("gpt-4o-2024-11-20"),
        "generation_t": 0.5,      
        "validation_t": 0.3,      
        "correction_t": 0.7,
        "batch_size": 1,          
        "context_size": 2
    },
    "action_step": {
        "generation_model": configured_model("gpt-4o-2024-11-20"),
        "validation_model": configured_model("gpt-4o-2024-11-20"),
        "correction_model": configured_model("gpt-4o-2024-11-20"),
        "generation_t": 0.5,      
        "validation_t": 0.3,     
        "correction_t": 0.7,
        "batch_size": 1,
        "context_size": 2
    },
    "trajectory": {
        "speed_range": (0.6, 0.8),
        "dt": 0.5,
        "stationary_dt": 0.5,
        "unit": "dm"
    }
}


def run_single(profile_path: str, 
               house_layout_path: str, 
               output_dir: str, 
               task_id: str = None, 
               log_level: str = "INFO",
               agent_configs: dict = None,
               save_workflow_graph: bool = True,
               to_console: bool = True,
               prompt_version: str = None):
    """
    Run single trajectory simulation
    
    Args:
        profile_path: Resident profile file path
        house_layout_path: House layout file path
        output_dir: Output directory
        task_id: Task ID (for log differentiation)
        log_level: Log level
        agent_configs: Agent configuration (optional)
        save_workflow_graph: Whether to save workflow graph
        to_console: Whether to output logs to console
        prompt_version: Prompt version ('full' or 'core')
    
    Returns:
        output_filepath: Output file path
    """
    from workflow import TrajectorySimulationWorkflow
    
    # Set prompt version before importing agents
    set_prompt_version(prompt_version or PROMPT_VERSION)
    
    # Use profile filename as task_id if not provided
    profile_filename = os.path.splitext(os.path.basename(profile_path))[0]
    if task_id is None:
        task_id = profile_filename
    
    # Setup logging system (safe to call multiple times)
    log_dir = os.path.join(output_dir, "logs")
    setup_logging(log_dir, level=log_level, console=to_console)
    
    # Set task context - all logs will route to {task_id}.log
    set_task_context(task_id)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    output_filepath = os.path.join(output_dir, f"{profile_filename}.json")
    
    # Use provided configs or default
    configs = agent_configs if agent_configs is not None else DEFAULT_AGENT_CONFIGS

    # Initialize workflow
    workflow = TrajectorySimulationWorkflow(
        recursion_limit=1000,
        max_retry_count=10,
        agent_configs=configs
    )
    
    # Generate workflow graph (only once)
    if save_workflow_graph:
        try:
            graph_png = workflow.workflow.get_graph().draw_mermaid_png()
            graph_path = os.path.join(output_dir, "workflow_graph.png")
            if not os.path.exists(graph_path):
                with open(graph_path, "wb") as f:
                    f.write(graph_png)
        except Exception:
            pass

    # Run simulation
    workflow.run_simulation(
        profile_path=profile_path,
        house_layout_path=house_layout_path,
        output_filepath=output_filepath
    )
    
    return output_filepath


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Trajectory Simulation Program')
    
    project_root = os.getenv("PROJECT_ROOT", ".")
    default_profile = os.path.join(project_root, "1-data_simulation/profile_generator/data/1-Accountant.txt")
    default_layout = os.path.join(project_root, "1-data_simulation/yaml_layout/data/1.yaml")
    default_output = os.path.join(project_root, "1-data_simulation/output/gpt5")
    
    parser.add_argument('--profile', '-p', default=default_profile)
    parser.add_argument('--layout', '-l', default=default_layout)
    parser.add_argument('--output', '-o', default=default_output)
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    parser.add_argument('--task-id', default=None)
    parser.add_argument('--prompt-version', '-pv', default='full', choices=['full', 'core'],
                        help='Prompt version: full (with constraints) or core (without)')
    
    return parser.parse_args()


def main():
    """Main entry point"""
    try:
        args = parse_args()
        run_single(
            profile_path=args.profile, 
            house_layout_path=args.layout, 
            output_dir=args.output, 
            task_id=args.task_id, 
            log_level=args.log_level,
            to_console=True,
            prompt_version=args.prompt_version
        )
    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as e:
        raise e


if __name__ == "__main__":
    main()
