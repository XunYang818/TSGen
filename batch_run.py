import os
import glob
import asyncio
import argparse
import logging
from datetime import datetime
from typing import Tuple, Optional
from single_run import run_single, DEFAULT_AGENT_CONFIGS
from utils.logger import setup_logging


class TrajectoryBatchRunner:
    """Trajectory simulation batch runner"""
    
    def __init__(self,
                 profile_folder: str,
                 layout_folder: str,
                 output_folder: str,
                 max_concurrent: int = 4,
                 max_retries: int = 3,
                 log_level: str = "INFO",
                 agent_configs: dict = None,
                 prompt_version: str = "full"):
        """
        Initialize batch runner
        """
        self.profile_folder = profile_folder
        self.layout_folder = layout_folder
        self.output_folder = output_folder
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.log_level = log_level
        self.agent_configs = agent_configs if agent_configs else DEFAULT_AGENT_CONFIGS
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.prompt_version = prompt_version
        
        os.makedirs(self.output_folder, exist_ok=True)
        
        # Setup logging system (console + auto file routing)
        setup_logging(
            log_dir=os.path.join(self.output_folder, "logs"),
            level=log_level,
            console=True
        )
        
        # Use standard logger - will auto-route to batch log file
        self.logger = logging.getLogger("batch_runner")
        
        self.failed_tasks = []
    
    def run_single_task(self, 
                        profile_path: str, 
                        layout_path: str, 
                        task_id: int, 
                        total: int) -> Tuple[bool, str]:
        """
        Run single trajectory simulation task (synchronous function)
        
        Args:
            profile_path: Profile file path
            layout_path: Layout file path
            task_id: Task ID
            total: Total number of tasks
        
        Returns:
            (success, error_message)
        """
        profile_filename = os.path.basename(profile_path)
        
        # Check if output already exists
        output_path = os.path.join(self.output_folder, os.path.splitext(profile_filename)[0] + ".json")
        if os.path.exists(output_path):
            self.logger.info(f"[{task_id}/{total}] Skipped (exists): {profile_filename}")
            return True, "skipped"
        
        # Check if layout file exists
        if not os.path.exists(layout_path):
            error_msg = f"{profile_filename} | Layout not found: {layout_path}"
            self.logger.warning(f"[{task_id}/{total}] {error_msg}")
            return False, error_msg
        
        # Retry logic
        for attempt in range(self.max_retries):
            try:
                log_task_id = os.path.splitext(profile_filename)[0]
                
                run_single(
                    profile_path=profile_path,
                    house_layout_path=layout_path,
                    output_dir=self.output_folder,
                    task_id=log_task_id,
                    log_level=self.log_level,
                    agent_configs=self.agent_configs,
                    save_workflow_graph=False, 
                    to_console=False,
                    prompt_version=self.prompt_version
                )
                
                self.logger.info(f"[{task_id}/{total}] ✓ {profile_filename}")
                return True, ""
                
            except Exception as e:
                if attempt < self.max_retries - 1:
                    self.logger.warning(f"[{task_id}/{total}] Retry ({attempt + 1}/{self.max_retries}): {profile_filename}")
                    continue
                error_msg = f"{profile_filename} | Failed: {str(e)}"
                self.logger.error(f"[{task_id}/{total}] ✗ {error_msg}")
                return False, error_msg
        
        return False, f"{profile_filename} | Unknown error"
    
    async def run_batch(self):
        """Execute batch generation"""
        # Get all profile files
        profile_files = glob.glob(os.path.join(self.profile_folder, "*.txt"))
        
        if not profile_files:
            self.logger.error(f"No profile files found in {self.profile_folder}")
            return
        
        # Build task list
        tasks_info = []
        skipped_no_layout = []
        
        for profile_path in profile_files:
            profile_filename = os.path.basename(profile_path)
            layout_name = profile_filename.split('-')[0] + ".yaml"
            layout_path = os.path.join(self.layout_folder, layout_name)
            
            if os.path.exists(layout_path):
                tasks_info.append((profile_path, layout_path))
            else:
                skipped_no_layout.append(profile_filename)
        
        # Output statistics
        self.logger.info("=" * 60)
        self.logger.info("Batch Trajectory Simulation")
        self.logger.info("=" * 60)
        self.logger.info(f"Profile folder: {self.profile_folder}")
        self.logger.info(f"Layout folder: {self.layout_folder}")
        self.logger.info(f"Output folder: {self.output_folder}")
        self.logger.info("-" * 60)
        self.logger.info(f"Profile files found: {len(profile_files)}")
        self.logger.info(f"Valid tasks: {len(tasks_info)}")
        self.logger.info(f"Skipped (no layout): {len(skipped_no_layout)}")
        self.logger.info(f"Max concurrent: {self.max_concurrent}")
        self.logger.info(f"Max retries: {self.max_retries}")
        self.logger.info("=" * 60)
        
        if skipped_no_layout:
            self.logger.warning("Files skipped due to missing layout:")
            for f in skipped_no_layout[:10]:
                self.logger.warning(f"  - {f}")
            if len(skipped_no_layout) > 10:
                self.logger.warning(f"  ... and {len(skipped_no_layout) - 10} more files")
        
        if not tasks_info:
            self.logger.warning("No valid tasks to execute")
            return
        
        # Wrapper function for async execution with semaphore
        async def run_with_semaphore(profile_path, layout_path, task_id, total):
            """Wrapper function to control concurrency with semaphore"""
            async with self.semaphore:
                return await asyncio.to_thread(
                    self.run_single_task,
                    profile_path,
                    layout_path,
                    task_id,
                    total
                )
        
        # Build async tasks
        tasks = []
        total = len(tasks_info)
        
        for idx, (profile_path, layout_path) in enumerate(tasks_info, 1):
            tasks.append(run_with_semaphore(profile_path, layout_path, idx, total))
        
        # Execute all tasks
        start_time = datetime.now()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = datetime.now()
        
        # Count results
        success_count = 0
        skip_count = 0
        fail_count = 0
        
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                fail_count += 1
                error_msg = f"Task {idx+1} exception: {result}"
                self.failed_tasks.append(error_msg)
                self.logger.error(error_msg)
            else:
                success, error_msg = result
                if success:
                    if error_msg == "skipped":
                        skip_count += 1
                    else:
                        success_count += 1
                else:
                    fail_count += 1
                    self.failed_tasks.append(error_msg)
        
        # Output summary
        duration = (end_time - start_time).total_seconds()
        self.logger.info("=" * 60)
        self.logger.info("Batch Generation Completed")
        self.logger.info("=" * 60)
        self.logger.info(f"Total tasks: {total}")
        self.logger.info(f"Successfully generated: {success_count}")
        self.logger.info(f"Skipped (already exists): {skip_count}")
        self.logger.info(f"Failed: {fail_count}")
        self.logger.info(f"Total duration: {duration:.2f} seconds")
        
        if fail_count > 0:
            self.logger.error("Failed task list:")
            for task in self.failed_tasks:
                self.logger.error(f"  - {task}")
        
        self.logger.info("=" * 60)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Batch run trajectory simulation")
    
    # Default paths
    project_root = os.getenv("PROJECT_ROOT", ".")
    default_profile_folder = os.path.join(project_root, "1-data_simulation/profile_generator/data")
    default_layout_folder = os.path.join(project_root, "1-data_simulation/yaml_layout/data")
    default_output_folder = os.path.join(project_root, "1-data_simulation/data/4o-full-05t")
    
    parser.add_argument("-p", "--profile-folder", type=str, 
                       default=default_profile_folder,
                       help=f"Profile folder path (default: {default_profile_folder})")
    
    parser.add_argument("-l", "--layout-folder", type=str,
                       default=default_layout_folder,
                       help=f"Layout folder path (default: {default_layout_folder})")
    
    parser.add_argument("-o", "--output-folder", type=str,
                       default=default_output_folder,
                       help=f"Output folder path (default: {default_output_folder})")
    
    parser.add_argument("--max-concurrent", type=int, default=50,
                       help="Maximum concurrent tasks (default: 16)")
    
    parser.add_argument("--max-retries", type=int, default=3,
                       help="Maximum retry attempts (default: 3)")
    
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Log level (default: INFO)")
    
    parser.add_argument('--prompt-version', '-pv', default='full', choices=['full', 'core'],
                        help='Prompt version: full (with constraints) or core (without)')
    
    args = parser.parse_args()
    
    runner = TrajectoryBatchRunner(
        profile_folder=args.profile_folder,
        layout_folder=args.layout_folder,
        output_folder=args.output_folder,
        max_concurrent=args.max_concurrent,
        max_retries=args.max_retries,
        log_level=args.log_level,
        prompt_version=args.prompt_version
    )
    
    asyncio.run(runner.run_batch())


if __name__ == "__main__":
    main()
