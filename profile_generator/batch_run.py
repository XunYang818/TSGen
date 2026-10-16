import os
import glob
import asyncio
import time
from datetime import datetime
from typing import Tuple
from profile_generator import generate_resident

from dotenv import load_dotenv
load_dotenv()

# Default occupation list
DEFAULT_OCCUPATIONS = [
    "Engineer", "Teacher", "Healthcare-Worker", "Accountant", "Lawyer", "Civil-Servant",
    "Salesperson", "Chef", "Designer", "Content-Creator", "Athlete", "Photographer", "Journalist",
    "Stay-at-home-Mom", "Stay-at-home-Dad", "Child", "Elementary-Student", "Middle-School-Student", "Elderly-Living-Alone", 
    "Retiree", "Worker", "Farmer", "Vendor", "Freelancer"
]


class ResidentGenerator:
    """Resident profile batch generator"""
    
    def __init__(self, 
                 yaml_folder: str,
                 output_folder: str,
                 model_name: str = "gpt-4o-2024-11-20",
                 temperature_creative: float = 0.8,
                 temperature_validation: float = 0.3,
                 max_concurrent: int = 8,
                 max_retries: int = 3):
        self.yaml_folder = yaml_folder
        self.output_folder = output_folder
        self.occupations = DEFAULT_OCCUPATIONS
        self.model_name = model_name
        self.temperature_creative = temperature_creative
        self.temperature_validation = temperature_validation
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        os.makedirs(self.output_folder, exist_ok=True)
        
        self.error_log_path = os.path.join(
            self.output_folder, 
            f"error_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        self.failed_tasks = []

    def check_file_exists(self, yaml_filename: str, occupation: str) -> bool:
        """Check if file already exists"""
        yaml_base = os.path.splitext(yaml_filename)[0]
        output_filename = f"{yaml_base}-{occupation}.txt"
        output_path = os.path.join(self.output_folder, output_filename)
        return os.path.exists(output_path)
    
    def run_single(self, yaml_path: str, occupation: str, task_id: int, total: int) -> Tuple[bool, str]:
        """Generate single resident profile (synchronous function)"""
        yaml_filename = os.path.basename(yaml_path)
        
        # Check if already exists
        if self.check_file_exists(yaml_filename, occupation):
            print(f"[{task_id}/{total}] Skipped: {yaml_filename} - {occupation}")
            return True, ""
        
        # Read YAML
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                layout_yaml = f.read()
        except Exception as e:
            error_msg = f"{yaml_filename} | {occupation} | Read failed: {e}"
            print(f"[{task_id}/{total}] ✗ {error_msg}")
            return False, error_msg
        
        # Retry generation
        for attempt in range(self.max_retries):
            try:
                resident_profile = generate_resident(
                    layout_yaml=layout_yaml,
                    occupation=occupation,
                    model_name=self.model_name,
                    temperature_creative=self.temperature_creative,
                    temperature_validation=self.temperature_validation
                )
                
                yaml_base = os.path.splitext(yaml_filename)[0]
                output_filename = f"{yaml_base}-{occupation}.txt"
                output_path = os.path.join(self.output_folder, output_filename)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(resident_profile)
                
                print(f"[{task_id}/{total}] ✓ {output_filename}")
                return True, ""
                
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(0.1)  # Synchronous delay
                    continue
                error_msg = f"{yaml_filename} | {occupation} | Failed: {e}"
                print(f"[{task_id}/{total}] ✗ {error_msg}")
                return False, error_msg
        
        return False, f"{yaml_filename} | {occupation} | Unknown error"
    
    def log_error(self, error_msg: str):
        """Log error"""
        with open(self.error_log_path, 'a', encoding='utf-8') as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {error_msg}\n")
    
    async def run_batch(self):
        """Execute batch generation"""
        yaml_files = glob.glob(os.path.join(self.yaml_folder, "*.yaml"))
        
        if not yaml_files:
            print(f"Error: No YAML files found in {self.yaml_folder}")
            return
        
        print(f"\n{'='*60}")
        print(f"Found {len(yaml_files)} YAML files")
        print(f"Occupations: {len(self.occupations)}")
        print(f"Total tasks: {len(yaml_files) * len(self.occupations)}")
        print(f"Max concurrent: {self.max_concurrent}")
        print(f"{'='*60}\n")
        
        # Build tasks (wrap synchronous function with asyncio.to_thread and control concurrency with semaphore)
        async def run_with_semaphore(yaml_path, occupation, task_id, total):
            """Wrapper function to control concurrency with semaphore"""
            async with self.semaphore:
                return await asyncio.to_thread(
                    self.run_single,
                    yaml_path,
                    occupation,
                    task_id,
                    total
                )
        
        tasks = []
        task_id = 0
        total = len(yaml_files) * len(self.occupations)
        
        for yaml_path in yaml_files:
            for occupation in self.occupations:
                task_id += 1
                tasks.append(run_with_semaphore(yaml_path, occupation, task_id, total))
        
        # Execute tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count results
        success_count = 0
        skip_count = 0
        fail_count = 0
        
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                fail_count += 1
                error_msg = f"Task {idx+1} exception: {result}"
                self.failed_tasks.append(error_msg)
                self.log_error(error_msg)
            else:
                success, error_msg = result
                if success:
                    if not error_msg:  # Newly generated
                        success_count += 1
                else:
                    fail_count += 1
                    self.failed_tasks.append(error_msg)
                    self.log_error(error_msg)
        
        # Output statistics
        print(f"\n{'='*60}")
        print(f"Batch generation completed")
        print(f"{'='*60}")
        print(f"Total tasks: {total}")
        print(f"Successfully generated: {success_count}")
        print(f"Failed: {fail_count}")
        
        if fail_count > 0:
            print(f"\nFailed task list:")
            for task in self.failed_tasks:
                print(f"  - {task}")
            print(f"\nError log: {self.error_log_path}")
        
        print(f"{'='*60}\n")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch generate resident profiles")
    parser.add_argument("-y", "--yaml-folder", type=str, required=True, help="YAML folder path")
    parser.add_argument("-o", "--output-folder", type=str, required=True, help="Output folder path")
    parser.add_argument("--model", type=str, default=os.getenv("TSGEN_MODEL") or "gpt-4o-2024-11-20", help="Model name")
    parser.add_argument("--temp-creative", type=float, default=0.8, help="Creative temperature")
    parser.add_argument("--temp-validation", type=float, default=0.3, help="Validation temperature")
    parser.add_argument("--max-concurrent", type=int, default=8, help="Maximum concurrent tasks")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum retry attempts")
    
    args = parser.parse_args()
    
    generator = ResidentGenerator(
        yaml_folder=args.yaml_folder,
        output_folder=args.output_folder,
        model_name=args.model,
        temperature_creative=args.temp_creative,
        temperature_validation=args.temp_validation,
        max_concurrent=args.max_concurrent,
        max_retries=args.max_retries
    )
    
    asyncio.run(generator.run_batch())


if __name__ == "__main__":
    main()
