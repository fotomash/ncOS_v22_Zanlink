#!/usr/bin/env python3
"""
ncOS v22.0 - Zanlink Enhanced Edition
Main launcher with integrated LLM support
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
import subprocess
import signal
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NCOSLauncher:
    """Main launcher for ncOS with Zanlink integration"""

    def __init__(self):
        self.processes = {}
        self.running = False
        self.base_dir = Path(__file__).parent

    def start_service(self, name: str, command: List[str], env: Dict = None):
        """Start a service subprocess"""
        try:
            env_vars = os.environ.copy()
            if env:
                env_vars.update(env)

            process = subprocess.Popen(
                command,
                env=env_vars,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            self.processes[name] = process
            logger.info(f"Started {name} (PID: {process.pid})")
        except Exception as e:
            logger.error(f"Failed to start {name}: {e}")

    def start_all_services(self):
        """Start all ncOS services"""
        logger.info("Starting ncOS v22.0 - Zanlink Enhanced Edition")

        # Set Python path
        python_path = f"{self.base_dir}:{self.base_dir}/agents:{os.environ.get('PYTHONPATH', '')}"

        services = [
            # Core services
            ("orchestrator", ["python", "-m", "agents.master_orchestrator"]),
            ("market_data", ["python", "-m", "unified_mt4_processor"]),
            ("pattern_engine", ["python", "-m", "engine"]),

            # API services
            ("journal_api", ["python", "-m", "app"]),
            ("zbar_api", ["python", "-m", "zbar_routes"]),

            # LLM services
            ("llm_assistant", ["python", "-m", "llm_assistant"]),
            ("zanlink_bridge", ["python", "-m", "ncos_zanlink_bridge"]),
        ]

        # Start each service
        for name, command in services:
            self.start_service(name, command, {"PYTHONPATH": python_path})
            time.sleep(2)  # Give each service time to start

        self.running = True
        logger.info("All services started successfully")

        # Print status
        self.print_status()

    def print_status(self):
        """Print service status"""
        print("\n" + "="*60)
        print("ncOS v22.0 - Service Status")
        print("="*60)

        for name, process in self.processes.items():
            if process.poll() is None:
                print(f"✅ {name:<20} PID: {process.pid:<10} Status: Running")
            else:
                print(f"❌ {name:<20} PID: {process.pid:<10} Status: Stopped")

        print("\n🌐 Zanlink Endpoints:")
        print("   - Quick Status: https://zanlink.com/api/v1/quick/status")
        print("   - Analysis: https://zanlink.com/api/v1/analyze")
        print("   - Patterns: https://zanlink.com/api/v1/patterns/detect")
        print("\n💻 Local Endpoints:")
        print("   - Journal: http://localhost:8000")
        print("   - Dashboard: http://localhost:8501")
        print("   - LLM Bridge: http://localhost:8004")
        print("="*60 + "\n")

    def stop_all_services(self):
        """Stop all services"""
        logger.info("Stopping all services...")

        for name, process in self.processes.items():
            if process.poll() is None:
                process.terminate()
                logger.info(f"Stopped {name}")

        # Wait for processes to terminate
        time.sleep(2)

        # Force kill if needed
        for name, process in self.processes.items():
            if process.poll() is None:
                process.kill()
                logger.warning(f"Force killed {name}")

        self.running = False
        logger.info("All services stopped")

    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info("Received shutdown signal")
        self.stop_all_services()
        sys.exit(0)

    def run(self):
        """Main run loop"""
        # Set up signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        try:
            # Start services
            self.start_all_services()

            # Monitor loop
            while self.running:
                time.sleep(10)

                # Check service health
                for name, process in self.processes.items():
                    if process.poll() is not None:
                        logger.warning(f"{name} has stopped, restarting...")
                        # Restart logic here if needed

        except Exception as e:
            logger.error(f"Error in main loop: {e}")
        finally:
            self.stop_all_services()

def main():
    """Main entry point"""
    launcher = NCOSLauncher()
    launcher.run()

if __name__ == "__main__":
    main()
