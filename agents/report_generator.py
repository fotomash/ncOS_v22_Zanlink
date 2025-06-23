"""
ReportGenerator - NCOS v21 Agent
Fixed version with proper config fields
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional


@dataclass
class ReportGeneratorConfig:
    """Configuration for ReportGenerator"""
    agent_id: str = "report_generator"
    enabled: bool = True
    log_level: str = "INFO"
    max_retries: int = 3
    timeout: float = 30.0
    output_dir: Optional[str] = "reports"
    report_interval: Optional[int] = 3600
    max_reports: Optional[int] = 100
    formats: Optional[List[str]] = field(default_factory=lambda: ["json", "csv"])
    custom_params: Dict[str, Any] = field(default_factory=dict)


class ReportGenerator:
    """
    ReportGenerator - Report generation and data visualization
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = ReportGeneratorConfig(**(config or {}))
        self.logger = logging.getLogger(f"NCOS.{self.config.agent_id}")
        self.logger.setLevel(getattr(logging, self.config.log_level))

        self.status = "initialized"
        self.metrics = {
            "messages_processed": 0,
            "errors": 0,
            "uptime_start": datetime.now(),
            "reports_generated": 0,
            "scheduled_reports": 0
        }

        self.templates = {}
        self.output_formats = self.config.formats
        self.scheduled_reports = {}
        self.generated_reports = []

        self.logger.info(f"{self.config.agent_id} initialized")

    async def initialize(self):
        """Initialize the agent"""
        try:
            self.logger.info(f"Initializing {self.config.agent_id}")
            await self._setup()
            self.status = "ready"
            self.logger.info(f"{self.config.agent_id} ready for operation")
        except Exception as e:
            self.logger.error(f"Failed to initialize {self.config.agent_id}: {e}")
            self.status = "error"
            raise

    async def _setup(self):
        """Agent-specific setup logic"""
        # Initialize reporting engine
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Output directory: {output_dir}")
        self.logger.info(f"Supported formats: {self.output_formats}")

        # Load default templates
        self.templates = {
            "system_status": {
                "title": "System Status Report",
                "sections": ["agents", "metrics", "alerts"]
            },
            "performance": {
                "title": "Performance Report",
                "sections": ["portfolio", "trades", "pnl"]
            },
            "risk": {
                "title": "Risk Report",
                "sections": ["positions", "limits", "violations"]
            }
        }

        # Start scheduled reporting if configured
        if self.config.report_interval:
            asyncio.create_task(self._scheduled_reporting_loop())

    async def _scheduled_reporting_loop(self):
        """Background scheduled reporting loop"""
        while self.status == "ready":
            try:
                await self._process_scheduled_reports()
                await asyncio.sleep(self.config.report_interval)
            except Exception as e:
                self.logger.error(f"Scheduled reporting loop error: {e}")
                break

    async def _process_scheduled_reports(self):
        """Process all scheduled reports"""
        current_time = datetime.now()

        for report_id, report_config in self.scheduled_reports.items():
            try:
                next_run = datetime.fromisoformat(report_config.get("next_run", current_time.isoformat()))

                if current_time >= next_run:
                    await self._generate_scheduled_report(report_id, report_config)

                    # Update next run time
                    interval = report_config.get("interval", 3600)
                    report_config["next_run"] = (current_time + timedelta(seconds=interval)).isoformat()

            except Exception as e:
                self.logger.error(f"Error processing scheduled report {report_id}: {e}")

    async def _generate_scheduled_report(self, report_id: str, report_config: Dict[str, Any]):
        """Generate a scheduled report"""
        template = report_config.get("template", "system_status")
        data = report_config.get("data", {})

        result = await self._generate_report(template, data)

        if result.get("status") == "success":
            self.logger.info(f"Generated scheduled report: {report_id}")
        else:
            self.logger.error(f"Failed to generate scheduled report {report_id}: {result.get('error')}")

    async def _generate_report(self, template: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a report using a template"""
        if template not in self.templates:
            return {"error": f"Unknown template: {template}"}

        template_config = self.templates[template]

        # Generate report content
        report_content = {
            "title": template_config["title"],
            "generated_at": datetime.now().isoformat(),
            "template": template,
            "sections": {}
        }

        # Process each section
        for section in template_config["sections"]:
            try:
                section_data = await self._generate_section(section, data)
                report_content["sections"][section] = section_data
            except Exception as e:
                self.logger.error(f"Error generating section {section}: {e}")
                report_content["sections"][section] = {"error": str(e)}

        # Save report
        report_filename = f"{template}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        saved_files = []
        for format_type in self.output_formats:
            try:
                file_path = await self._save_report(report_content, report_filename, format_type)
                saved_files.append(str(file_path))
            except Exception as e:
                self.logger.error(f"Error saving report in {format_type} format: {e}")

        # Track generated report
        report_record = {
            "id": len(self.generated_reports) + 1,
            "template": template,
            "filename": report_filename,
            "files": saved_files,
            "generated_at": report_content["generated_at"]
        }

        self.generated_reports.append(report_record)
        self.metrics["reports_generated"] += 1

        # Clean up old reports if limit exceeded
        if len(self.generated_reports) > self.config.max_reports:
            old_report = self.generated_reports.pop(0)
            await self._cleanup_report_files(old_report)

        return {
            "status": "success",
            "report_id": report_record["id"],
            "files": saved_files,
            "content": report_content
        }

    async def _generate_section(self, section: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a specific report section"""
        if section == "agents":
            return {
                "active_agents": data.get("active_agents", []),
                "agent_count": len(data.get("active_agents", [])),
                "status": "operational"
            }
        elif section == "metrics":
            return {
                "system_uptime": data.get("uptime", 0),
                "messages_processed": data.get("messages_processed", 0),
                "errors": data.get("errors", 0)
            }
        elif section == "alerts":
            return {
                "active_alerts": data.get("alerts", []),
                "alert_count": len(data.get("alerts", []))
            }
        elif section == "portfolio":
            return {
                "total_value": data.get("portfolio_value", 0),
                "positions": data.get("positions", {}),
                "performance": data.get("performance", {})
            }
        elif section == "trades":
            return {
                "trades_today": data.get("trades_today", 0),
                "volume": data.get("volume", 0),
                "pnl": data.get("pnl", 0)
            }
        elif section == "positions":
            return {
                "open_positions": data.get("open_positions", []),
                "position_count": len(data.get("open_positions", []))
            }
        elif section == "limits":
            return {
                "risk_limits": data.get("risk_limits", {}),
                "utilization": data.get("limit_utilization", {})
            }
        elif section == "violations":
            return {
                "violations": data.get("violations", []),
                "violation_count": len(data.get("violations", []))
            }
        else:
            return {"error": f"Unknown section: {section}"}

    async def _save_report(self, content: Dict[str, Any], filename: str, format_type: str) -> Path:
        """Save report in specified format"""
        output_dir = Path(self.config.output_dir)

        if format_type == "json":
            file_path = output_dir / f"{filename}.json"
            with open(file_path, 'w') as f:
                json.dump(content, f, indent=2)
        elif format_type == "csv":
            file_path = output_dir / f"{filename}.csv"
            # Simplified CSV export
            with open(file_path, 'w') as f:
                f.write("Section,Key,Value\n")
                for section_name, section_data in content.get("sections", {}).items():
                    if isinstance(section_data, dict):
                        for key, value in section_data.items():
                            f.write(f"{section_name},{key},{value}\n")
        elif format_type == "html":
            file_path = output_dir / f"{filename}.html"
            html_content = self._generate_html_report(content)
            with open(file_path, 'w') as f:
                f.write(html_content)
        else:
            raise ValueError(f"Unsupported format: {format_type}")

        return file_path

    def _generate_html_report(self, content: Dict[str, Any]) -> str:
        """Generate HTML report content"""
        html = f"""
        <html>
        <head><title>{content['title']}</title></head>
        <body>
        <h1>{content['title']}</h1>
        <p>Generated: {content['generated_at']}</p>
        """

        for section_name, section_data in content.get("sections", {}).items():
            html += f"<h2>{section_name.title()}</h2>"
            if isinstance(section_data, dict):
                html += "<ul>"
                for key, value in section_data.items():
                    html += f"<li><strong>{key}:</strong> {value}</li>"
                html += "</ul>"

        html += "</body></html>"
        return html

    async def _cleanup_report_files(self, report_record: Dict[str, Any]):
        """Clean up old report files"""
        for file_path in report_record.get("files", []):
            try:
                Path(file_path).unlink(missing_ok=True)
                self.logger.debug(f"Cleaned up report file: {file_path}")
            except Exception as e:
                self.logger.error(f"Error cleaning up file {file_path}: {e}")

    async def _schedule_report(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Schedule a report for automatic generation"""
        report_id = f"scheduled_{len(self.scheduled_reports) + 1}"

        schedule_config = {
            "template": config.get("template", "system_status"),
            "interval": config.get("interval", 3600),
            "data": config.get("data", {}),
            "next_run": datetime.now().isoformat()
        }

        self.scheduled_reports[report_id] = schedule_config
        self.metrics["scheduled_reports"] += 1

        return {
            "status": "scheduled",
            "report_id": report_id,
            "config": schedule_config
        }

    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming message"""
        try:
            self.metrics["messages_processed"] += 1
            self.logger.debug(f"Processing message: {message.get('type', 'unknown')}")

            result = await self._handle_message(message)

            return {
                "status": "success",
                "agent_id": self.config.agent_id,
                "result": result,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            self.metrics["errors"] += 1
            self.logger.error(f"Error processing message: {e}")
            return {
                "status": "error",
                "agent_id": self.config.agent_id,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def _handle_message(self, message: Dict[str, Any]) -> Any:
        """Agent-specific message handling"""
        # Handle report requests
        msg_type = message.get("type")
        if msg_type == "generate_report":
            return await self._generate_report(message.get("template"), message.get("data"))
        elif msg_type == "schedule_report":
            return await self._schedule_report(message.get("config"))
        elif msg_type == "get_templates":
            return {"templates": list(self.templates.keys())}
        elif msg_type == "get_reports":
            return {"reports": self.generated_reports}
        elif msg_type == "get_scheduled":
            return {"scheduled_reports": self.scheduled_reports}

        return {"processed": True, "agent": self.config.agent_id}

    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        uptime = datetime.now() - self.metrics["uptime_start"]
        return {
            "agent_id": self.config.agent_id,
            "status": self.status,
            "uptime_seconds": uptime.total_seconds(),
            "metrics": self.metrics.copy(),
            "available_templates": list(self.templates.keys()),
            "scheduled_reports_count": len(self.scheduled_reports),
            "generated_reports_count": len(self.generated_reports)
        }

    async def shutdown(self):
        """Shutdown the agent"""
        self.logger.info(f"Shutting down {self.config.agent_id}")
        self.status = "shutdown"


# Agent factory function
def create_agent(config: Dict[str, Any] = None) -> ReportGenerator:
    """Factory function to create ReportGenerator instance"""
    return ReportGenerator(config)


# Export the agent class
__all__ = ["ReportGenerator", "ReportGeneratorConfig", "create_agent"]
