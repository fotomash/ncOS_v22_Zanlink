# core/mission_report.py

import os
import datetime

OBJECTIVES_DIR = 'journal/daily_objectives/'
COMPLETION_DIR = 'journal/daily_objectives/completed/'
REPORTS_DIR = 'journal/daily_objectives/reports/'

# Ensure reports folder exists
os.makedirs(REPORTS_DIR, exist_ok=True)

def get_today_filename():
    today_str = datetime.datetime.now().strftime('%Y-%m-%d')
    return today_str

def load_objectives():
    file_path = os.path.join(OBJECTIVES_DIR, f"{get_today_filename()}_objectives.txt")

    if not os.path.exists(file_path):
        return []

    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    return lines

def load_completions():
    file_path = os.path.join(COMPLETION_DIR, f"{get_today_filename()}_completed.txt")
    if not os.path.exists(file_path):
        return []
    with open(file_path, 'r') as f:
        return [int(line.strip()) for line in f.readlines() if line.strip().isdigit()]

def generate_mission_report():
    objectives = load_objectives()
    completions = load_completions()

    today_str = get_today_filename()
    report_path = os.path.join(REPORTS_DIR, f"mission_report_{today_str}.md")

    if not objectives:
        print("⚠️ No objectives to generate report for.")
        return

    completed_count = len([idx for idx in completions if idx <= len(objectives)])
    total_count = len(objectives)

    with open(report_path, 'w') as f:
        f.write(f"# Captain Zanzibar Mission Report - {today_str}\n\n")
        f.write(f"**Objectives Completed:** {completed_count}/{total_count}\n\n")
        f.write("## Checklist:\n")
        for idx, obj in enumerate(objectives, 1):
            mark = "✅" if idx in completions else "❌"
            f.write(f"- {mark} Objective {idx}: {obj}\n")

    print(f"✅ Mission report generated: {report_path}")

if __name__ == "__main__":
    generate_mission_report()
