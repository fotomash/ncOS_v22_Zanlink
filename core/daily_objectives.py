# core/objectives_tracker.py

import os
import datetime

OBJECTIVES_DIR = 'journal/daily_objectives/'
COMPLETION_DIR = 'journal/daily_objectives/completed/'

# Ensure completion folder exists
os.makedirs(COMPLETION_DIR, exist_ok=True)

def get_today_filename():
    today_str = datetime.datetime.now().strftime('%Y-%m-%d')
    return today_str

def load_objectives():
    file_path = os.path.join(OBJECTIVES_DIR, f"{get_today_filename()}_objectives.txt")

    if not os.path.exists(file_path):
        print("⚠️ No objectives file found for today.")
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

def save_completions(completed_indices):
    file_path = os.path.join(COMPLETION_DIR, f"{get_today_filename()}_completed.txt")
    with open(file_path, 'w') as f:
        for idx in completed_indices:
            f.write(f"{idx}\n")

def display_objectives_checklist():
    objectives = load_objectives()
    completed = load_completions()

    if not objectives:
        print("⚠️ No objectives to track today.")
        return

    print("\n🧭 Captain's Daily Objectives Checklist:")
    for idx, objective in enumerate(objectives, 1):
        mark = "[x]" if idx in completed else "[ ]"
        print(f"{mark} Objective {idx}: {objective}")

def mark_objectives():
    objectives = load_objectives()
    if not objectives:
        return

    display_objectives_checklist()

    print("\n🧭 Captain, which objectives have been completed? (Enter numbers separated by commas)")
    input_str = input("Completed Objectives: ")

    try:
        indices = [int(x.strip()) for x in input_str.split(',') if x.strip().isdigit()]
        save_completions(indices)
        print("✅ Objectives updated.")
    except Exception as e:
        print(f"⚠️ Error processing input: {e}")

if __name__ == "__main__":
    print("1. View Objectives\n2. Mark Objectives Completed")
    choice = input("Select an option (1/2): ")
    if choice == '2':
        mark_objectives()
    else:
        display_objectives_checklist()
