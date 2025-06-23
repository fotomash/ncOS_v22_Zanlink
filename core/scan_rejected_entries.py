import os
import argparse
from datetime import datetime

def scan_rejected_entries(journal_dir, min_score=0.0):
    if not os.path.isdir(journal_dir):
        print(f"[ERROR] Directory not found: {journal_dir}")
        return

    markdown_files = [f for f in os.listdir(journal_dir) if f.startswith("rejected_entry_") and f.endswith(".md")]
    markdown_files.sort(reverse=True)
    markdown_lines = [f"# Rejected Entry Summary (Filtered score > {min_score:.2f})", ""]

    for md_file in markdown_files:
        path = os.path.join(journal_dir, md_file)
        with open(path, 'r') as f:
            content = f.read()

        score_line = next((line for line in content.splitlines() if "POI Score" in line), "")
        phase_line = next((line for line in content.splitlines() if "Phase" in line), "")
        reason_line = next((line for line in content.splitlines() if "Reason" in line), "")
        symbol_line = next((line for line in content.splitlines() if "Symbol" in line), "")
        time_line = next((line for line in content.splitlines() if "Time" in line), "")

        try:
            score_value = float(score_line.split(":")[1].strip())
        except Exception:
            score_value = 0.0

        if score_value >= min_score:
            print(f"📍 {symbol_line} @ {time_line}")
            print(f"   → Score: {score_value:.2f} | {phase_line} | {reason_line}")
            print("")
            markdown_lines.append(f"## ❌ {symbol_line} @ {time_line}")
            markdown_lines.append(f"- **POI Score**: {score_value:.2f}")
            markdown_lines.append(f"- {phase_line}")
            markdown_lines.append(f"- {reason_line}")
            markdown_lines.append("")

    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(journal_dir, f"rejected_summary_scan_{timestamp}.md")
    with open(out_path, "w") as f:
        f.write("\n".join(markdown_lines))

    print(f"\n📄 Markdown summary saved to {out_path}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Scan Markdown log of rejected entries.')
    parser.add_argument('--journal-dir', type=str, default='journal', help='Directory containing markdown log files.')
    parser.add_argument('--min-score', type=float, default=0.0, help='Filter to only show entries above this POI score.')
    args = parser.parse_args()

    scan_rejected_entries(args.journal_dir, min_score=args.min_score)
