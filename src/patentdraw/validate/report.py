"""
Validation report generation for Patent Draw.

Creates JSON and optional PDF reports of validation results.
"""

import json
import os

from patentdraw.io.save_artifacts import ensure_dir, save_json
from patentdraw.tracer import get_tracer, trace


@trace(label="generate_report")
def generate_report(document, out_dir, debug_writer=None):
    """
    Generate validation report files.
    
    Creates:
    - validation_report.json: Full check results
    - validation_summary.txt: Human-readable summary
    """
    tracer = get_tracer()
    
    report = document.validation
    
    # Save JSON report
    report_path = os.path.join(out_dir, "validation_report.json")
    save_json(report.model_dump(), report_path)
    
    # Generate summary
    summary_lines = ["Patent Draw Validation Report", "=" * 40, ""]
    
    passed = [c for c in report.checks if c.passed]
    failed = [c for c in report.checks if not c.passed]
    
    summary_lines.append(f"Total checks: {len(report.checks)}")
    summary_lines.append(f"Passed: {len(passed)}")
    summary_lines.append(f"Failed: {len(failed)}")
    summary_lines.append("")
    
    if failed:
        summary_lines.append("ISSUES:")
        summary_lines.append("-" * 40)
        for check in failed:
            severity_mark = "[ERROR]" if check.severity.value == "error" else "[WARN]"
            summary_lines.append(f"{severity_mark} {check.rule_id}: {check.message}")
        summary_lines.append("")
    
    summary_lines.append("ALL CHECKS:")
    summary_lines.append("-" * 40)
    for check in report.checks:
        status = "PASS" if check.passed else "FAIL"
        summary_lines.append(f"[{status}] {check.rule_id}: {check.message}")
    
    summary_text = "\n".join(summary_lines)
    
    # Save summary
    summary_path = os.path.join(out_dir, "validation_summary.txt")
    ensure_dir(os.path.dirname(summary_path))
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    
    tracer.event(f"Report saved: {len(report.checks)} checks, {report.error_count} errors")
    
    # Debug artifacts
    if debug_writer:
        metrics = {
            "total_checks": len(report.checks),
            "passed": len(passed),
            "failed": len(failed),
            "errors": report.error_count,
            "warnings": report.warning_count,
        }
        debug_writer.save_json(metrics, "stage8", "stage8_metrics.json")
        debug_writer.save_json(report.model_dump(), "stage8", "validation_report.json")
    
    return report_path, summary_path


def format_check_result(check):
    """Format a single check result for display."""
    status = "PASS" if check.passed else "FAIL"
    severity = check.severity.value.upper()
    return f"[{status}][{severity}] {check.rule_id}: {check.message}"
