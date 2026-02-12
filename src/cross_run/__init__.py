"""Cross-Run Convergence Detection — finds buried gold across S3-failed runs."""

from src.cross_run.loader import RunData, load_run, find_runs
from src.cross_run.matcher import CrossMatch, CrossRunResult, cross_match
from src.cross_run.report import save_report
