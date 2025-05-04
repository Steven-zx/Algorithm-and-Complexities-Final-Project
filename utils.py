from typing import List, Dict, Any, Tuple, Optional
# utils.py
import numpy as np

def handle_user_weights(weights_input: Dict[str, float]) -> Dict[str, float]:
    """Normalizes user-provided importance levels to weights between 0 and 1."""
    return weights_input

# Constants for validation
REQUIRED_FIELDS = [
    "Company Name",
    "Role/Position",
    "Skills Required",
    "Allowance",
    "Location",
    "Remote Option",
    "Company Reputation Score"
]
NUMERIC_FIELDS = ["Allowance", "Company Reputation Score"]
REMOTE_OPTIONS = {"Yes", "No"}
DEFAULT_WEIGHTS = {
    "Allowance": 0.2,
    "Location": 0.2,
    "Skills Match": 0.2,
    "Remote Option": 0.2,
    "Company Reputation Score": 0.2
}


def validate_internship_option(option: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validates a single internship option.
    Returns (is_valid, list_of_errors)
    """
    errors = []
    for field in REQUIRED_FIELDS:
        if field not in option or option[field] in (None, ""):
            errors.append(f"Missing or empty field: {field}")
    for field in NUMERIC_FIELDS:
        try:
            float(option.get(field, ""))
        except (ValueError, TypeError):
            errors.append(f"Field '{field}' must be a valid number.")
    remote = option.get("Remote Option", "")
    if remote not in REMOTE_OPTIONS:
        errors.append(f"Remote Option must be 'Yes' or 'No'.")
    return (len(errors) == 0, errors)


def validate_internship_dataset(dataset: List[Dict[str, Any]]) -> Tuple[bool, List[Tuple[int, List[str]]]]:
    """
    Validates a list of internship options.
    Returns (all_valid, list_of_(index, errors))
    """
    all_valid = True
    error_list = []
    for idx, option in enumerate(dataset):
        valid, errors = validate_internship_option(option)
        if not valid:
            all_valid = False
            error_list.append((idx, errors))
    return all_valid, error_list


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """
    Normalizes the weights so they sum to 1. If all weights are zero, returns default weights.
    """
    total = sum(weights.values())
    if total == 0:
        return DEFAULT_WEIGHTS.copy()
    return {k: v / total for k, v in weights.items()}


def handle_user_weights(user_weights: Optional[Dict[str, float]]) -> Dict[str, float]:
    """
    Processes user-provided weights, normalizes them, and fills in missing criteria with default weights.
    Warns if all weights are zero.
    """
    if not user_weights:
        return DEFAULT_WEIGHTS.copy()
    # Fill missing keys with 0
    weights = {k: user_weights.get(k, 0.0) for k in DEFAULT_WEIGHTS.keys()}
    normalized = normalize_weights(weights)
    return normalized


def check_edge_cases(dataset: List[Dict[str, Any]], weights: Dict[str, float]) -> List[str]:
    """
    Checks for edge cases and returns a list of warning messages.
    """
    warnings = []
    if not dataset:
        warnings.append("No internship options provided.")
        return warnings
    # Check if all options have missing scores
    all_missing_scores = all(
        any(option.get(field, "") in (None, "") for field in NUMERIC_FIELDS)
        for option in dataset
    )
    if all_missing_scores:
        warnings.append("All options have missing numeric scores.")
    # Check if all internships are remote or none are
    remote_values = {option.get("Remote Option", "") for option in dataset}
    if remote_values == {"Yes"}:
        warnings.append("All internships are remote.")
    elif remote_values == {"No"}:
        warnings.append("No internships are remote.")
    # Check if all weights are zero
    if sum(weights.values()) == 0:
        warnings.append("All weights are zero. Default weights will be used.")
    return warnings


def get_skills_match(user_skills: List[str], required_skills: str) -> float:
    """
    Computes a simple skills match score between user skills and required skills (comma-separated string).
    Returns a float between 0 and 1.
    """
    required = [s.strip().lower() for s in required_skills.split(",") if s.strip()]
    if not required:
        return 1.0  # If no skills required, full match
    user = set(s.strip().lower() for s in user_skills)
    match_count = sum(1 for skill in required if skill in user)
    return match_count / len(required)
