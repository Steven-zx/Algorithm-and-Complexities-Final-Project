from typing import List, Dict, Any, Tuple, Optional
# utils.py
import numpy as np
from difflib import SequenceMatcher

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

# Skill aliases and variations mapping
SKILL_ALIASES = {
    "python": ["python programming", "python3", "python 3", "py", "data analysis", "data science", "machine learning", "ai", "artificial intelligence"],
    "javascript": ["js", "ecmascript", "es6", "node.js", "nodejs", "web development", "front-end", "frontend"],
    "java": ["java programming", "j2ee", "j2se", "android development"],
    "c++": ["cpp", "c plus plus", "cplusplus", "game development", "unreal engine"],
    "c#": ["csharp", "dotnet", ".net", "asp.net", "unity", "unity3d", "unity development", "game development", "game dev", "gamedev"],
    "html": ["html5", "hypertext markup language", "web development", "front-end", "frontend"],
    "css": ["css3", "cascading style sheets", "web development", "front-end", "frontend"],
    "sql": ["mysql", "postgresql", "postgres", "oracle sql", "mssql", "database", "data analysis"],
    "react": ["reactjs", "react.js", "react native", "front-end", "frontend", "web development"],
    "angular": ["angularjs", "angular.js", "front-end", "frontend", "web development"],
    "vue": ["vue.js", "vuejs", "front-end", "frontend", "web development"],
    "php": ["php programming", "php7", "php 7", "web development", "backend"],
    "ruby": ["ruby on rails", "rails", "ror", "web development", "backend"],
    "swift": ["swift programming", "ios development", "mobile development"],
    "kotlin": ["android development", "android programming", "mobile development"],
    "typescript": ["ts", "typescript programming", "web development", "front-end", "frontend"],
    "unity": ["unity3d", "unity development", "unity game development", "unity engine", "game development", "game dev", "gamedev", "c#", "csharp"],
    "game development": ["game dev", "gamedev", "unity", "unreal", "unreal engine", "c#", "csharp", "game design", "game programming"],
    "machine learning": ["ml", "ai", "artificial intelligence", "deep learning", "data science", "python"],
    "data science": ["data analysis", "data analytics", "statistics", "python", "machine learning", "ai"],
    "devops": ["ci/cd", "continuous integration", "continuous deployment", "cloud", "aws", "azure"],
    "cloud": ["aws", "azure", "google cloud", "gcp", "devops"],
    "mobile": ["ios", "android", "react native", "flutter", "swift", "kotlin"],
    "web": ["frontend", "backend", "full stack", "fullstack", "html", "css", "javascript"],
    "database": ["sql", "nosql", "mongodb", "postgresql", "mysql", "data analysis"],
    "networking": ["network security", "cybersecurity", "security", "network administration"],
    "ui/ux": ["user interface", "user experience", "design", "front-end", "frontend"],
    "testing": ["qa", "quality assurance", "automation testing", "selenium", "junit"],
    "blockchain": ["web3", "cryptocurrency", "smart contracts", "solidity", "ethereum"],
    "embedded": ["iot", "internet of things", "hardware programming", "arduino", "raspberry pi"]
}

def get_skill_variations(skill: str) -> List[str]:
    """Get all variations of a skill including aliases."""
    skill = skill.lower().strip()
    variations = {skill}  # Start with the skill itself
    
    # Add aliases if they exist
    for main_skill, aliases in SKILL_ALIASES.items():
        if skill in aliases or skill == main_skill:
            variations.update(aliases)
            variations.add(main_skill)
    
    return list(variations)

def similar(a: str, b: str) -> float:
    """Calculate similarity ratio between two strings."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def get_skills_match(user_skills: List[str], required_skills: str) -> float:
    """
    Computes a skills match score between user skills and required skills using fuzzy matching.
    Returns a float between 0 and 1.
    """
    if not required_skills.strip():
        return 1.0  # If no skills required, full match
    
    # Get all variations of required skills
    required = []
    for skill in required_skills.split(','):
        skill = skill.strip().lower()
        if skill:
            required.extend(get_skill_variations(skill))
    
    if not required:
        return 1.0  # If no valid required skills, full match
    
    # Get all variations of user skills
    user_variations = []
    for skill in user_skills:
        user_variations.extend(get_skill_variations(skill))
    
    if not user_variations:
        return 0.0  # If no user skills, no match
    
    # Calculate match scores for each required skill
    match_scores = []
    for req_skill in required:
        best_match = max(
            (similar(req_skill, user_skill) for user_skill in user_variations),
            default=0.0
        )
        # Only count matches above 0.6 similarity
        if best_match > 0.6:
            match_scores.append(best_match)
    
    # If no good matches found, return 0
    if not match_scores:
        return 0.0
    
    # Return average match score
    return sum(match_scores) / len(match_scores)

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
