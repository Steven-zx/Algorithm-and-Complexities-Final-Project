# core.py
# Predefined location weights relative to a chosen location
LOCATION_WEIGHTS = {
    "zamboanga city": 1.0,  # Highest weight for exact match
    "general santos city": 0.8,  # Nearby city
    "davao city": 0.7,  # Nearby city
    "cebu city": 0.6,  # Nearby island
    "dumaguete city": 0.6,  # Nearby island
    "iloilo city": 0.5,  # Nearby island
    "bacolod city": 0.5,  # Nearby island
    "manila": 0.3,    # Far away
    "quezon city": 0.3,  # Far away
    "makati city": 0.3,  # Far away
    "taguig city": 0.3,  # Far away
    "pasig city": 0.3,  # Far away
    "baguio city": 0.3,  # Far away
    "pasay city": 0.3,  # Far away
    "laguna": 0.3,    # Far away
    "antique": 0.4,   # Nearby island
    "palawan": 0.4,   # Nearby island
    "cavite": 0.3,    # Far away
    "bago city": 0.5,  # Nearby island
}

def calculate_distance_relevance(user_location, internship_location):
    """
    Assigns a relevance score based on the user's chosen location
    and the internship location, using predefined weights.
    Higher score means closer to user's location.
    """
    user_loc_lower = user_location.lower()
    intern_loc_lower = internship_location.lower()

    # If locations match exactly, return highest score
    if user_loc_lower == intern_loc_lower:
        return 1.0

    # Get the weight for the internship location
    if intern_loc_lower in LOCATION_WEIGHTS:
        return LOCATION_WEIGHTS[intern_loc_lower]
    
    # If location not in our predefined list, return a low score
    return 0.2

    # A very simplistic way to calculate relevance: higher if weights are similar
    relevance = 1.0 - abs(user_base_weight - intern_weight)
    return max(0.1, relevance) # Ensure a minimum relevance 