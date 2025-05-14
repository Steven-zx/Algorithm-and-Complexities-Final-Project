# main.py
import streamlit as st
from ui import render_ui

# Predefined location weights relative to a chosen location
LOCATION_WEIGHTS = {
    "bago city": 0.8,  # Weight if the chosen location is Bago City
    "iloilo city": 0.9, # Weight if the chosen location is Iloilo City
    "bacolod city": 0.85, # Weight if the chosen location is Bacolod City
    "cebu city": 0.65, # Weight if the chosen location is Cebu City
    "davao city": 0.25, # Weight if the chosen location is Davao City
    "manila": 0.35,    # Weight if the chosen location is in Manila
    "quezon city": 0.35, # Weight if the chosen location is Quezon City
    "makati city": 0.4,  # Weight if the chosen location is Makati City
    "taguig city": 0.4,  # Weight if the chosen location is Taguig City
    "pasig city": 0.35,  # Weight if the chosen location is Pasig City
    "baguio city": 0.45, # Weight if the chosen location is Baguio City
    "pasay city": 0.35,  # Weight if the chosen location is Pasay City
    "dumaguete city": 0.55, # Weight if the chosen location is Dumaguete City
    "laguna": 0.45,    # Weight if the chosen location is in Laguna
    "antique": 0.75,   # Weight if the chosen location is Antique
    "general santos city": 0.25, # Weight if the chosen location is General Santos City
    "palawan": 0.35,   # Weight if the chosen location is Palawan
    "cavite": 0.45,    # Weight if the chosen location is in Cavite
}

def calculate_distance_relevance(user_location, internship_location):
    """
    Assigns a relevance score based on the user's chosen location
    and the internship location, using predefined weights.
    """
    user_loc_lower = user_location.lower()
    intern_loc_lower = internship_location.lower()

    if user_loc_lower in LOCATION_WEIGHTS:
        user_base_weight = LOCATION_WEIGHTS[user_loc_lower]
    else:
        user_base_weight = 0.5 # Default if user location is unknown

    if intern_loc_lower in LOCATION_WEIGHTS:
        intern_weight = LOCATION_WEIGHTS[intern_loc_lower]
    else:
        intern_weight = 0.5 # Default if internship location is unknown

    # A very simplistic way to calculate relevance: higher if weights are similar
    relevance = 1.0 - abs(user_base_weight - intern_weight)
    return max(0.1, relevance) # Ensure a minimum relevance

def main():
    render_ui()

if __name__ == "__main__":
    main()