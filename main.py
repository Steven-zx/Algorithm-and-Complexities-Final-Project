# main.py
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List
import io

from data import load_internship_data
from algorithms import calculate_rankings
from utils import (
    validate_internship_dataset,
    handle_user_weights,
    check_edge_cases,
    get_skills_match,
)

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
    "zamboanga city": 0.25, # Weight if the chosen location is Zamboanga City
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
    st.title("🎓 OJT Optimizer")
    st.markdown("Find the best On-the-Job Training based on your preferences!")

    try:
        internship_df = load_internship_data()
        internship_data = internship_df.to_dict(orient="records")

        if not internship_data:
            st.warning("⚠️ No internship data available.")
            return

        valid_data, validation_errors = validate_internship_dataset(internship_data)
        if not valid_data:
            st.error("Error in the dataset:")
            for index, errors in validation_errors:
                st.markdown(f"**Row {index + 1}:** {', '.join(errors)}")
            return

        st.success("✅ Internship data loaded and validated successfully!")

        all_skills = set()
        for internship in internship_data:
            if isinstance(internship.get("Skills Required"), str):
                skills = [skill.strip().lower() for skill in internship["Skills Required"].split(",")]
                all_skills.update(skills)
        all_skills_list = sorted(list(all_skills))

        # --- Preferences Section (Centered) ---
        with st.container():
            st.markdown('<div style="display: flex; flex-direction: column; align-items: center;">', unsafe_allow_html=True)
            st.header("Your Preferences")

            # User Location Input
            user_location = st.selectbox("Your Current Location", sorted(list(LOCATION_WEIGHTS.keys())))

            # Autocompletion for Skills
            user_skills_input = st.text_input("Your Skills", "")
            suggested_skills = [
                skill for skill in all_skills_list if user_skills_input.lower() in skill
            ]
            user_skills = [
                st.selectbox("Select Skill", suggested_skills)
                for _ in range(max(1, user_skills_input.count(',') + 1))  # Basic handling for multiple skills
            ]
            user_skills = [skill for skill in user_skills if skill] # Remove None values

            st.subheader("Importance of Factors:")
            importance_levels = ["Not Important", "Slightly Important", "Important", "Very Important"]
            importance_mapping = {
                "Not Important": 0.0,
                "Slightly Important": 0.2,
                "Important": 0.6,
                "Very Important": 1.0,
            }

            weights_input = {}
            weights_input["Allowance"] = importance_mapping[
                st.selectbox("Allowance", importance_levels, index=2)
            ]
            weights_input["Location"] = importance_mapping[
                st.selectbox("Location", importance_levels, index=1)
            ]
            weights_input["Skills Match"] = importance_mapping[
                st.selectbox("Skills Match", importance_levels, index=3)
            ]
            weights_input["Remote Option"] = importance_mapping[
                st.selectbox("Remote Option", importance_levels, index=1)
            ]
            weights_input["Company Reputation Score"] = importance_mapping[
                st.selectbox("Reputation Score", importance_levels, index=1)
            ]

            calculate_button = st.button("✨ Find My Best OJT Matches!")
            st.markdown('</div>', unsafe_allow_html=True) # Closing the centering div

        # Placeholder for Results
        results_container = st.container()

        # Processing and Display Logic (Only when the button is pressed)
        if calculate_button:
            with results_container:
                st.header("🏆 Top 10 Recommended OJT Placements")
                normalized_weights = handle_user_weights(weights_input)

                warnings = check_edge_cases(internship_data, normalized_weights)
                if warnings:
                    st.warning("⚠️ Potential Issues:")
                    for warning in warnings:
                        st.markdown(f"- {warning}")

                features_matrix = []
                for option in internship_data:
                    # Calculate location relevance based on user's choice
                    location_relevance = calculate_distance_relevance(user_location, option["Location"])

                    features_matrix.append(
                        [
                            float(option["Allowance"]),
                            location_relevance * normalized_weights["Location"], # Apply the weight
                            get_skills_match(user_skills, option["Skills Required"]),
                            1.0 if str(option["Remote Option"]).lower() in ["yes", "true", "1"] else 0.0,
                            float(option["Company Reputation Score"]),
                        ]
                    )

                if features_matrix:
                    ranking_scores = calculate_rankings(
                        np.array(features_matrix), np.array(list(normalized_weights.values()))
                    )

                    ranked_internships = sorted(
                        zip(internship_data, ranking_scores), key=lambda x: x[1], reverse=True
                    )

                    if ranked_internships:
                        top_10_results = ranked_internships[:10] # Take only the top 10

                        for i, (internship, score) in enumerate(top_10_results, 1):
                            st.subheader(
                                f"Rank {i}: {internship['Company Name']} - {internship['Role/Position']}"
                            )
                            st.markdown(f"**Score:** {score:.4f}")
                            st.markdown(f"**Location:** {internship['Location']}")
                            st.markdown(f"**Allowance:** {internship['Allowance']}")
                            st.markdown(f"**Skills Required:** {internship['Skills Required']}")
                            st.markdown(f"**Remote Option:** {internship['Remote Option']}")
                            st.markdown(f"**Reputation Score:** {internship['Company Reputation Score']}")
                            st.markdown("---")
                        if len(ranked_internships) > 10:
                            st.info("Showing the top 10 results.")
                    else:
                        st.info("No internships to display based on your preferences.")
                else:
                    st.info("No internship data available.")

    except FileNotFoundError as e:
        st.error(str(e))
    except pd.errors.EmptyDataError:
        st.error("Error: The CSV file is empty.")
    except pd.errors.ParserError as e:
        st.error(f"Error parsing CSV file: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred during data loading or processing: {e}")

if __name__ == "__main__":
    main()