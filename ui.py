import streamlit as st
import pandas as pd
from typing import Dict, List
import numpy as np

from data import load_internship_data
from algorithms import calculate_rankings
from utils import (
     validate_internship_dataset,
     handle_user_weights,
     check_edge_cases,
     get_skills_match
 )

def render_ui():
     st.title("🎓 Internship Position Ranker")
     st.markdown("Make smarter OJT choices based on your preferences!")

     # --- Data Loading ---
     st.header("📁 Internship Dataset")
     st.info("Data will be loaded automatically from 'internships.csv'.")

     try:
         internship_df = load_internship_data()
         dataset = internship_df.to_dict(orient="records")

         valid, errors = validate_internship_dataset(dataset)
         if not valid:
             st.error("Some entries in the dataset are invalid:")
             for idx, errs in errors:
                 st.markdown(f"**Entry {idx+1}:**")
                 for err in errs:
                     st.markdown(f"- {err}")
             return

         st.success("✅ Internship data loaded and validated successfully!")

         # User Preferences
         st.header("⚙️ Your Preferences")
         user_skills = st.text_input("Enter your skills (comma-separated)", "").split(",")

         st.subheader("Weights (importance of each criterion)")
         user_weights = {
             "Allowance": st.slider("Allowance", 0.0, 1.0, 0.2),
             "Location": st.slider("Location", 0.0, 1.0, 0.2),
             "Skills Match": st.slider("Skills Match", 0.0, 1.0, 0.2),
             "Remote Option": st.slider("Remote Option", 0.0, 1.0, 0.2),
             "Company Reputation Score": st.slider("Reputation", 0.0, 1.0, 0.2)
         }

         normalized_weights = handle_user_weights(user_weights)
         warnings = check_edge_cases(dataset, normalized_weights)
         if warnings:
             st.warning("⚠️ Warnings:")
             for warn in warnings:
                 st.markdown(f"- {warn}")

         # Prepare data matrix
         features_matrix = []
         for option in dataset:
             features_matrix.append([
                 float(option["Allowance"]),
                 1.0 if normalized_weights["Location"] > 0 else 0.5,  # Adjusted logic
                 get_skills_match(user_skills, option["Skills Required"]),
                 1.0 if str(option["Remote Option"]).lower() == "yes" else 0.0,
                 float(option["Company Reputation Score"])
             ])

         scores = calculate_rankings(np.array(features_matrix), np.array(list(normalized_weights.values())))

         # Display Rankings
         st.header("🏆 Internship Rankings")
         ranked = sorted(zip(dataset, scores), key=lambda x: x[1], reverse=True)
         for i, (option, score) in enumerate(ranked, 1):
             st.markdown(f"### {i}. {option['Company Name']} - {option['Role/Position']}")
             st.markdown(f"**Score:** {score:.4f}")
             st.markdown(f"📍 Location: {option['Location']} | 💵 Allowance: {option['Allowance']} | ⭐ Reputation: {option['Company Reputation Score']}")
             st.markdown("---")

     except FileNotFoundError:
         st.error(f"Error: The file 'internships.csv' was not found in the same directory.")
     except pd.errors.EmptyDataError:
         st.error("Error: The CSV file 'internships.csv' is empty.")
     except Exception as e:
         st.error(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
     render_ui()