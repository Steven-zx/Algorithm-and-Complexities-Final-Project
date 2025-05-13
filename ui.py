import streamlit as st
import pandas as pd
from typing import Dict, List, Tuple, Any
import numpy as np

from data import load_internship_data
from algorithms import (
    compare_algorithms,
    plot_algorithm_comparison,
)
from utils import (
    validate_internship_dataset,
    handle_user_weights,
    check_edge_cases,
    get_skills_match,
)
from core import calculate_distance_relevance, LOCATION_WEIGHTS

def render_skills_input(all_skills_list: List[str]) -> List[str]:
    """Renders the skills input section and returns selected skills."""
    st.markdown("**Select Your Skills**")
    
    # Initialize session state for skills if not exists
    if 'selected_skills' not in st.session_state:
        st.session_state.selected_skills = []
    
    # Create a container for the skills selection
    skills_container = st.container()
    with skills_container:
        # Add a new skill selection with search
        new_skill = st.selectbox(
            "Add a skill",
            options=[""] + sorted(all_skills_list),  # Empty string as first option
            key="skill_selector"
        )
        
        if new_skill and new_skill not in st.session_state.selected_skills:
            st.session_state.selected_skills.append(new_skill)
            st.rerun()
        
        # Display selected skills with remove option
        if st.session_state.selected_skills:
            st.markdown("**Selected Skills:**")
            for skill in st.session_state.selected_skills:
                col1, col2 = st.columns([0.9, 0.1])
                with col1:
                    st.markdown(f"- {skill}")
                with col2:
                    if st.button("❌", key=f"remove_{skill}"):
                        st.session_state.selected_skills.remove(skill)
                        st.rerun()
    
    return st.session_state.selected_skills

def render_preferences_section() -> Tuple[str, List[str], Dict[str, float], bool]:
    """Renders the preferences section and returns user inputs."""
    with st.container():
        st.markdown('<div style="display: flex; flex-direction: column; align-items: center;">', unsafe_allow_html=True)
        st.header("Your Preferences")

        # User Location Input
        user_location = st.selectbox("Your Current Location", sorted(list(LOCATION_WEIGHTS.keys())))

        # Get all available skills
        internship_df = load_internship_data()
        internship_data = internship_df.to_dict(orient="records")
        all_skills = set()
        for internship in internship_data:
            if isinstance(internship.get("Skills Required"), str):
                skills = [skill.strip().lower() for skill in internship["Skills Required"].split(",")]
                all_skills.update(skills)
        all_skills_list = sorted(list(all_skills))

        # Skills Input
        user_skills = render_skills_input(all_skills_list)

        st.subheader("Importance of Factors:")
        importance_levels = ["Not Important", "Slightly Important", "Important", "Very Important"]
        importance_mapping = {
            "Not Important": 0.0,
            "Slightly Important": 0.3,
            "Important": 0.7,
            "Very Important": 1.0,
        }

        weights_input = {}
        weights_input["Allowance"] = importance_mapping[
            st.selectbox("Allowance", importance_levels, index=2)
        ]
        weights_input["Location"] = importance_mapping[
            st.selectbox("Location", importance_levels, index=2)
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

    return user_location, user_skills, weights_input, calculate_button

def render_results(internship_data: List[Dict], user_location: str, user_skills: List[str], 
                  weights_input: Dict[str, float], results_container: Any):
    """Renders the results section."""
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
            # Lowercase location for comparison
            option_location = option["Location"].strip().lower() if isinstance(option["Location"], str) else ""
            user_location_lower = user_location.strip().lower() if isinstance(user_location, str) else ""
            location_relevance = calculate_distance_relevance(user_location_lower, option_location)
            
            # Lowercase skills for comparison
            option_skills = option["Skills Required"].lower() if isinstance(option["Skills Required"], str) else ""
            user_skills_lower = [s.lower() for s in user_skills]
            skills_match = get_skills_match(user_skills_lower, option_skills)

            # Calculate remote option score (1.0 for remote, 0.0 for non-remote)
            remote_score = 1.0 if str(option["Remote Option"]).strip().lower() in ["yes", "true", "1"] else 0.0
            
            # Calculate allowance score (normalize to 0-1 range)
            max_allowance = max(float(opt["Allowance"]) for opt in internship_data)
            allowance_score = float(option["Allowance"]) / max_allowance if max_allowance > 0 else 0.0
            
            # Calculate reputation score (normalize to 0-1 range)
            max_reputation = max(float(opt["Company Reputation Score"]) for opt in internship_data)
            reputation_score = float(option["Company Reputation Score"]) / max_reputation if max_reputation > 0 else 0.0

            features_matrix.append(
                [
                    allowance_score * normalized_weights["Allowance"],
                    location_relevance * normalized_weights["Location"],
                    skills_match * normalized_weights["Skills Match"],
                    remote_score * normalized_weights["Remote Option"],
                    reputation_score * normalized_weights["Company Reputation Score"],
                ]
            )

        if features_matrix:
            # Compare different algorithms
            algorithm_results = compare_algorithms(
                np.array(features_matrix), 
                np.array(list(normalized_weights.values()))
            )
            
            # Use QR Decomposition for final rankings (most stable)
            ranking_scores = algorithm_results["qr"][0]
            
            ranked_internships = sorted(
                zip(internship_data, ranking_scores), key=lambda x: x[1], reverse=True
            )

            if ranked_internships:
                st.subheader("🏆 Top 10 Recommended OJT Placements")
                top_10_results = ranked_internships[:10]

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

                # Move Algorithm Performance Comparison and Visualization here
                st.subheader("📊 Algorithm Performance Comparison")
                metrics_df = pd.DataFrame([
                    {
                        "Algorithm": metrics["method"],
                        "Time Complexity": metrics["time_complexity"],
                        "Execution Time (s)": f"{metrics['execution_time']:.4f}",
                        "Numerical Stability": metrics["numerical_stability"]
                    }
                    for _, (_, metrics) in algorithm_results.items()
                ])
                st.dataframe(metrics_df)
                
                st.subheader("📈 Algorithm Comparison Visualization")
                fig = plot_algorithm_comparison(algorithm_results)
                st.pyplot(fig)

                # Display algorithm comparison explanation
                st.subheader("📝 Algorithm Analysis")
                st.markdown("""
                ### Algorithm Comparison Analysis
                
                1. **Gaussian Elimination**
                   - Time Complexity: O(n^3)
                   - Medium numerical stability
                   - Good for exact solutions
                   - Efficient for small to medium-sized datasets
                
                2. **QR Decomposition**
                   - Time Complexity: O(n^3)
                   - Highest numerical stability
                   - Most robust to ill-conditioned data
                   - Best choice for large datasets with potential numerical issues
                
                3. **Cramer's Rule**
                   - Time Complexity: O(n^4)
                   - Most computationally expensive
                   - Low numerical stability
                   - Demonstrates theoretical approach but not practical for large datasets
                
                The QR Decomposition method was chosen as the final ranking algorithm because it provides the best balance of numerical stability and efficiency for our use case.
                """)
            else:
                st.info("No internships to display based on your preferences.")
        else:
            st.info("No internship data available.")

def render_ui():
    """Main UI rendering function."""
    st.title("🎓 OJT Optimizer")
    st.markdown("Find the best On-the-Job Training based on your preferences!")

    try:
        # Load and validate data
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

        # Render preferences section
        user_location, user_skills, weights_input, calculate_button = render_preferences_section()

        # Placeholder for Results
        results_container = st.container()

        # Processing and Display Logic (Only when the button is pressed)
        if calculate_button:
            render_results(internship_data, user_location, user_skills, weights_input, results_container)

    except FileNotFoundError as e:
        st.error(str(e))
    except pd.errors.EmptyDataError:
        st.error("Error: The CSV file is empty.")
    except pd.errors.ParserError as e:
        st.error(f"Error parsing CSV file: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred during data loading or processing: {e}")

if __name__ == "__main__":
    render_ui()