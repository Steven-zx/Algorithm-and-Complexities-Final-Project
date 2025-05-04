import pandas as pd
import numpy as np

# Define the expected columns in the CSV (to ensure data integrity)
EXPECTED_COLUMNS = [
    "Company Name",
    "Role/Position",
    "Skills Required",
    "Allowance",
    "Location",
    "Remote Option",
    "Company Reputation Score",
]

def load_internship_data(csv_file="internships.csv"):
    """
    Loads internship data from a CSV file into a Pandas DataFrame.

    Args:
        csv_file (str, optional): The path to the CSV file.
                                 Defaults to "internships.csv".

    Returns:
        pandas.DataFrame: A DataFrame containing the internship data.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If the CSV file is missing required columns.
    """

    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: CSV file '{csv_file}' not found.")

    # Validate columns
    missing_columns = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Error: CSV file '{csv_file}' is missing the following columns: "
            f"{', '.join(missing_columns)}"
        )

    # Data type conversion and cleaning
    # Convert 'Allowance' to numeric (handling potential errors)
    df['Allowance'] = pd.to_numeric(df['Allowance'], errors='coerce')
    df.dropna(subset=['Allowance'], inplace=True)  # Remove rows with missing Allowance

    # Convert 'Remote Option' to boolean
    df['Remote Option'] = df['Remote Option'].apply(
        lambda x: True if str(x).lower() in ['yes', 'true', '1'] else False
    )

    # Convert  'Company Reputation Score' to numeric
    df['Company Reputation Score'] = pd.to_numeric(df['Company Reputation Score'], errors='coerce')
    df.dropna(subset=['Company Reputation Score'], inplace=True)

    return df


def save_internship_data(df, csv_file="internships.csv"):
    """
    Saves the internship data from a Pandas DataFrame back to a CSV file.

    Args:
        df (pandas.DataFrame): The DataFrame to save.
        csv_file (str, optional): The path to the CSV file.
                                 Defaults to "internships.csv".
    """

    # Basic validation (optional, but recommended)
    missing_columns = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Error: DataFrame is missing the following columns: "
            f"{', '.join(missing_columns)}"
        )

    df.to_csv(csv_file, index=False)  # index=False to avoid saving Pandas index


if __name__ == '__main__':
    # Example Usage
    try:
        internships_df = load_internship_data()
        print("Data loaded successfully!")
        print(internships_df.head())  # Print the first few rows

        # Example: Modifying data (for demonstration)
        # internships_df.loc[0, 'Allowance'] = 5500  # Update allowance

        # Example: Saving data (optional)
        # save_internship_data(internships_df, "updated_internships.csv")
        # print("\nData saved to updated_internships.csv!")

    except (FileNotFoundError, ValueError) as e:
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")