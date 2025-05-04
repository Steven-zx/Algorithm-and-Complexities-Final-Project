# data.py
import pandas as pd
import numpy as np
import os
import io
import csv

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
    Loads and cleans internship data from a CSV file, handling inconsistencies
    in the "Skills Required" field.
    """

    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, csv_file)

    print(f"Attempting to load CSV from: {csv_path}")

    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            if [col.strip() for col in header] != EXPECTED_COLUMNS:
                raise ValueError(f"Header mismatch. Expected: {EXPECTED_COLUMNS}, Found: {[col.strip() for col in header]}")

            data = []
            for row in reader:
                if len(row) > len(EXPECTED_COLUMNS):
                    # Combine extra fields in "Skills Required"
                    skills_required = ', '.join(row[2:len(row) - (len(row) - 3)])
                    cleaned_row = [row[0], row[1], skills_required] + row[len(row) - (len(row) - 3):]
                    if len(cleaned_row) == len(EXPECTED_COLUMNS):
                        data.append(cleaned_row)
                elif len(row) < len(EXPECTED_COLUMNS):
                    # Pad missing fields
                    padding = [''] * (len(EXPECTED_COLUMNS) - len(row))
                    data.append(row + padding)
                else:
                    data.append(row)

        # Create DataFrame
        df = pd.DataFrame(data, columns=EXPECTED_COLUMNS)
        print(f"DataFrame loaded successfully. Shape before cleaning: {df.shape}")

    except FileNotFoundError:
        raise FileNotFoundError(f"Error: CSV file '{csv_file}' not found in the same directory.")
    except ValueError as ve:
        raise ValueError(str(ve))
    except pd.errors.EmptyDataError:
        raise pd.errors.EmptyDataError(f"Error: CSV file '{csv_file}' is empty.")
    except Exception as e:
        raise Exception(f"An unexpected error occurred while processing '{csv_file}': {e}")

    # --- Data Cleaning ---

    # Handle Missing Values
    df.fillna(
        {
            "Allowance": 0,
            "Remote Option": "No",
            "Company Reputation Score": 0,
        },
        inplace=True,
    )
    df.dropna(subset=["Company Name", "Role/Position", "Skills Required"], inplace=True)

    # Standardize Text
    remote_options = {
        "Yes": ["yes", "true", "1", "y", "YES"],
        "No": ["no", "false", "0", "n", "NO"],
    }

    def standardize_remote(value):
        value = str(value).lower().strip()
        for key, synonyms in remote_options.items():
            if value in synonyms:
                return key
        return "No"

    df["Remote Option"] = df["Remote Option"].apply(standardize_remote)
    df["Location"] = df["Location"].str.strip()

    # Correct Data Types
    df["Allowance"] = pd.to_numeric(df["Allowance"], errors="coerce").fillna(0)
    df["Company Reputation Score"] = pd.to_numeric(df["Company Reputation Score"], errors="coerce").fillna(0).astype(int)

    # Remove Duplicates
    df.drop_duplicates(inplace=True)
    df.drop_duplicates(subset=["Company Name", "Role/Position"], inplace=True)

    print(f"DataFrame shape after cleaning: {df.shape}")

    return df


def save_internship_data(df, csv_file="internships.csv"):
    """
    Saves the internship data from a Pandas DataFrame back to a CSV file.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, csv_file)
    missing_columns = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Error: DataFrame is missing the following columns: "
            f"{', '.join(missing_columns)}"
        )
    df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    try:
        internships_df = load_internship_data()
        print("Data loaded successfully!")
        print(internships_df.head())
    except (
        FileNotFoundError,
        ValueError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as e:
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")