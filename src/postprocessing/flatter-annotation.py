#!/usr/bin/env python3
"""
Script to flatten JSON data from optimized_final_decision column in CSV files.
Processes all CSV files in result/optimized/ directory and outputs flattened versions.
"""

import pandas as pd
import json
import os
import glob
from pathlib import Path


def flatten_json_column(df, json_column='optimized_final_decision'):
    """
    Flatten JSON data from a specific column and add as new columns.

    Args:
        df: pandas DataFrame containing the data
        json_column: name of the column containing JSON data

    Returns:
        DataFrame with flattened JSON data as separate columns
    """
    # Create a copy of the original dataframe
    result_df = df.copy()

    # Lists to store the flattened data
    roles = []
    confidences = []
    notes = []
    justifications = []
    context_assessments = []

    for idx, row in df.iterrows():
        json_data = row[json_column]

        try:
            # Parse JSON string
            if pd.isna(json_data) or json_data == '':
                # Handle empty/null values
                parsed_data = {}
            else:
                parsed_data = json.loads(json_data)

            # Extract fields with defaults
            roles.append(parsed_data.get('role', ''))
            confidences.append(parsed_data.get('confidence', ''))
            notes.append(parsed_data.get('note', ''))
            justifications.append(parsed_data.get('justification', ''))
            context_assessments.append(parsed_data.get('context_assessment', ''))

        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse JSON at row {idx}: {e}")
            # Add empty values for failed parsing
            roles.append('')
            confidences.append('')
            notes.append('')
            justifications.append('')
            context_assessments.append('')
        except Exception as e:
            print(f"Warning: Unexpected error at row {idx}: {e}")
            roles.append('')
            confidences.append('')
            notes.append('')
            justifications.append('')
            context_assessments.append('')

    # Add flattened columns to the result dataframe
    result_df['role'] = roles
    result_df['confidence'] = confidences
    result_df['note'] = notes
    result_df['justification'] = justifications
    result_df['context_assessment'] = context_assessments

    return result_df


def process_csv_files(input_dir='result/optimized', output_dir='result/flattened-annotation'):
    """
    Process all CSV files in the input directory and create flattened versions.

    Args:
        input_dir: directory containing CSV files to process
        output_dir: directory to save flattened CSV files
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Find all CSV files in the input directory
    csv_pattern = os.path.join(input_dir, '*.csv')
    csv_files = glob.glob(csv_pattern)

    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return

    print(f"Found {len(csv_files)} CSV files to process:")

    for csv_file in csv_files:
        try:
            print(f"\nProcessing: {csv_file}")

            # Read the CSV file
            df = pd.read_csv(csv_file)

            # Check if optimized_final_decision column exists
            if 'optimized_final_decision' not in df.columns:
                print(f"Warning: 'optimized_final_decision' column not found in {csv_file}")
                continue

            # Flatten the JSON data
            flattened_df = flatten_json_column(df)

            # Generate output filename
            input_filename = os.path.basename(csv_file)
            output_filename = f"flattened_{input_filename}"
            output_path = os.path.join(output_dir, output_filename)

            # Save the flattened data
            flattened_df.to_csv(output_path, index=False)

            print(f"✓ Saved flattened data to: {output_path}")
            print(f"  Original rows: {len(df)}")
            print(f"  Flattened columns added: role, confidence, note, justification, context_assessment")

        except Exception as e:
            print(f"✗ Error processing {csv_file}: {e}")


def main():
    """Main function to run the flattening process."""
    print("Starting JSON flattening process...")
    print("=" * 50)

    # Default directories (adjust as needed)
    input_directory = 'result/optimized'
    output_directory = 'result/flattened-annotation'

    # Check if input directory exists
    if not os.path.exists(input_directory):
        print(f"Error: Input directory '{input_directory}' does not exist.")
        print("Please ensure the directory path is correct.")
        return

    # Process the files
    process_csv_files(input_directory, output_directory)

    print("\n" + "=" * 50)
    print("JSON flattening process completed!")


if __name__ == "__main__":
    main()