#!/usr/bin/env python3
"""
Script to flatten and unpivot JSON data from Optimized_final_decision column.
Each expression (expr_1, expr_2, expr_3) becomes a separate row with its corresponding analysis.

Generates TWO output versions:
  - LONG: All flattened columns (45+ fields)
  - SHORT: Essential classification columns only (sentence, expression, classifications, reasoning)

Output structure:
  result/flattened-by-expression/
    ├── long/
    │   └── {filename}_LONG.csv
    └── short/
        └── {filename}_SHORT.csv
"""

import pandas as pd
import json
import csv
import os
import glob
from pathlib import Path


def extract_nested_fields(analysis_data):
    """
    Extract key fields from an analysis object for flattening.

    Args:
        analysis_data: dict containing the analysis for one expression

    Returns:
        dict with flattened key fields
    """
    if not analysis_data or not isinstance(analysis_data, dict):
        return {
            'form': '',
            'position': '',
            'punctuation': '',
            'grammatical_integration': '',
            'is_reflexive': '',
            'reflexivity_type': '',
            'reflexivity_reasoning': '',
            'scope_classification': '',
            'scope_reach': '',
            'removal_test': '',
            'boundary_test': '',
            'scope_reasoning': '',
            'L1_classification': '',
            'L1_reasoning': '',
            'L2_classification': '',
            'L2_reasoning': '',
            'L3_classification': '',
            'L3_borderline': '',
            'L3_reasoning': '',
            'confidence_D1': '',
            'confidence_D2': '',
            'confidence_D3': '',
            'confidence_overall': '',
            'confidence_justification': '',
            'L1_borderline_is': '',
            'L1_borderline_dominant': '',
            'L1_borderline_secondary': '',
            'L1_borderline_why': '',
            'L2_borderline_is': '',
            'L2_borderline_primary': '',
            'L2_borderline_secondary': '',
            'L2_borderline_tertiary': '',
            'L2_borderline_strength': '',
            'L2_borderline_why': '',
            'comprehensive_justification': '',
            'adjudication_agree': '',
            'adjudication_disagree': '',
            'adjudication_cause': '',
            'adjudication_basis': '',
            'adjudication_boundary_correction': '',
            'adjudication_violations': '',
            'adjudication_disqualified': '',
            'adjudication_context_sufficiency': '',
            'adjudication_improvement': '',
            'analysis_json_raw': ''
        }

    # Store raw JSON for reference (compact format - no newlines)
    # Using separators to ensure no extra whitespace that could cause CSV parsing issues
    result = {'analysis_json_raw': json.dumps(analysis_data, ensure_ascii=False, separators=(',', ':'))}

    # D1: Observable Realization
    d1 = analysis_data.get('D1', {})
    result['form'] = d1.get('form', '')
    result['position'] = d1.get('pos', '')
    result['punctuation'] = d1.get('punct', '')
    result['grammatical_integration'] = d1.get('gram_int', '')

    reflex = d1.get('reflex', {})
    result['is_reflexive'] = reflex.get('is', '')
    result['reflexivity_type'] = reflex.get('type', '')
    result['reflexivity_reasoning'] = reflex.get('why', '')

    # D2: Functional Scope
    d2 = analysis_data.get('D2', {})
    result['scope_classification'] = d2.get('class', '')
    result['scope_reach'] = d2.get('reach', '')
    result['removal_test'] = d2.get('removal', '')
    result['boundary_test'] = d2.get('boundary', '')
    result['scope_reasoning'] = d2.get('why', '')

    # D3: Metadiscourse Classification
    d3 = analysis_data.get('D3', {})
    result['L1_classification'] = d3.get('L1', '')
    result['L1_reasoning'] = d3.get('L1_why', '')
    result['L2_classification'] = d3.get('L2', '')
    result['L2_reasoning'] = d3.get('L2_why', '')
    result['L3_classification'] = d3.get('L3', '')
    result['L3_borderline'] = d3.get('L3_border', '')
    result['L3_reasoning'] = d3.get('L3_why', '')

    # Confidence ratings
    conf = analysis_data.get('conf', {})
    result['confidence_D1'] = conf.get('D1', '')
    result['confidence_D2'] = conf.get('D2', '')
    result['confidence_D3'] = conf.get('D3', '')
    result['confidence_overall'] = conf.get('overall', '')
    result['confidence_justification'] = conf.get('just', '')

    # Borderline classifications
    border = analysis_data.get('border', {})

    l1_border = border.get('L1', {})
    result['L1_borderline_is'] = l1_border.get('is', '')
    result['L1_borderline_dominant'] = l1_border.get('dom', '')
    result['L1_borderline_secondary'] = l1_border.get('sec', '')
    result['L1_borderline_why'] = l1_border.get('why', '')

    l2_border = border.get('L2', {})
    result['L2_borderline_is'] = l2_border.get('is', '')
    result['L2_borderline_primary'] = l2_border.get('pri', '')
    result['L2_borderline_secondary'] = l2_border.get('sec', '')
    result['L2_borderline_tertiary'] = l2_border.get('ter', '')
    result['L2_borderline_strength'] = l2_border.get('strength', '')
    result['L2_borderline_why'] = l2_border.get('why', '')

    # Comprehensive justification
    result['comprehensive_justification'] = analysis_data.get('comp_just', '')

    # Adjudication fields
    adj = analysis_data.get('adj', {})
    result['adjudication_agree'] = adj.get('agree', '')
    result['adjudication_disagree'] = adj.get('disagree', '')
    result['adjudication_cause'] = adj.get('cause', '')
    result['adjudication_basis'] = adj.get('basis', '')
    result['adjudication_boundary_correction'] = adj.get('boundary_corr', '')
    result['adjudication_violations'] = adj.get('violations', '')
    result['adjudication_disqualified'] = adj.get('disqual', '')
    result['adjudication_context_sufficiency'] = adj.get('ctx_suff', '')
    result['adjudication_improvement'] = adj.get('improve', '')

    return result


def flatten_and_unpivot(df, json_column='Optimized_final_decision'):
    """
    Flatten JSON data and create separate rows for each expression.

    Args:
        df: pandas DataFrame containing the data
        json_column: name of the column containing JSON data

    Returns:
        DataFrame with one row per expression, flattened analysis data
    """
    rows = []

    for idx, row in df.iterrows():
        json_data = row[json_column]

        try:
            # Parse JSON string
            if pd.isna(json_data) or json_data == '':
                print(f"Warning: Empty JSON at row {idx}")
                continue

            parsed_data = json.loads(json_data)

            # Common fields from the JSON
            thesis_code_json = parsed_data.get('thesis_code', '')
            section_json = parsed_data.get('section', '')
            context_json = parsed_data.get('ctx', '')

            # Original CSV columns to preserve
            base_row = {
                'thesis_code': row.get('thesis_code', thesis_code_json),
                'section': row.get('section', section_json),
                'sentence': row.get('sentence', ''),
                'context_before': row.get('context_before', ''),
                'context_after': row.get('context_after', ''),
                'context_from_json': context_json,
                'claude_metadiscourse_annotation': row.get('claude_metadiscourse_annotation', ''),
                'gemini_metadiscourse_annotation': row.get('gemini_metadiscourse_annotation', ''),
                'deepseek_metadiscourse_annotation': row.get('deepseek_metadiscourse_annotation', ''),
                'original_row_index': idx
            }

            # Check for expressions (1, 2, 3)
            for expr_num in range(1, 4):
                expr_key = f'expr_{expr_num}'
                analysis_key = f'analysis_{expr_num}'

                expression = parsed_data.get(expr_key, '')
                analysis = parsed_data.get(analysis_key, {})

                # Only create a row if the expression exists
                if expression and expression.strip():
                    new_row = base_row.copy()
                    new_row['expression_number'] = expr_num
                    new_row['expression'] = expression

                    # Extract and add flattened analysis fields
                    analysis_fields = extract_nested_fields(analysis)
                    new_row.update(analysis_fields)

                    rows.append(new_row)

        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse JSON at row {idx}: {e}")
        except Exception as e:
            print(f"Warning: Unexpected error at row {idx}: {e}")
            import traceback
            traceback.print_exc()

    # Create DataFrame from collected rows
    result_df = pd.DataFrame(rows)

    return result_df


# Columns for the SHORT version (essential classification fields only)
SHORT_COLUMNS = [
    # Identification
    'thesis_code',
    'section',
    'sentence',
    'expression_number',
    'expression',
    # Reflexivity
    'is_reflexive',
    'reflexivity_type',
    'reflexivity_reasoning',
    # Scope
    'scope_classification',
    'scope_reach',
    'scope_reasoning',
    # L1 Classification
    'L1_classification',
    'L1_reasoning',
    # L2 Classification
    'L2_classification',
    'L2_reasoning',
    # L3 Classification
    'L3_classification',
    'L3_borderline',
    'L3_reasoning',
    # L1 Borderline
    'L1_borderline_is',
    'L1_borderline_dominant',
    'L1_borderline_secondary',
    'L1_borderline_why',
    # L2 Borderline
    'L2_borderline_is',
    'L2_borderline_primary',
    'L2_borderline_secondary',
    'L2_borderline_tertiary',
    'L2_borderline_strength',
    'L2_borderline_why',
    # Justification
    'comprehensive_justification',
]


def process_csv_files(input_dir='result/optimized/without_boundary', output_dir='result/flattened-by-expression/without_boundary'):
    """
    Process all CSV files in the input directory and create flattened versions.
    Generates both LONG (all columns) and SHORT (essential columns) versions.

    Args:
        input_dir: directory containing CSV files to process
        output_dir: directory to save flattened CSV files
    """
    # Create output directories
    long_dir = os.path.join(output_dir, 'long')
    short_dir = os.path.join(output_dir, 'short')
    Path(long_dir).mkdir(parents=True, exist_ok=True)
    Path(short_dir).mkdir(parents=True, exist_ok=True)

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

            # Check if Optimized_final_decision column exists
            if 'Optimized_final_decision' not in df.columns:
                print(f"Warning: 'Optimized_final_decision' column not found in {csv_file}")
                continue

            # Flatten and unpivot the JSON data
            flattened_df = flatten_and_unpivot(df)

            if flattened_df.empty:
                print(f"Warning: No data extracted from {csv_file}")
                continue

            # Generate output filenames
            input_filename = os.path.basename(csv_file)
            base_name = input_filename.replace('.csv', '')

            # === LONG VERSION (all columns) ===
            long_filename = f"{base_name}_LONG.csv"
            long_path = os.path.join(long_dir, long_filename)

            flattened_df.to_csv(
                long_path,
                index=False,
                quoting=csv.QUOTE_ALL,
                escapechar='\\',
                doublequote=True
            )

            # === SHORT VERSION (essential columns only) ===
            # Filter to only include columns that exist in the dataframe
            available_short_cols = [col for col in SHORT_COLUMNS if col in flattened_df.columns]
            short_df = flattened_df[available_short_cols].copy()

            short_filename = f"{base_name}_SHORT.csv"
            short_path = os.path.join(short_dir, short_filename)

            short_df.to_csv(
                short_path,
                index=False,
                quoting=csv.QUOTE_ALL,
                escapechar='\\',
                doublequote=True
            )

            print(f"✓ Saved LONG version to: {long_path}")
            print(f"  Columns: {len(flattened_df.columns)}")
            print(f"✓ Saved SHORT version to: {short_path}")
            print(f"  Columns: {len(short_df.columns)}")
            print(f"  Original rows: {len(df)}")
            print(f"  Flattened rows (one per expression): {len(flattened_df)}")

            # Show expression count distribution
            if 'expression_number' in flattened_df.columns:
                expr_counts = flattened_df['expression_number'].value_counts().sort_index()
                print(f"  Expression distribution:")
                for expr_num, count in expr_counts.items():
                    print(f"    Expression {expr_num}: {count} rows")

        except Exception as e:
            print(f"✗ Error processing {csv_file}: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main function to run the flattening process."""
    print("Starting JSON flattening and unpivoting process...")
    print("=" * 60)
    print("This script will:")
    print("  1. Parse the Optimized_final_decision JSON column")
    print("  2. Create separate rows for each expression (expr_1, expr_2, expr_3)")
    print("  3. Flatten the corresponding analysis into columns")
    print("  4. Generate TWO output versions:")
    print("     - LONG: All columns (full detail)")
    print("     - SHORT: Essential classification columns only")
    print("=" * 60)

    # Default directories (adjust as needed)
    input_directory = 'result/optimized/without_boundary'
    output_directory = 'result/flattened-by-expression/without_boundary'

    # Check if input directory exists
    if not os.path.exists(input_directory):
        print(f"Error: Input directory '{input_directory}' does not exist.")
        print("Please ensure the directory path is correct.")
        return

    # Process the files
    process_csv_files(input_directory, output_directory)

    print("\n" + "=" * 60)
    print("JSON flattening and unpivoting process completed!")
    print(f"Output locations:")
    print(f"  LONG:  {output_directory}/long/")
    print(f"  SHORT: {output_directory}/short/")


if __name__ == "__main__":
    main()