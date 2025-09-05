import pandas as pd
import json
import re
from collections import Counter


def extract_role(annotation):
    """
    Extract role from annotation (handles both JSON and plain text)
    """
    if not annotation or pd.isna(annotation):
        return None

    annotation = str(annotation).strip()

    try:
        # Try to parse as JSON first
        parsed = json.loads(annotation)
        return parsed.get('role') or parsed.get('Role')
    except (json.JSONDecodeError, ValueError):
        # If not JSON, check if it's a plain text role
        cleaned_annotation = annotation.lower().strip()
        if cleaned_annotation in ['metadiscourse', 'proposition']:
            return cleaned_annotation.capitalize()

        # Try to extract role from text patterns
        role_match = re.search(r'role[\'\":\s]*[\'\"]?([^\'\"",\s}]+)', annotation, re.IGNORECASE)
        if role_match:
            return role_match.group(1)

        return annotation.strip()


def extract_from_json_field(json_string, field_name):
    """
    Robust extraction of specific field from JSON string
    """
    if not json_string or pd.isna(json_string):
        return None

    try:
        # Convert to string and clean up
        json_str = str(json_string).strip()

        # Handle cases where the JSON might be malformed
        if not json_str.startswith('{'):
            return None

        # Parse JSON
        data = json.loads(json_str)

        # Return the specific field
        return data.get(field_name, None)

    except (json.JSONDecodeError, ValueError, TypeError) as e:
        # If JSON parsing fails, try regex extraction as fallback
        try:
            # Pattern to match the field in JSON-like text
            pattern = f'"{field_name}":\s*"([^"]*)"'
            match = re.search(pattern, str(json_string))
            if match:
                return match.group(1)

            # Alternative pattern without quotes around value
            pattern = f'"{field_name}":\s*([^,}}]+)'
            match = re.search(pattern, str(json_string))
            if match:
                return match.group(1).strip().strip('"')

        except Exception:
            pass

        return None


def debug_json_extraction(csv_path='result/optimized/optimized_annotations.csv', num_samples=5):
    """
    Debug function to check JSON extraction from Optimized_final_decision column
    """
    try:
        print(f"Loading CSV file: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")

        # Find the Optimized_final_decision column (case insensitive)
        optimized_col = None
        for col in df.columns:
            if 'optimized' in col.lower() and ('final' in col.lower() or 'decision' in col.lower()):
                optimized_col = col
                break

        if optimized_col is None:
            print("❌ No Optimized_final_decision column found!")
            potential_cols = [col for col in df.columns if 'optimized' in col.lower() or 'decision' in col.lower()]
            if potential_cols:
                print(f"Potential columns: {potential_cols}")
            return False

        print(f"✅ Found column: '{optimized_col}'")
        print(f"Non-null values in {optimized_col}: {df[optimized_col].notna().sum()}/{len(df)}")

        # Sample a few rows and try to parse JSON
        success_count = 0
        for i in range(min(num_samples, len(df))):
            print(f"\n--- Sample {i + 1} ---")
            sample_json = df.iloc[i][optimized_col]

            if pd.isna(sample_json):
                print("❌ NULL value")
                continue

            print(f"Raw content length: {len(str(sample_json))} characters")
            print(f"First 200 chars: {str(sample_json)[:200]}...")

            try:
                parsed = json.loads(str(sample_json))
                success_count += 1
                print(f"✅ JSON parsed successfully")
                print(f"Available keys: {list(parsed.keys())}")

                # Check for our target fields
                discrepancy_cause = parsed.get('discrepancy_cause')
                prompt_suggestions = parsed.get('prompt_improvement_suggestions')

                print(f"discrepancy_cause: {'✅ FOUND' if discrepancy_cause else '❌ NOT FOUND'}")
                if discrepancy_cause:
                    print(f"  Content: {str(discrepancy_cause)[:100]}...")

                print(f"prompt_improvement_suggestions: {'✅ FOUND' if prompt_suggestions else '❌ NOT FOUND'}")
                if prompt_suggestions:
                    print(f"  Content: {str(prompt_suggestions)[:100]}...")

            except Exception as e:
                print(f"❌ JSON parsing error: {e}")
                # Try to extract using regex
                discrepancy_cause = extract_from_json_field(sample_json, 'discrepancy_cause')
                prompt_suggestions = extract_from_json_field(sample_json, 'prompt_improvement_suggestions')
                print(f"Regex extraction - discrepancy_cause: {'✅' if discrepancy_cause else '❌'}")
                print(f"Regex extraction - prompt_improvement_suggestions: {'✅' if prompt_suggestions else '❌'}")

        print(f"\n✅ Successfully parsed {success_count}/{num_samples} samples")
        return True

    except FileNotFoundError:
        print(f"❌ File not found: {csv_path}")
        return False
    except Exception as e:
        print(f"❌ Error in debug function: {e}")
        return False


def analyze_discrepancies(csv_path='result/optimized/optimized_annotations.csv'):
    """
    Analyze discrepancies between three annotators
    """
    try:
        # Read the CSV file
        print(f"Reading CSV file: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"Total rows: {len(df)}")

        # Check if required columns exist
        required_columns = ['claude_metadiscourse_annotation', 'gemini_metadiscourse_annotation',
                            'deepseek_metadiscourse_annotation']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            print(f"❌ Missing columns: {missing_columns}")
            print(f"Available columns: {list(df.columns)}")
            return None

        # Find the Optimized_final_decision column
        optimized_col = None
        for col in df.columns:
            if 'optimized' in col.lower() and ('final' in col.lower() or 'decision' in col.lower()):
                optimized_col = col
                break

        if optimized_col is None:
            print("❌ Warning: No Optimized_final_decision column found!")
            print("Proceeding without prompt improvement analysis...")
        else:
            print(f"✅ Found optimized decision column: '{optimized_col}'")

        # Extract roles from each annotation
        print("Extracting roles from annotations...")
        df['claude_role'] = df['claude_metadiscourse_annotation'].apply(extract_role)
        df['gemini_role'] = df['gemini_metadiscourse_annotation'].apply(extract_role)
        df['deepseek_role'] = df['deepseek_metadiscourse_annotation'].apply(extract_role)

        # Check for discrepancies
        def check_discrepancy(row):
            roles = [row['claude_role'], row['gemini_role'], row['deepseek_role']]
            roles = [role for role in roles if role is not None]
            unique_roles = list(set(roles))
            return len(unique_roles) > 1

        df['is_discrepancy'] = df.apply(check_discrepancy, axis=1)

        # Create unique_roles column
        def get_unique_roles(row):
            roles = [row['claude_role'], row['gemini_role'], row['deepseek_role']]
            roles = [role for role in roles if role is not None]
            unique_roles = list(set(roles))
            return ', '.join(unique_roles)

        df['unique_roles'] = df.apply(get_unique_roles, axis=1)

        # Extract prompt_improvement_suggestions and discrepancy_cause if column exists
        if optimized_col:
            print("Extracting prompt improvement suggestions and discrepancy causes...")

            df['discrepancy_cause'] = df[optimized_col].apply(
                lambda x: extract_from_json_field(x, 'discrepancy_cause')
            )
            df['prompt_improvement_suggestions'] = df[optimized_col].apply(
                lambda x: extract_from_json_field(x, 'prompt_improvement_suggestions')
            )

            # Report extraction success
            cause_count = df['discrepancy_cause'].notna().sum()
            suggestion_count = df['prompt_improvement_suggestions'].notna().sum()
            print(f"✅ Successfully extracted {cause_count} discrepancy causes")
            print(f"✅ Successfully extracted {suggestion_count} prompt improvement suggestions")

        # Split into regular cases and discrepancies
        discrepancies = df[df['is_discrepancy'] == True].copy()
        regular_cases = df[df['is_discrepancy'] == False].copy()

        # Print summary statistics
        print("\n" + "=" * 50)
        print("SUMMARY STATISTICS")
        print("=" * 50)
        print(f"Total annotations: {len(df)}")
        print(f"Agreement cases: {len(regular_cases)} ({len(regular_cases) / len(df) * 100:.2f}%)")
        print(f"Discrepancy cases: {len(discrepancies)} ({len(discrepancies) / len(df) * 100:.2f}%)")

        # Analyze prompt improvement suggestions and discrepancy causes
        if optimized_col and (cause_count > 0 or suggestion_count > 0):
            print(f"\n" + "=" * 50)
            print("PROMPT IMPROVEMENT ANALYSIS")
            print("=" * 50)

            print(f"Cases with discrepancy causes: {cause_count}")
            print(f"Cases with prompt improvement suggestions: {suggestion_count}")

            # Show most common discrepancy causes
            if cause_count > 0:
                cause_counts = df['discrepancy_cause'].dropna().value_counts()
                print(f"\n📊 Most common discrepancy causes:")
                for i, (cause, count) in enumerate(cause_counts.head(10).items(), 1):
                    print(f"  {i}. [{count} cases] {cause[:150]}{'...' if len(cause) > 150 else ''}")

            # Show most common prompt improvement suggestions
            if suggestion_count > 0:
                suggestion_counts = df['prompt_improvement_suggestions'].dropna().value_counts()
                print(f"\n💡 Most common prompt improvement suggestions:")
                for i, (suggestion, count) in enumerate(suggestion_counts.head(10).items(), 1):
                    print(f"  {i}. [{count} cases] {suggestion[:150]}{'...' if len(suggestion) > 150 else ''}")

        # Show sample discrepancies
        if len(discrepancies) > 0:
            print("\n" + "=" * 50)
            print("SAMPLE DISCREPANCIES")
            print("=" * 50)

            for i, (idx, row) in enumerate(discrepancies.head(3).iterrows()):
                print(f"\n📝 Example {i + 1}:")
                print(f"  Expression: \"{row.get('expression', 'N/A')}\"")
                print(f"  Claude role: {row['claude_role']}")
                print(f"  Gemini role: {row['gemini_role']}")
                print(f"  DeepSeek role: {row['deepseek_role']}")

                if optimized_col:
                    if pd.notna(row.get('discrepancy_cause')):
                        print(
                            f"  🔍 Discrepancy Cause: {str(row['discrepancy_cause'])[:200]}{'...' if len(str(row['discrepancy_cause'])) > 200 else ''}")
                    if pd.notna(row.get('prompt_improvement_suggestions')):
                        print(
                            f"  💡 Improvement Suggestion: {str(row['prompt_improvement_suggestions'])[:200]}{'...' if len(str(row['prompt_improvement_suggestions'])) > 200 else ''}")

                if 'sentence' in row and pd.notna(row['sentence']):
                    sentence = str(row['sentence'])
                    print(f"  📄 Sentence: \"{sentence[:100]}{'...' if len(sentence) > 100 else ''}\"")

        # Role distribution
        all_roles = []
        for col in ['claude_role', 'gemini_role', 'deepseek_role']:
            all_roles.extend([role for role in df[col] if role is not None])

        role_counts = Counter(all_roles)

        print("\n" + "=" * 50)
        print("ROLE DISTRIBUTION")
        print("=" * 50)
        for role, count in role_counts.most_common():
            print(f"{role}: {count}")

        # Save CSV files
        discrepancy_filename = 'discrepancy_cases.csv'
        regular_filename = 'regular_cases.csv'

        discrepancies.to_csv(discrepancy_filename, index=False)
        regular_cases.to_csv(regular_filename, index=False)

        print(f"\n" + "=" * 50)
        print("FILES SAVED")
        print("=" * 50)
        print(f"✅ Discrepancy cases saved to: {discrepancy_filename}")
        print(f"✅ Regular cases saved to: {regular_filename}")

        # Detailed discrepancy analysis
        if len(discrepancies) > 0:
            print(f"\n" + "=" * 50)
            print("DISCREPANCY PATTERNS")
            print("=" * 50)

            # Most common discrepancy patterns
            discrepancy_patterns = discrepancies['unique_roles'].value_counts()
            print("📊 Most common disagreement patterns:")
            for pattern, count in discrepancy_patterns.head(5).items():
                print(f"  {pattern}: {count} cases")

            # Annotator agreement rates
            claude_gemini_agree = sum(discrepancies['claude_role'] == discrepancies['gemini_role'])
            claude_deepseek_agree = sum(discrepancies['claude_role'] == discrepancies['deepseek_role'])
            gemini_deepseek_agree = sum(discrepancies['gemini_role'] == discrepancies['deepseek_role'])

            print(f"\n🤝 Pairwise agreement in discrepancy cases:")
            print(
                f"  Claude-Gemini: {claude_gemini_agree}/{len(discrepancies)} ({claude_gemini_agree / len(discrepancies) * 100:.1f}%)")
            print(
                f"  Claude-DeepSeek: {claude_deepseek_agree}/{len(discrepancies)} ({claude_deepseek_agree / len(discrepancies) * 100:.1f}%)")
            print(
                f"  Gemini-DeepSeek: {gemini_deepseek_agree}/{len(discrepancies)} ({gemini_deepseek_agree / len(discrepancies) * 100:.1f}%)")

        return {
            'total_cases': len(df),
            'regular_cases': len(regular_cases),
            'discrepancy_cases': len(discrepancies),
            'discrepancy_df': discrepancies,
            'regular_df': regular_cases,
            'role_distribution': role_counts,
            'discrepancy_causes': df['discrepancy_cause'].dropna().tolist() if optimized_col else [],
            'prompt_suggestions': df['prompt_improvement_suggestions'].dropna().tolist() if optimized_col else []
        }

    except FileNotFoundError:
        print(f"❌ Error: File not found at {csv_path}")
        print("Please check the file path and try again.")
        return None
    except Exception as e:
        print(f"❌ Error processing file: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def analyze_prompt_improvements(result_dict):
    """
    Detailed analysis of prompt improvement suggestions
    """
    if not result_dict or not result_dict.get('prompt_suggestions'):
        print("No prompt improvement data available")
        return

    suggestions = result_dict['prompt_suggestions']
    causes = result_dict['discrepancy_causes']

    print("\n" + "=" * 60)
    print("DETAILED PROMPT IMPROVEMENT ANALYSIS")
    print("=" * 60)

    print(f"📊 Total unique improvement suggestions: {len(set(suggestions))}")
    print(f"📊 Total unique discrepancy causes: {len(set(causes))}")

    # Analyze common themes in improvement suggestions
    common_themes = {}
    for suggestion in suggestions:
        if 'guideline' in suggestion.lower():
            common_themes['Guidelines'] = common_themes.get('Guidelines', 0) + 1
        if 'example' in suggestion.lower():
            common_themes['Examples'] = common_themes.get('Examples', 0) + 1
        if 'distinction' in suggestion.lower() or 'distinguish' in suggestion.lower():
            common_themes['Distinctions'] = common_themes.get('Distinctions', 0) + 1
        if 'context' in suggestion.lower():
            common_themes['Context'] = common_themes.get('Context', 0) + 1
        if 'definition' in suggestion.lower() or 'define' in suggestion.lower():
            common_themes['Definitions'] = common_themes.get('Definitions', 0) + 1

    if common_themes:
        print(f"\n🎯 Common improvement themes:")
        for theme, count in sorted(common_themes.items(), key=lambda x: x[1], reverse=True):
            print(f"  {theme}: {count} suggestions")


def main():
    """
    Main function to run the analysis
    """
    print("🔍 Metadiscourse Annotation Discrepancy Analyzer")
    print("=" * 60)

    # First run debug to check JSON structure
    print("🔍 Running debug check on JSON extraction...")
    debug_success = debug_json_extraction()

    if not debug_success:
        print("❌ Debug check failed. Please fix the issues above before proceeding.")
        return None

    print("\n" + "=" * 60)
    print("🚀 Running main analysis...")
    print("=" * 60)

    # Run the main analysis
    result = analyze_discrepancies()

    if result:
        print(f"\n✅ Analysis complete!")

        # Run additional prompt improvement analysis
        if result.get('prompt_suggestions') or result.get('discrepancy_causes'):
            analyze_prompt_improvements(result)

        print(f"\n📁 Next steps:")
        print(f"1. Check the generated CSV files for detailed data")
        print(f"2. Review the prompt improvement suggestions above")
        print(f"3. Use the discrepancy patterns to improve your annotator prompts")

        return result
    else:
        print("❌ Analysis failed. Please check your file path and data format.")
        return None


if __name__ == "__main__":
    # Run the analysis
    result = main()