import pandas as pd
import os
import openai
from dotenv import load_dotenv
from datetime import datetime
import json

# Load environment variables
load_dotenv()


class PromptImprover:
    def __init__(self):
        # Initialize OpenAI client
        openai.api_key = os.getenv('OPENAI_API_KEY')
        self.client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

        # Define paths
        self.discrepancy_file = "result/prompt_analysis/discrepancy/discrepancy_cases.csv"
        self.original_prompt_file = "src/annotation/prompt/internalexternal.txt"
        self.system_prompt_file = "src/annotation/prompt/promptimprovement.txt"
        self.output_dir = "result/prompt-analysis/final-result"

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

    def load_discrepancy_data(self):
        """Load and filter discrepancy dataset"""
        try:
            df = pd.read_csv(self.discrepancy_file)

            # Select only the required columns
            required_cols = [
                'sentence', 'expression', 'claude_role', 'gemini_role',
                'deepseek_role', 'discrepancy_cause', 'prompt_improvement_suggestions'
            ]

            # Filter columns that exist in the dataset
            existing_cols = [col for col in required_cols if col in df.columns]
            df_filtered = df[existing_cols].copy()

            print(f"Loaded {len(df_filtered)} discrepancy cases")
            print(f"Available columns: {existing_cols}")

            return df_filtered

        except Exception as e:
            print(f"Error loading discrepancy data: {e}")
            return None

    def load_original_prompt(self):
        """Load the original prompt to be improved"""
        try:
            with open(self.original_prompt_file, 'r', encoding='utf-8') as f:
                prompt = f.read()
            print(f"Loaded original prompt ({len(prompt)} characters)")
            return prompt
        except Exception as e:
            print(f"Error loading original prompt: {e}")
            return None

    def load_system_prompt(self):
        """Load the system prompt for GPT-5"""
        try:
            with open(self.system_prompt_file, 'r', encoding='utf-8') as f:
                system_prompt = f.read()
            print(f"Loaded system prompt ({len(system_prompt)} characters)")
            return system_prompt
        except Exception as e:
            print(f"Error loading system prompt: {e}")
            return None

    def prepare_analysis_data(self, df):
        """Prepare discrepancy data for analysis"""
        analysis_summary = {
            'total_discrepancies': len(df),
            'discrepancy_causes': df[
                'discrepancy_cause'].value_counts().to_dict() if 'discrepancy_cause' in df.columns else {},
            'improvement_suggestions': df[
                'prompt_improvement_suggestions'].dropna().tolist() if 'prompt_improvement_suggestions' in df.columns else [],
            'role_disagreements': []
        }

        # Analyze role disagreements
        if all(col in df.columns for col in ['claude_role', 'gemini_role', 'deepseek_role']):
            for idx, row in df.iterrows():
                roles = [row['claude_role'], row['gemini_role'], row['deepseek_role']]
                unique_roles = list(set([r for r in roles if pd.notna(r)]))

                if len(unique_roles) > 1:
                    analysis_summary['role_disagreements'].append({
                        'sentence': row['sentence'][:100] + "..." if len(str(row['sentence'])) > 100 else str(
                            row['sentence']),
                        'expression': str(row['expression']),
                        'roles': {
                            'claude': str(row['claude_role']),
                            'gemini': str(row['gemini_role']),
                            'deepseek': str(row['deepseek_role'])
                        }
                    })

        return analysis_summary

    def call_gpt_for_report_generation(self, system_prompt, original_prompt, analysis_data):
        """Call GPT to generate analysis report only - FOCUSED ON REPORT"""
        try:
            user_message = f"""
Please analyze the following annotation prompt based on the discrepancy analysis data provided and generate a comprehensive analysis report.

ORIGINAL PROMPT TO ANALYZE:
{original_prompt}

DISCREPANCY ANALYSIS DATA:
Total discrepancies found: {analysis_data['total_discrepancies']}

Discrepancy causes breakdown:
{json.dumps(analysis_data['discrepancy_causes'], indent=2)}

Improvement suggestions from the data:
{json.dumps(analysis_data['improvement_suggestions'][:15], indent=2)}

Sample role disagreement cases:
{json.dumps(analysis_data['role_disagreements'][:8], indent=2)}

Your task is to generate a detailed analysis report that identifies:
1. Specific weaknesses and ambiguities in the current prompt
2. Patterns in the discrepancy data that indicate prompt issues
3. Root causes of annotation inconsistencies
4. Concrete, actionable recommendations for improvement
5. Priority areas that need immediate attention
6. Specific examples of how instructions should be clarified or rewritten

Focus ONLY on analysis and recommendations. Do NOT rewrite the prompt - just analyze it and provide improvement guidance.
"""

            response = self.client.chat.completions.create(
                model="gpt-5",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=3000,
                temperature=0.1
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"Error calling GPT for report generation: {e}")
            return None

    def call_gpt_for_prompt_improvement(self, original_prompt, analysis_report):
        """Call GPT to improve prompt using the generated report - FOCUSED ON PROMPT IMPROVEMENT"""
        try:
            user_message = f"""
You are an expert prompt engineer. Use the analysis report below to rewrite and improve the annotation prompt.

ORIGINAL PROMPT TO IMPROVE:
{original_prompt}

ANALYSIS REPORT WITH RECOMMENDATIONS:
{analysis_report}

Your task:
1. Carefully study the analysis report and understand all identified issues
2. Rewrite the original prompt to address EVERY issue and recommendation mentioned in the report
3. Make the prompt more specific, clear, and consistent
4. Add examples and clarifications where the report suggests
5. Ensure the new prompt will reduce the types of discrepancies identified

Requirements:
- Address all weaknesses identified in the analysis report
- Make instructions more specific and unambiguous  
- Add examples for edge cases mentioned in the report
- Improve consistency guidelines based on the report's recommendations
- Make the prompt more robust against the annotation disagreements found

Provide ONLY the improved prompt text, without any additional commentary or explanation.
"""

            response = self.client.chat.completions.create(
                model="gpt-5",
                messages=[
                    {"role": "system",
                     "content": "You are an expert prompt engineer. Your job is to rewrite prompts to be clear, specific, and effective based on analysis feedback. Focus on creating prompts that will reduce annotation discrepancies."},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=2500,
                temperature=0.1
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"Error calling GPT for prompt improvement: {e}")
            return None

    def save_results(self, analysis_report, improved_prompt):
        """Save both the analysis report and improved prompt"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Save analysis report
            report_file = os.path.join(self.output_dir, f"analysis_report_{timestamp}.md")
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(f"# Discrepancy Analysis Report\n\n")
                f.write(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"**Original Prompt File:** {self.original_prompt_file}\n\n")
                f.write(f"**Discrepancy Data Source:** {self.discrepancy_file}\n\n")
                f.write("## Analysis Results\n\n")
                f.write(analysis_report)

            # Save improved prompt
            improved_prompt_file = os.path.join(self.output_dir, f"improved_prompt_{timestamp}.txt")
            with open(improved_prompt_file, 'w', encoding='utf-8') as f:
                f.write(f"# IMPROVED ANNOTATION PROMPT\n")
                f.write(f"# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# Based on analysis of discrepancy data\n")
                f.write(f"# Original prompt: {self.original_prompt_file}\n\n")
                f.write(improved_prompt)

            print(f"✅ Analysis report saved to: {report_file}")
            print(f"✅ Improved prompt saved to: {improved_prompt_file}")

            return report_file, improved_prompt_file

        except Exception as e:
            print(f"Error saving results: {e}")
            return None, None

    def run_improvement_process(self):
        """Main method to run the entire improvement process with two separate API calls"""
        print("🚀 Starting two-step prompt improvement process...")

        # Step 1: Load all required data
        print("\n📊 Loading discrepancy data...")
        df = self.load_discrepancy_data()
        if df is None:
            return False

        print("\n📄 Loading original prompt...")
        original_prompt = self.load_original_prompt()
        if original_prompt is None:
            return False

        print("\n🤖 Loading system prompt...")
        system_prompt = self.load_system_prompt()
        if system_prompt is None:
            return False

        print("\n🔍 Analyzing discrepancy patterns...")
        analysis_data = self.prepare_analysis_data(df)

        # Step 2: First API call - Generate analysis report
        print("\n🧠 Step 1: Calling GPT to generate analysis report...")
        analysis_report = self.call_gpt_for_report_generation(system_prompt, original_prompt, analysis_data)
        if analysis_report is None:
            return False

        print("✅ Analysis report generated successfully")

        # Step 3: Second API call - Use report to improve prompt
        print("\n🔄 Step 2: Calling GPT to improve prompt using the analysis report...")
        improved_prompt = self.call_gpt_for_prompt_improvement(original_prompt, analysis_report)
        if improved_prompt is None:
            return False

        print("✅ Improved prompt generated successfully")

        # Step 4: Save both results
        print("\n💾 Saving results...")
        report_file, prompt_file = self.save_results(analysis_report, improved_prompt)

        if report_file and prompt_file:
            print("\n✅ Two-step prompt improvement process completed successfully!")
            print(f"📁 Results saved in: {self.output_dir}")
            print(f"📊 Analysis Report: {os.path.basename(report_file)}")
            print(f"📝 Improved Prompt: {os.path.basename(prompt_file)}")
            return True
        else:
            print("\n❌ Failed to save results")
            return False


def main():
    """Main function"""
    improver = PromptImprover()
    success = improver.run_improvement_process()

    if success:
        print("\n🎉 Process completed successfully!")
    else:
        print("\n⚠️  Process completed with errors. Please check the logs above.")


if __name__ == "__main__":
    main()