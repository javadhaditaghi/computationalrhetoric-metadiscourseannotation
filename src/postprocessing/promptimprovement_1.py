import pandas as pd
import os
import openai
from dotenv import load_dotenv
from datetime import datetime
import json

# Load environment variables
load_dotenv()


class PromptImprovementPipeline:
    def __init__(self):
        # Initialize OpenAI client
        openai.api_key = os.getenv('OPENAI_API_KEY')
        self.client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

        # Define paths
        self.discrepancy_file = "result/prompt-analysis/discrepancy/discrepancy_cases.csv"
        self.original_prompt_file = "src/annotation/prompt/internalexternal.txt"
        self.system_prompt_file = "src/annotation/prompt/promptimprovement.txt"
        self.output_dir = "result/prompt-analysis/final-result"

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Pipeline state
        self.analysis_report_file = None
        self.timestamp = None

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

            print(f"✅ Loaded {len(df_filtered)} discrepancy cases")
            print(f"📊 Available columns: {existing_cols}")

            return df_filtered

        except Exception as e:
            print(f"❌ Error loading discrepancy data: {e}")
            return None

    def load_original_prompt(self):
        """Load the original prompt to be improved"""
        try:
            with open(self.original_prompt_file, 'r', encoding='utf-8') as f:
                prompt = f.read()
            print(f"✅ Loaded original prompt ({len(prompt)} characters)")
            return prompt
        except Exception as e:
            print(f"❌ Error loading original prompt: {e}")
            return None

    def load_system_prompt(self):
        """Load the system prompt for GPT"""
        try:
            with open(self.system_prompt_file, 'r', encoding='utf-8') as f:
                system_prompt = f.read()
            print(f"✅ Loaded system prompt ({len(system_prompt)} characters)")
            return system_prompt
        except Exception as e:
            print(f"❌ Error loading system prompt: {e}")
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

    # PIPELINE STEP 1: GENERATE ANALYSIS REPORT
    def step1_generate_analysis_report(self):
        """Pipeline Step 1: Generate comprehensive analysis report"""
        print("\n" + "=" * 60)
        print("🔍 PIPELINE STEP 1: GENERATING ANALYSIS REPORT")
        print("=" * 60)

        # Load all required data
        df = self.load_discrepancy_data()
        if df is None:
            return False

        original_prompt = self.load_original_prompt()
        if original_prompt is None:
            return False

        system_prompt = self.load_system_prompt()
        if system_prompt is None:
            return False

        # Prepare analysis data
        print("\n🔍 Analyzing discrepancy patterns...")
        analysis_data = self.prepare_analysis_data(df)

        # Call GPT to generate analysis report
        print("\n🧠 Calling GPT to generate analysis report...")
        analysis_report = self._call_gpt_for_analysis(system_prompt, original_prompt, analysis_data)
        if analysis_report is None:
            return False

        # Save the analysis report
        print("\n💾 Saving analysis report...")
        success = self._save_analysis_report(analysis_report)

        if success:
            print(f"\n✅ STEP 1 COMPLETED: Analysis report saved!")
            print(f"📄 Report file: {self.analysis_report_file}")
            return True
        else:
            print("\n❌ STEP 1 FAILED: Could not save analysis report")
            return False

    def _call_gpt_for_analysis(self, system_prompt, original_prompt, analysis_data):
        """Generate analysis report using GPT"""
        try:
            user_message = f"""
Analyze the following annotation prompt based on the discrepancy data provided. Generate a comprehensive analysis report.

ORIGINAL PROMPT TO ANALYZE:
{original_prompt}

DISCREPANCY DATA SUMMARY:
- Total discrepancies found: {analysis_data['total_discrepancies']}

Discrepancy causes breakdown:
{json.dumps(analysis_data['discrepancy_causes'], indent=2)}

Improvement suggestions from data:
{json.dumps(analysis_data['improvement_suggestions'][:15], indent=2)}

Sample role disagreement cases:
{json.dumps(analysis_data['role_disagreements'][:10], indent=2)}

Generate a detailed analysis report that includes:
1. Identification of specific weaknesses in the current prompt
2. Analysis of patterns in the discrepancy data
3. Root cause analysis of annotation inconsistencies
4. Concrete, actionable recommendations for improvement
5. Priority areas that need immediate attention
6. Specific examples of how instructions should be clarified

Focus on actionable insights that can guide prompt rewriting.
"""

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                max_completion_tokens=3500,
                temperature=0
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"❌ Error calling GPT for analysis: {e}")
            return None

    def _save_analysis_report(self, analysis_report):
        """Save the analysis report to file"""
        try:
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            self.analysis_report_file = os.path.join(self.output_dir, f"analysis_report_{self.timestamp}.md")
            with open(self.analysis_report_file, 'w', encoding='utf-8') as f:
                f.write(f"# Discrepancy Analysis Report\n\n")
                f.write(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"**Original Prompt File:** {self.original_prompt_file}\n\n")
                f.write(f"**Discrepancy Data Source:** {self.discrepancy_file}\n\n")
                f.write("## Analysis Results\n\n")
                f.write(analysis_report)

            return True

        except Exception as e:
            print(f"❌ Error saving analysis report: {e}")
            return False

    # PIPELINE STEP 2: USE REPORT TO IMPROVE PROMPT
    def step2_improve_prompt_using_report(self):
        """Pipeline Step 2: Load saved report and use it to improve prompt"""
        print("\n" + "=" * 60)
        print("🔄 PIPELINE STEP 2: IMPROVING PROMPT USING REPORT")
        print("=" * 60)

        # Check if we have a report file from step 1
        if not self.analysis_report_file or not os.path.exists(self.analysis_report_file):
            print("❌ No analysis report found. Please run step1_generate_analysis_report first.")
            return False

        # Load the saved analysis report
        print(f"\n📖 Loading analysis report from: {self.analysis_report_file}")
        analysis_report = self._load_analysis_report()
        if analysis_report is None:
            return False

        # Load original prompt
        original_prompt = self.load_original_prompt()
        if original_prompt is None:
            return False

        # Call GPT to improve prompt based on the report
        print("\n🧠 Calling GPT to improve prompt based on analysis report...")
        improved_prompt = self._call_gpt_for_prompt_improvement(original_prompt, analysis_report)
        if improved_prompt is None:
            return False

        # Save the improved prompt
        print("\n💾 Saving improved prompt...")
        success = self._save_improved_prompt(improved_prompt)

        if success:
            print(f"\n✅ STEP 2 COMPLETED: Improved prompt saved!")
            print(f"📝 Improved prompt file: improved_prompt_{self.timestamp}.txt")
            return True
        else:
            print("\n❌ STEP 2 FAILED: Could not save improved prompt")
            return False

    def _load_analysis_report(self):
        """Load the previously saved analysis report"""
        try:
            with open(self.analysis_report_file, 'r', encoding='utf-8') as f:
                report_content = f.read()
            print(f"✅ Loaded analysis report ({len(report_content)} characters)")
            return report_content
        except Exception as e:
            print(f"❌ Error loading analysis report: {e}")
            return None

    def _call_gpt_for_prompt_improvement(self, original_prompt, analysis_report):
        """Use analysis report to improve the prompt"""
        try:
            user_message = f"""
You are a prompt engineering expert. Use the analysis report below to completely rewrite and improve the annotation prompt.

ORIGINAL PROMPT TO IMPROVE:
{original_prompt}

ANALYSIS REPORT WITH IMPROVEMENT RECOMMENDATIONS:
{analysis_report}

Your task:
1. Carefully read the analysis report and understand all identified issues
2. Rewrite the original prompt to address EVERY issue mentioned in the report
3. Make the prompt more specific, clear, and consistent
4. Add examples and clarifications where the report suggests
5. Ensure the new prompt will reduce the types of discrepancies identified

Requirements:
- Address all weaknesses identified in the analysis
- Make instructions more specific and unambiguous
- Add examples for edge cases mentioned in the report
- Improve consistency guidelines
- Make the prompt more robust against annotation disagreements

Provide ONLY the improved prompt text.
"""

            response = self.client.chat.completions.create(
                model="gpt-5",
                messages=[
                    {"role": "system",
                     "content": "You are an expert prompt engineer. Rewrite prompts to be clear, specific, and effective based on detailed analysis feedback. Focus on creating prompts that will reduce annotation discrepancies."},
                    {"role": "user", "content": user_message}
                ]
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"❌ Error calling GPT for prompt improvement: {e}")
            return None

    def _save_improved_prompt(self, improved_prompt):
        """Save the improved prompt"""
        try:
            improved_prompt_file = os.path.join(self.output_dir, f"improved_prompt_{self.timestamp}.txt")
            with open(improved_prompt_file, 'w', encoding='utf-8') as f:
                f.write(f"# IMPROVED ANNOTATION PROMPT\n")
                f.write(f"# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# Based on analysis report: analysis_report_{self.timestamp}.md\n")
                f.write(f"# Original prompt: {self.original_prompt_file}\n\n")
                f.write(improved_prompt)

            return True

        except Exception as e:
            print(f"❌ Error saving improved prompt: {e}")
            return False

    # PIPELINE RUNNER
    def run_complete_pipeline(self):
        """Run the complete pipeline: Step 1 → Step 2"""
        print("\n🚀 STARTING COMPLETE PROMPT IMPROVEMENT PIPELINE")
        print("=" * 80)

        # Step 1: Generate analysis report
        step1_success = self.step1_generate_analysis_report()
        if not step1_success:
            print("\n❌ PIPELINE FAILED AT STEP 1")
            return False

        # Step 2: Use report to improve prompt
        step2_success = self.step2_improve_prompt_using_report()
        if not step2_success:
            print("\n❌ PIPELINE FAILED AT STEP 2")
            return False

        # Pipeline completed successfully
        print("\n" + "=" * 80)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"📁 Output directory: {self.output_dir}")
        print(f"📊 Analysis report: analysis_report_{self.timestamp}.md")
        print(f"📝 Improved prompt: improved_prompt_{self.timestamp}.txt")
        print("\n✅ You can now review both files and test the improved prompt!")

        return True


def main():
    """Main function with options to run individual steps or complete pipeline"""
    pipeline = PromptImprovementPipeline()

    print("🔧 Prompt Improvement Pipeline")
    print("Choose an option:")
    print("1. Run complete pipeline (Step 1 → Step 2)")
    print("2. Run Step 1 only (Generate analysis report)")
    print("3. Run Step 2 only (Improve prompt using existing report)")

    choice = input("\nEnter your choice (1, 2, or 3): ").strip()

    if choice == "1":
        success = pipeline.run_complete_pipeline()
    elif choice == "2":
        success = pipeline.step1_generate_analysis_report()
    elif choice == "3":
        # For step 2, we need to ask for the report file
        report_file = input("Enter the path to the analysis report file: ").strip()
        if os.path.exists(report_file):
            pipeline.analysis_report_file = report_file
            pipeline.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            success = pipeline.step2_improve_prompt_using_report()
        else:
            print(f"❌ Report file not found: {report_file}")
            success = False
    else:
        print("❌ Invalid choice")
        success = False

    if success:
        print("\n🎉 Process completed successfully!")
    else:
        print("\n⚠️  Process completed with errors.")


if __name__ == "__main__":
    main()