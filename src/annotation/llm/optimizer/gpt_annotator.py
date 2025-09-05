import pandas as pd
import json
import os
import sys
import re
from typing import Dict, List, Tuple, Optional
from openai import OpenAI
from collections import Counter
import logging

# Add the project root to the path to import config
# From src/annotation/llm/optimizer/ we need to go up 4 levels to reach project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
sys.path.insert(0, project_root)

from util.config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnnotationOptimizer:
    def __init__(self, openai_api_key: str, model: str = "gpt-5",
                 context_words_before: int = 10, context_words_after: int = 10):
        """
        Initialize the Annotation Optimizer

        Args:
            openai_api_key: OpenAI API key
            model: GPT model to use for optimization
            context_words_before: Number of words to include from context_before column
            context_words_after: Number of words to include from context_after column
        """
        self.client = OpenAI(api_key=openai_api_key)
        self.model = model
        self.context_words_before = context_words_before
        self.context_words_after = context_words_after
        self.optimization_prompt = self._load_optimization_prompt()

    def _load_optimization_prompt(self) -> str:
        """Load the optimization prompt from file"""
        # Construct path relative to project root
        prompt_path = os.path.join(project_root, Config.OPTIMIZATION_PROMPT_PATH)

        if not os.path.exists(prompt_path):
            # Get the absolute path to show in error message
            absolute_prompt_path = os.path.abspath(prompt_path)
            expected_directory = os.path.dirname(absolute_prompt_path)

            raise FileNotFoundError(
                f"Prompt is not inside the correct directory.\n"
                f"Expected directory: {expected_directory}\n"
                f"Expected file path: {absolute_prompt_path}\n"
                f"Please ensure the prompt file exists in the correct directory."
            )

        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            absolute_prompt_path = os.path.abspath(prompt_path)
            raise IOError(f"Error reading prompt file from {absolute_prompt_path}: {e}")

    def _extract_limited_context(self, context_text: str, word_limit: int) -> str:
        """
        Extract limited number of words from context text

        Args:
            context_text: The context text to limit
            word_limit: Maximum number of words to extract

        Returns:
            Limited context string
        """
        if pd.isna(context_text) or not context_text:
            return ""

        words = str(context_text).strip().split()
        limited_words = words[:word_limit] if len(words) > word_limit else words
        return " ".join(limited_words)

    def load_csv_files(self, base_path: str = None) -> Dict[str, pd.DataFrame]:
        """
        Load CSV files from the three model directories

        Args:
            base_path: Base directory containing the model subdirectories (relative to project root)

        Returns:
            Dictionary with model names as keys and DataFrames as values
        """
        if base_path is None:
            base_path = Config.BASE_DATA_PATH

        # Ensure path is relative to project root
        full_base_path = os.path.join(project_root, base_path)

        models = Config.MODEL_DIRECTORIES
        dataframes = {}

        for model in models:
            model_path = os.path.join(full_base_path, model)

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model directory not found: {model_path}")

            csv_files = [f for f in os.listdir(model_path) if f.endswith('.csv')]

            if not csv_files:
                raise FileNotFoundError(f"No CSV files found in {model_path}")

            # Assuming one CSV file per model directory, take the first one
            csv_file = csv_files[0]
            file_path = os.path.join(model_path, csv_file)

            df = pd.read_csv(file_path)
            dataframes[model] = df
            logger.info(f"Loaded {len(df)} rows from {file_path}")

        return dataframes

    def parse_json_annotation(self, json_str: str) -> Optional[Dict]:
        """
        Parse JSON annotation string, handling various markdown formats

        Args:
            json_str: JSON string from metadiscourse_annotation column

        Returns:
            Parsed JSON dictionary or None if parsing fails
        """
        try:
            if pd.isna(json_str) or not json_str:
                return None

            # Convert to string if not already
            json_str = str(json_str).strip()

            # Handle various markdown wrapper formats
            json_content = self._extract_json_from_markdown(json_str)

            # Try to parse the extracted JSON
            return json.loads(json_content)

        except (json.JSONDecodeError, TypeError, ValueError) as e:
            logger.warning(f"Failed to parse JSON annotation. Error: {e}")
            logger.debug(f"Problematic content: {json_str[:200]}...")
            return None

    def _extract_json_from_markdown(self, text: str) -> str:
        """
        Extract JSON content from various markdown formats

        Args:
            text: Text that may contain JSON wrapped in markdown

        Returns:
            Clean JSON string
        """
        # Remove leading/trailing whitespace
        text = text.strip()

        # Pattern 1: ```json ... ```
        json_block_pattern = r'```json\s*\n?(.*?)\n?```'
        match = re.search(json_block_pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Pattern 2: ``` ... ``` (without json specifier)
        code_block_pattern = r'```\s*\n?(.*?)\n?```'
        match = re.search(code_block_pattern, text, re.DOTALL)
        if match:
            content = match.group(1).strip()
            # Check if it looks like JSON (starts with { and ends with })
            if content.startswith('{') and content.endswith('}'):
                return content

        # Pattern 3: Text containing "The json file is like this:" followed by JSON
        json_intro_pattern = r'(?:The json file is like this:|json file is like this:)\s*```?json?\s*\n?(.*?)\n?```?'
        match = re.search(json_intro_pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Pattern 4: Look for JSON object directly (starts with { and ends with })
        # Find the first { and last } to extract JSON object
        first_brace = text.find('{')
        last_brace = text.rfind('}')

        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            potential_json = text[first_brace:last_brace + 1]
            # Basic validation - count braces to ensure it's complete
            open_braces = potential_json.count('{')
            close_braces = potential_json.count('}')
            if open_braces == close_braces:
                return potential_json

        # Pattern 5: Remove common prefixes/suffixes that might interfere
        cleaned_text = text

        # Remove common prefixes
        prefixes_to_remove = [
            "The json file is like this:",
            "json file is like this:",
            "JSON:",
            "json:",
            "Response:",
            "Output:",
        ]

        for prefix in prefixes_to_remove:
            if cleaned_text.lower().startswith(prefix.lower()):
                cleaned_text = cleaned_text[len(prefix):].strip()
                break

        # Remove markdown code indicators
        cleaned_text = re.sub(r'^```json?\s*\n?', '', cleaned_text, flags=re.IGNORECASE | re.MULTILINE)
        cleaned_text = re.sub(r'\n?```\s*$', '', cleaned_text, flags=re.MULTILINE)

        # Final check - if it starts and ends with braces, return as is
        cleaned_text = cleaned_text.strip()
        if cleaned_text.startswith('{') and cleaned_text.endswith('}'):
            return cleaned_text

        # If nothing worked, return the original text and let JSON parser handle the error
        return text

    def validate_and_clean_annotation(self, annotation: Dict) -> Optional[Dict]:
        """
        Validate and clean annotation dictionary to ensure it has the expected structure

        Args:
            annotation: Parsed annotation dictionary

        Returns:
            Cleaned annotation dictionary or None if invalid
        """
        if not annotation or not isinstance(annotation, dict):
            return None

        # Required fields
        required_fields = ['role', 'confidence', 'justification']

        # Check if all required fields are present
        for field in required_fields:
            if field not in annotation:
                logger.warning(f"Missing required field '{field}' in annotation")
                return None

        # Clean and validate the annotation
        cleaned = {}

        # Validate role
        role = str(annotation.get('role', '')).strip()
        if role not in Config.VALID_ROLES:
            # Try to find a close match
            role_lower = role.lower()
            for valid_role in Config.VALID_ROLES:
                if valid_role.lower() in role_lower or role_lower in valid_role.lower():
                    role = valid_role
                    break
            else:
                logger.warning(f"Invalid role '{role}', defaulting to 'Borderline'")
                role = 'Borderline'

        cleaned['role'] = role

        # Validate confidence
        try:
            confidence = int(annotation.get('confidence', 1))
            if confidence < Config.MIN_CONFIDENCE or confidence > Config.MAX_CONFIDENCE:
                logger.warning(f"Confidence {confidence} out of range, clamping to valid range")
                confidence = max(Config.MIN_CONFIDENCE, min(Config.MAX_CONFIDENCE, confidence))
        except (ValueError, TypeError):
            logger.warning(f"Invalid confidence value, defaulting to 1")
            confidence = 1

        cleaned['confidence'] = confidence

        # Clean text fields
        cleaned['note'] = str(annotation.get('note', '')).strip()
        cleaned['justification'] = str(annotation.get('justification', '')).strip()
        cleaned['context_assessment'] = str(annotation.get('context_assessment', '')).strip()

        return cleaned

    def debug_json_parsing(self, csv_file_path: str, sample_size: int = 5):
        """
        Debug JSON parsing issues by examining a sample of annotations

        Args:
            csv_file_path: Path to a CSV file to examine
            sample_size: Number of samples to examine
        """
        logger.info(f"Debugging JSON parsing for {csv_file_path}")

        try:
            df = pd.read_csv(csv_file_path)

            if 'metadiscourse_annotation' not in df.columns:
                logger.error("metadiscourse_annotation column not found")
                return

            # Sample some annotations
            sample_annotations = df['metadiscourse_annotation'].dropna().head(sample_size)

            for idx, annotation in enumerate(sample_annotations):
                logger.info(f"\n--- Sample {idx + 1} ---")
                logger.info(f"Raw content (first 200 chars): {str(annotation)[:200]}...")

                # Try to extract JSON
                try:
                    extracted_json = self._extract_json_from_markdown(str(annotation))
                    logger.info(f"Extracted JSON: {extracted_json[:200]}...")

                    # Try to parse
                    parsed = json.loads(extracted_json)
                    logger.info(
                        f"✓ Successfully parsed: {parsed.get('role', 'N/A')} (confidence: {parsed.get('confidence', 'N/A')})")

                    # Validate
                    validated = self.validate_and_clean_annotation(parsed)
                    if validated:
                        logger.info(f"✓ Validation passed")
                    else:
                        logger.warning(f"✗ Validation failed")

                except json.JSONDecodeError as e:
                    logger.error(f"✗ JSON parsing failed: {e}")
                except Exception as e:
                    logger.error(f"✗ Unexpected error: {e}")

        except Exception as e:
            logger.error(f"Failed to debug CSV file: {e}")

    def merge_annotations(self, dataframes: Dict[str, pd.DataFrame], max_rows: int = None) -> pd.DataFrame:
        """
        Merge annotations from all three models into a single DataFrame

        Args:
            dataframes: Dictionary of DataFrames from each model
            max_rows: Maximum number of rows to process (None for all rows)

        Returns:
            Merged DataFrame with all annotations including context columns
        """
        # Start with the first model's DataFrame structure
        base_df = dataframes['claude'].copy()

        # Apply row limit early if specified
        if max_rows is not None:
            logger.info(f"Limiting processing to {max_rows} rows")
            base_df = base_df.head(max_rows)

        # Rename the annotation column for Claude
        base_df = base_df.rename(columns={'metadiscourse_annotation': 'claude_metadiscourse_annotation'})

        # Merge with other models
        for model in ['gemini', 'deepseek']:
            model_df = dataframes[model][
                ['thesis_code', 'section', 'sentence', 'expression', 'metadiscourse_annotation']].copy()

            # Apply same row limit to other models for consistency
            if max_rows is not None:
                model_df = model_df.head(max_rows)

            model_df = model_df.rename(columns={'metadiscourse_annotation': f'{model}_metadiscourse_annotation'})

            base_df = base_df.merge(
                model_df,
                on=['thesis_code', 'section', 'sentence', 'expression'],
                how='outer'
            )

        # Select required columns including context columns if they exist
        required_columns = [
            'thesis_code', 'section', 'sentence', 'expression',
            'claude_metadiscourse_annotation', 'gemini_metadiscourse_annotation', 'deepseek_metadiscourse_annotation'
        ]

        # Add context columns if they exist in the dataset
        if 'context_before' in base_df.columns:
            required_columns.append('context_before')
        if 'context_after' in base_df.columns:
            required_columns.append('context_after')

        return base_df[required_columns]

    def optimize_annotation(self, claude_ann: Dict, gemini_ann: Dict, deepseek_ann: Dict,
                            sentence: str, expression: str, context_before: str = "", context_after: str = "") -> Dict:
        """
        Use GPT to optimize annotations from three models

        Args:
            claude_ann: Claude's annotation
            gemini_ann: Gemini's annotation
            deepseek_ann: DeepSeek's annotation
            sentence: The sentence being annotated
            expression: The specific expression being annotated
            context_before: Context before the sentence
            context_after: Context after the sentence

        Returns:
            Optimized annotation dictionary
        """
        # Extract limited context based on word limits
        limited_context_before = self._extract_limited_context(context_before, self.context_words_before)
        limited_context_after = self._extract_limited_context(context_after, self.context_words_after)

        # Build the context section for the prompt
        context_section = ""
        if limited_context_before or limited_context_after:
            context_section = f"""
CONTEXT INFORMATION:
CONTEXT_BEFORE ({self.context_words_before} words): {limited_context_before if limited_context_before else "[No context before]"}
CONTEXT_AFTER ({self.context_words_after} words): {limited_context_after if limited_context_after else "[No context after]"}
"""

        # Build the full prompt
        prompt = f"""
{self.optimization_prompt}

ANNOTATION TARGET:
SENTENCE: {sentence}
EXPRESSION: {expression}
{context_section}
MODEL ANNOTATIONS TO ANALYZE:

CLAUDE ANNOTATION:
{json.dumps(claude_ann, indent=2) if claude_ann else "No annotation available"}

GEMINI ANNOTATION:
{json.dumps(gemini_ann, indent=2) if gemini_ann else "No annotation available"}

DEEPSEEK ANNOTATION:
{json.dumps(deepseek_ann, indent=2) if deepseek_ann else "No annotation available"}

Based on the sentence, expression, context (if available), and the three model annotations above, provide your optimized final decision in JSON format.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=Config.TEMPERATURE,
                max_tokens=Config.MAX_TOKENS
            )

            response_text = response.choices[0].message.content.strip()

            # Use our robust JSON extraction method
            extracted_json = self._extract_json_from_markdown(response_text)
            return json.loads(extracted_json)

        except Exception as e:
            logger.error(f"Error in GPT optimization: {e}")
            # Fallback: return majority vote or highest confidence annotation
            return self._fallback_optimization(claude_ann, gemini_ann, deepseek_ann)

    def _fallback_optimization(self, claude_ann: Dict, gemini_ann: Dict, deepseek_ann: Dict) -> Dict:
        """
        Fallback optimization method using majority vote and confidence scores

        Args:
            claude_ann: Claude's annotation
            gemini_ann: Gemini's annotation
            deepseek_ann: DeepSeek's annotation

        Returns:
            Optimized annotation dictionary
        """
        annotations = [ann for ann in [claude_ann, gemini_ann, deepseek_ann] if ann is not None]

        if not annotations:
            return {
                "role": "Borderline",
                "confidence": 1,
                "note": "No valid annotations available",
                "justification": "Fallback due to missing or invalid annotations",
                "context_assessment": "Insufficient data"
            }

        # Get majority role
        roles = [ann.get('role', 'Borderline') for ann in annotations]
        role_counts = Counter(roles)
        majority_role = role_counts.most_common(1)[0][0]

        # Get highest confidence
        confidences = [ann.get('confidence', 1) for ann in annotations]
        max_confidence = max(confidences)

        # Find annotation with highest confidence for that role
        best_ann = max(annotations, key=lambda x: x.get('confidence', 1))

        return {
            "role": majority_role,
            "confidence": max_confidence,
            "note": f"Fallback optimization: majority role with highest confidence",
            "justification": best_ann.get('justification', 'No justification available'),
            "context_assessment": best_ann.get('context_assessment', 'No context assessment available')
        }

    def process_all_annotations(self, merged_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process all annotations and create optimized decisions

        Args:
            merged_df: Merged DataFrame with all model annotations

        Returns:
            DataFrame with optimized final decisions
        """
        optimized_decisions = []
        total_rows = len(merged_df)

        logger.info(f"Processing {total_rows} rows for optimization...")

        for idx, row in merged_df.iterrows():
            if idx % 10 == 0 or idx < 10:  # More frequent logging for small datasets
                logger.info(f"Processing row {idx + 1}/{total_rows}")

            # Parse and validate annotations
            claude_ann = self.parse_json_annotation(row['claude_metadiscourse_annotation'])
            claude_ann = self.validate_and_clean_annotation(claude_ann) if claude_ann else None

            gemini_ann = self.parse_json_annotation(row['gemini_metadiscourse_annotation'])
            gemini_ann = self.validate_and_clean_annotation(gemini_ann) if gemini_ann else None

            deepseek_ann = self.parse_json_annotation(row['deepseek_metadiscourse_annotation'])
            deepseek_ann = self.validate_and_clean_annotation(deepseek_ann) if deepseek_ann else None

            # Get context information if available
            context_before = row.get('context_before', '')
            context_after = row.get('context_after', '')

            # Optimize annotation with context
            optimized = self.optimize_annotation(
                claude_ann, gemini_ann, deepseek_ann,
                row['sentence'], row['expression'],
                context_before, context_after
            )

            optimized_decisions.append(json.dumps(optimized))

        # Add optimized decisions to DataFrame
        merged_df['Optimized_final_decision'] = optimized_decisions

        logger.info(f"Completed processing {total_rows} rows")
        return merged_df

    def analyze_model_errors(self, optimized_df: pd.DataFrame) -> Dict:
        """
        Analyze patterns in model errors and disagreements

        Args:
            optimized_df: DataFrame with optimized decisions

        Returns:
            Dictionary containing error analysis
        """
        analysis = {
            'disagreement_cases': [],
            'model_accuracy': {'claude': 0, 'gemini': 0, 'deepseek': 0},
            'common_error_patterns': [],
            'confidence_analysis': {}
        }

        total_cases = len(optimized_df)
        agreements = {'claude': 0, 'gemini': 0, 'deepseek': 0}

        for idx, row in optimized_df.iterrows():
            # Parse and validate all annotations
            claude_ann = self.parse_json_annotation(row['claude_metadiscourse_annotation'])
            claude_ann = self.validate_and_clean_annotation(claude_ann) if claude_ann else None

            gemini_ann = self.parse_json_annotation(row['gemini_metadiscourse_annotation'])
            gemini_ann = self.validate_and_clean_annotation(gemini_ann) if gemini_ann else None

            deepseek_ann = self.parse_json_annotation(row['deepseek_metadiscourse_annotation'])
            deepseek_ann = self.validate_and_clean_annotation(deepseek_ann) if deepseek_ann else None

            optimized_ann = self.parse_json_annotation(row['Optimized_final_decision'])
            optimized_ann = self.validate_and_clean_annotation(optimized_ann) if optimized_ann else None

            if not optimized_ann:
                continue

            optimized_role = optimized_ann.get('role')

            # Check agreements with optimized decision
            for model, ann in [('claude', claude_ann), ('gemini', gemini_ann), ('deepseek', deepseek_ann)]:
                if ann and ann.get('role') == optimized_role:
                    agreements[model] += 1

            # Identify disagreement cases
            model_roles = []
            if claude_ann: model_roles.append(('claude', claude_ann.get('role')))
            if gemini_ann: model_roles.append(('gemini', gemini_ann.get('role')))
            if deepseek_ann: model_roles.append(('deepseek', deepseek_ann.get('role')))

            unique_roles = set([role for _, role in model_roles])
            if len(unique_roles) > 1:  # Disagreement
                analysis['disagreement_cases'].append({
                    'index': idx,
                    'sentence': row['sentence'],
                    'expression': row['expression'],
                    'model_decisions': dict(model_roles),
                    'optimized_decision': optimized_role
                })

        # Calculate accuracies
        for model in agreements:
            analysis['model_accuracy'][model] = agreements[model] / total_cases if total_cases > 0 else 0

        return analysis

    def generate_improved_prompt(self, error_analysis: Dict, current_prompt_path: str = None) -> str:
        """
        Generate an improved annotation prompt based on error analysis

        Args:
            error_analysis: Results from analyze_model_errors
            current_prompt_path: Absolute path to current internal/external prompt

        Returns:
            Improved prompt text
        """
        current_prompt = ""
        if current_prompt_path and os.path.exists(current_prompt_path):
            try:
                with open(current_prompt_path, 'r', encoding='utf-8') as f:
                    current_prompt = f.read()
            except Exception as e:
                logger.warning(f"Could not read current prompt file: {e}")

        # Analyze common patterns in disagreement cases
        disagreements = error_analysis['disagreement_cases']

        # Generate improvement suggestions based on patterns
        improvement_prompt = f"""
IMPROVED ANNOTATION PROMPT
==========================

Based on analysis of {len(disagreements)} disagreement cases between annotator models,
the following improvements have been identified:

ACCURACY SCORES:
- Claude: {error_analysis['model_accuracy']['claude']:.2%}
- Gemini: {error_analysis['model_accuracy']['gemini']:.2%}
- DeepSeek: {error_analysis['model_accuracy']['deepseek']:.2%}

COMMON ERROR PATTERNS IDENTIFIED:
[Analysis of specific patterns would be added here based on disagreement cases]

ENHANCED GUIDELINES:

1. METADISCOURSE vs PROPOSITIONAL DISTINCTION:
   - Pay special attention to expressions that organize, evaluate, or guide reader interpretation
   - Consider the functional role of expressions in discourse management
   - Look for signals that help readers navigate the text structure

2. CONFIDENCE ASSESSMENT:
   - Use confidence scores more systematically
   - Score 1-2 for highly uncertain cases
   - Score 4-5 only for clear, unambiguous cases
   - Score 3 for borderline cases with some uncertainty

3. CONTEXT CONSIDERATIONS:
   - Always assess whether sufficient context is provided
   - Consider both local and global discourse context
   - Note when additional context might change the classification

4. JUSTIFICATION REQUIREMENTS:
   - Provide specific linguistic evidence
   - Reference discourse function explicitly
   - Explain why alternative classifications were rejected

ORIGINAL PROMPT:
{current_prompt}

[Additional specific improvements would be added based on detailed error pattern analysis]
"""

        return improvement_prompt

    def run_full_pipeline(self, base_path: str = None, output_path: str = None,
                          debug_parsing: bool = False, max_rows: int = None):
        """
        Run the complete optimization pipeline

        Args:
            base_path: Base directory containing model subdirectories (relative to project root)
            output_path: Path for output CSV file (relative to project root)
            debug_parsing: Whether to run JSON parsing debug before processing
            max_rows: Maximum number of rows to process (None for all rows)

        Returns:
            Tuple of (optimized DataFrame, error analysis)
        """
        logger.info("Starting annotation optimization pipeline...")

        if max_rows is not None:
            logger.info(f"Row processing limited to: {max_rows}")

        # Use config defaults if not provided
        if base_path is None:
            base_path = Config.BASE_DATA_PATH
        if output_path is None:
            output_path = Config.OUTPUT_CSV_PATH

        # Ensure output directory exists and path is relative to project root
        full_output_path = os.path.join(project_root, output_path)
        output_dir = os.path.dirname(full_output_path)
        os.makedirs(output_dir, exist_ok=True)

        # Load CSV files
        dataframes = self.load_csv_files(base_path)

        # Debug JSON parsing if requested
        if debug_parsing:
            logger.info("Running JSON parsing debug...")
            full_base_path = os.path.join(project_root, base_path)
            for model in Config.MODEL_DIRECTORIES:
                model_dir = os.path.join(full_base_path, model)
                if os.path.exists(model_dir):
                    csv_files = [f for f in os.listdir(model_dir) if f.endswith('.csv')]
                    if csv_files:
                        csv_path = os.path.join(model_dir, csv_files[0])
                        logger.info(f"Debugging {model} annotations...")
                        self.debug_json_parsing(csv_path)

        # Merge annotations with row limiting
        merged_df = self.merge_annotations(dataframes, max_rows)
        logger.info(f"Merged {len(merged_df)} annotation cases")

        # Process and optimize annotations
        optimized_df = self.process_all_annotations(merged_df)

        # Save optimized results
        optimized_df.to_csv(full_output_path, index=False)
        logger.info(f"Optimized annotations saved to {full_output_path}")

        # Analyze errors
        error_analysis = self.analyze_model_errors(optimized_df)

        # Generate improved prompt
        internal_external_path = os.path.join(project_root, Config.INTERNAL_EXTERNAL_PROMPT_PATH)
        improved_prompt = self.generate_improved_prompt(
            error_analysis,
            internal_external_path
        )

        # Save improved prompt
        improved_prompt_path = os.path.join(project_root, Config.IMPROVED_PROMPT_PATH)
        os.makedirs(os.path.dirname(improved_prompt_path), exist_ok=True)
        with open(improved_prompt_path, "w", encoding="utf-8") as f:
            f.write(improved_prompt)

        logger.info("Pipeline completed successfully!")
        return optimized_df, error_analysis


# Usage example
if __name__ == "__main__":
    # Initialize optimizer
    optimizer = AnnotationOptimizer(
        openai_api_key=Config.OPENAI_API_KEY,
        model=Config.OPENAI_MODEL
    )

    # Run full pipeline with optional debugging and row limiting
    try:
        optimized_df, error_analysis = optimizer.run_full_pipeline(
            debug_parsing=True,
            max_rows=5  # Example: process only 5 rows
        )

        print(f"Optimization completed!")
        print(f"Total cases processed: {len(optimized_df)}")
        print(f"Model accuracies: {error_analysis['model_accuracy']}")
        print(f"Disagreement cases: {len(error_analysis['disagreement_cases'])}")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise