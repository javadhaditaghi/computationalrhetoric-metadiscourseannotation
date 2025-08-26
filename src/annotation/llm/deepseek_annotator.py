import os
import pandas as pd
import json
import time
from typing import List, Dict, Optional
import logging
from pathlib import Path
import requests
from tqdm import tqdm
from dotenv import load_dotenv


class DeepSeekMetadiscourseAnnotator:
    def __init__(self, api_key: Optional[str] = None, model: str = "deepseek-reasoner",
                 context_words_before: Optional[int] = None, context_words_after: Optional[int] = None,
                 base_url: str = "https://api.deepseek.com/v1"):
        """
        Initialize the DeepSeek annotator for metadiscourse labeling.

        Args:
            api_key: DeepSeek API key. If None, will look for DEEPSEEK_API_KEY in .env file
            model: DeepSeek model to use for annotation (default: deepseek-chat)
            context_words_before: Number of words to include from context_before (None = use full context)
            context_words_after: Number of words to include from context_after (None = use full context)
            base_url: DeepSeek API base URL
        """
        # Load environment variables from .env file
        load_dotenv()

        self.api_key = api_key or os.getenv('DEEPSEEK_API_KEY')
        if not self.api_key:
            raise ValueError("API key must be provided either as parameter or DEEPSEEK_API_KEY in .env file")

        self.model = model
        self.base_url = base_url
        self.context_words_before = context_words_before
        self.context_words_after = context_words_after

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Load prompt
        self.prompt = self._load_prompt()

        # Log context settings
        if self.context_words_before or self.context_words_after:
            self.logger.info(f"Context window: {self.context_words_before or 'full'} words before, "
                             f"{self.context_words_after or 'full'} words after")

    def _load_prompt(self) -> str:
        """Load the metadiscourse prompt from file."""
        prompt_path = Path("src/annotation/prompt/internalexternal.txt")

        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found at {prompt_path}")

        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt = f.read().strip()

        self.logger.info(f"Loaded prompt from {prompt_path}")
        return prompt

    def _load_dataset(self, data_dir: str = "data", specific_file: Optional[str] = None) -> pd.DataFrame:
        """
        Load the dataset from CSV files in the data directory.

        Args:
            data_dir: Directory containing the CSV files
            specific_file: If provided, only load this specific file (e.g., "dataset1.csv")

        Returns:
            Combined DataFrame with all data
        """
        data_path = Path(data_dir)

        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found at {data_path}")

        # Handle specific file request
        if specific_file:
            specific_path = data_path / specific_file
            if not specific_path.exists():
                raise FileNotFoundError(f"Specific file not found: {specific_path}")

            if not specific_path.suffix.lower() == '.csv':
                raise ValueError(f"File must be a CSV: {specific_file}")

            df = pd.read_csv(specific_path)
            df['source_file'] = specific_file
            self.logger.info(f"Loaded {len(df)} rows from specific file: {specific_file}")

            # Validate required columns (added 'expression' to required columns)
            required_columns = ['sentence', 'context_before', 'context_after', 'thesis_code', 'section', 'expression']
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                raise ValueError(f"Missing required columns in {specific_file}: {missing_columns}")

            return df

        # Find CSV files (original behavior)
        csv_files = list(data_path.glob("*.csv"))

        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {data_path}")

        # Load and combine all CSV files
        dfs = []
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            df['source_file'] = csv_file.name
            dfs.append(df)

        combined_df = pd.concat(dfs, ignore_index=True)

        # Validate required columns (added 'expression' to required columns)
        required_columns = ['sentence', 'context_before', 'context_after', 'thesis_code', 'section', 'expression']
        missing_columns = [col for col in required_columns if col not in combined_df.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        self.logger.info(f"Loaded {len(combined_df)} rows from {len(csv_files)} CSV files")
        return combined_df

    def _truncate_context(self, text: str, num_words: Optional[int], from_end: bool = False) -> str:
        """
        Truncate context to specified number of words.

        Args:
            text: Text to truncate
            num_words: Number of words to keep (None = keep all)
            from_end: If True, take words from the end; if False, take from beginning

        Returns:
            Truncated text
        """
        if num_words is None or pd.isna(text) or not text.strip():
            return str(text) if pd.notna(text) else ""

        # Special case: if num_words is 0, return empty string
        if num_words == 0:
            return ""

        words = str(text).split()

        if len(words) <= num_words:
            return str(text)

        if from_end:
            # Take last N words (for context_before)
            selected_words = words[-num_words:]
            truncated = " ".join(selected_words)
            return f"...{truncated}"
        else:
            # Take first N words (for context_after)
            selected_words = words[:num_words]
            truncated = " ".join(selected_words)
            return f"{truncated}..."

    def _create_annotation_prompt(self, sentence: str, context_before: str, context_after: str,
                                  thesis_code: str, section: str, expression: str) -> str:
        """
        Create the full prompt for annotation by combining the base prompt with the data.

        Args:
            sentence: The target sentence to analyze
            context_before: Context before the sentence
            context_after: Context after the sentence
            thesis_code: The thesis code identifier
            section: The section identifier
            expression: The specific expression to classify

        Returns:
            Complete prompt for DeepSeek
        """
        # Handle NaN values
        original_context_before = str(context_before) if pd.notna(context_before) else ""
        original_context_after = str(context_after) if pd.notna(context_after) else ""
        sentence = str(sentence) if pd.notna(sentence) else ""
        thesis_code = str(thesis_code) if pd.notna(thesis_code) else ""
        section = str(section) if pd.notna(section) else ""
        expression = str(expression) if pd.notna(expression) else ""

        # Apply context truncation if specified
        if self.context_words_before is not None:
            context_before = self._truncate_context(original_context_before, self.context_words_before, from_end=True)
        else:
            context_before = original_context_before

        if self.context_words_after is not None:
            context_after = self._truncate_context(original_context_after, self.context_words_after, from_end=False)
        else:
            context_after = original_context_after

        # Debug prints
        print("=" * 50)
        print(f"THESIS CODE: '{thesis_code}'")
        print(f"SECTION: '{section}'")
        print(f"EXPRESSION: '{expression}'")
        print(f"TRUNCATED CONTEXT_BEFORE: '{context_before}'")
        print(f"CONTEXT_BEFORE LENGTH: {len(context_before.split()) if context_before else 0} words")
        print(f"TARGET SENTENCE: '{sentence}'")
        print(f"TRUNCATED CONTEXT_AFTER: '{context_after}'")
        print(f"CONTEXT_AFTER LENGTH: {len(context_after.split()) if context_after else 0} words")
        print("=" * 50)

        full_prompt = f"""{self.prompt}

Thesis Code: {thesis_code}

Section: {section}

Context Before: {context_before}

Target Sentence: {sentence}

Context After: {context_after}

Expression to Classify: {expression}

Please analyze the given expression within the context of the target sentence and classify it as either "propositional" or "metadiscourse". 

Provide your classification and reasoning for the expression: "{expression}" """

        return full_prompt

    def _call_deepseek(self, prompt: str, max_retries: int = 5, initial_delay: float = 2.0) -> Optional[str]:
        """
        Call DeepSeek API with improved retry logic for handling overloaded servers.

        Args:
            prompt: The prompt to send to DeepSeek
            max_retries: Maximum number of retries
            initial_delay: Initial delay between retries in seconds

        Returns:
            DeepSeek's response or None if failed
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 1000,
            "temperature": 0,
            "stream": False
        }

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=60
                )

                # Check for successful response
                if response.status_code == 200:
                    response_data = response.json()
                    if 'choices' in response_data and len(response_data['choices']) > 0:
                        return response_data['choices'][0]['message']['content']
                    else:
                        self.logger.error(f"Invalid response format: {response_data}")
                        return None

                # Handle rate limiting
                elif response.status_code == 429:
                    if attempt < max_retries - 1:
                        wait_time = initial_delay * (2 ** attempt)
                        self.logger.warning(
                            f"Rate limit hit, waiting {wait_time:.1f}s before retry {attempt + 1}/{max_retries}")
                        time.sleep(wait_time)
                    else:
                        self.logger.error("Max retries reached due to rate limits")
                        return None

                # Handle server errors
                elif response.status_code in [500, 502, 503, 504]:
                    if attempt < max_retries - 1:
                        wait_time = initial_delay * (2 ** attempt) + (time.time() % 1)  # Add jitter
                        self.logger.warning(
                            f"Server error {response.status_code}, waiting {wait_time:.1f}s before retry {attempt + 1}/{max_retries}")
                        time.sleep(wait_time)
                    else:
                        self.logger.error(f"Max retries reached due to server errors")
                        return None

                # Handle other errors
                else:
                    try:
                        error_data = response.json()
                        error_message = error_data.get('error', {}).get('message', f'HTTP {response.status_code}')
                    except:
                        error_message = f'HTTP {response.status_code}'

                    self.logger.error(f"API error (attempt {attempt + 1}/{max_retries}): {error_message}")
                    if attempt < max_retries - 1:
                        time.sleep(initial_delay)
                    else:
                        return None

            except requests.exceptions.Timeout:
                self.logger.warning(f"Request timeout (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(initial_delay)
                else:
                    return None

            except requests.exceptions.ConnectionError as e:
                self.logger.warning(f"Connection error (attempt {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(initial_delay)
                else:
                    return None

            except KeyboardInterrupt:
                self.logger.info("Process interrupted by user")
                raise

            except Exception as e:
                self.logger.error(f"Unexpected error (attempt {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(initial_delay)
                else:
                    return None

        return None

    def annotate_batch(self, df: pd.DataFrame, batch_size: int = 10, output_file: str = None,
                       resume_from: int = 0) -> pd.DataFrame:
        """
        Annotate a batch of sentences with metadiscourse labels.

        Args:
            df: DataFrame containing the data to annotate
            batch_size: Number of rows to process at once (for saving progress)
            output_file: Optional file to save results incrementally
            resume_from: Row index to resume from (useful if process was interrupted)

        Returns:
            DataFrame with annotations added
        """
        results = []

        # Check if there's an existing temp file to resume from
        temp_file = f"{output_file}.temp" if output_file else None
        if temp_file and Path(temp_file).exists() and resume_from == 0:
            try:
                existing_results = pd.read_csv(temp_file)
                results = existing_results.to_dict('records')
                resume_from = len(results)
                self.logger.info(f"Resuming from row {resume_from} using existing temp file")
            except Exception as e:
                self.logger.warning(f"Could not load temp file: {e}")
                results = []
                resume_from = 0

        # Setup progress bar
        total_rows = len(df)
        pbar = tqdm(total=total_rows, desc="Annotating metadiscourse", initial=resume_from)

        failed_consecutive = 0
        max_consecutive_failures = 10

        for idx, row in df.iterrows():
            # Skip rows if resuming
            if idx < resume_from:
                continue

            # Create prompt with thesis_code, section, and expression
            prompt = self._create_annotation_prompt(
                row['sentence'],
                row['context_before'],
                row['context_after'],
                row['thesis_code'],
                row['section'],
                row['expression']  # Added expression parameter
            )

            # Call DeepSeek
            annotation = self._call_deepseek(prompt)

            # Store result
            result_row = row.copy()
            result_row['metadiscourse_annotation'] = annotation
            result_row['annotation_status'] = 'success' if annotation else 'failed'
            result_row['annotation_model'] = self.model  # Add model name column
            result_row['row_index'] = idx

            results.append(result_row)

            # Track consecutive failures
            if annotation is None:
                failed_consecutive += 1
                if failed_consecutive >= max_consecutive_failures:
                    self.logger.error(f"Too many consecutive failures ({max_consecutive_failures}). Stopping process.")
                    self.logger.info("You can resume later by running the script again.")
                    break
            else:
                failed_consecutive = 0

            # Save progress incrementally
            if temp_file and len(results) % batch_size == 0:
                try:
                    temp_df = pd.DataFrame(results)
                    temp_df.to_csv(temp_file, index=False)
                    self.logger.info(f"Saved progress: {len(results)}/{total_rows} rows completed")
                except Exception as e:
                    self.logger.warning(f"Could not save progress: {e}")

            pbar.update(1)

            # Small delay to be respectful to the API
            time.sleep(0.5)

        pbar.close()

        # Create final DataFrame
        annotated_df = pd.DataFrame(results)

        # Report statistics
        if len(annotated_df) > 0:
            success_count = len(annotated_df[annotated_df['annotation_status'] == 'success'])
            self.logger.info(f"Annotation completed: {success_count}/{len(annotated_df)} successful")

            # Log statistics by thesis_code and section
            if 'thesis_code' in annotated_df.columns and 'section' in annotated_df.columns:
                self.logger.info("Processing statistics by thesis and section:")
                stats = annotated_df.groupby(['thesis_code', 'section']).agg({
                    'annotation_status': lambda x: (x == 'success').sum(),
                    'sentence': 'count'
                }).rename(columns={'annotation_status': 'successful', 'sentence': 'total'})
                for (thesis, section), row in stats.iterrows():
                    self.logger.info(f"  {thesis} - {section}: {row['successful']}/{row['total']} successful")
        else:
            self.logger.warning("No annotations completed")

        return annotated_df

    def save_results(self, df: pd.DataFrame, output_file: str = "annotated_metadiscourse.csv"):
        """
        Save the annotated results to a CSV file.

        Args:
            df: Annotated DataFrame
            output_file: Output file path
        """
        # Ensure the result/deepseek directory exists
        output_path = Path(output_file)
        result_dir = Path("result/deepseek")
        result_dir.mkdir(parents=True, exist_ok=True)

        # Update output path to be in result/deepseek directory
        output_path = result_dir / output_path.name

        df.to_csv(output_path, index=False)
        self.logger.info(f"Results saved to {output_path}")

        # Also save a summary with thesis_code and section breakdown
        summary_file = output_path.with_suffix('.json')
        summary = {
            'total_rows': int(len(df)),
            'successful_annotations': int(len(df[df['annotation_status'] == 'success'])),
            'failed_annotations': int(len(df[df['annotation_status'] == 'failed'])),
            'success_rate': float(len(df[df['annotation_status'] == 'success']) / len(df)) if len(df) > 0 else 0.0,
            'annotation_model': self.model  # Add model info to summary
        }

        # Add breakdown by thesis_code and section if available
        if 'thesis_code' in df.columns and 'section' in df.columns:
            summary['breakdown_by_thesis_section'] = {}

            # Create breakdown using standard groupby operations
            for (thesis, section), group in df.groupby(['thesis_code', 'section']):
                thesis_str = str(thesis)
                section_str = str(section)

                if thesis_str not in summary['breakdown_by_thesis_section']:
                    summary['breakdown_by_thesis_section'][thesis_str] = {}

                total = int(len(group))
                successful = int((group['annotation_status'] == 'success').sum())
                failed = int((group['annotation_status'] == 'failed').sum())
                success_rate = float(successful / total) if total > 0 else 0.0

                summary['breakdown_by_thesis_section'][thesis_str][section_str] = {
                    'total': total,
                    'successful': successful,
                    'failed': failed,
                    'success_rate': success_rate
                }

        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        self.logger.info(f"Summary saved to {summary_file}")


def main(specific_file: Optional[str] = None, context_before: Optional[int] = None,
         context_after: Optional[int] = None, num_rows: Optional[int] = None):
    """
    Main function to run the metadiscourse annotation.

    Args:
        specific_file: Optional specific CSV file to process (e.g., "dataset1.csv")
        context_before: Number of words to include from context_before (None = use full context)
        context_after: Number of words to include from context_after (None = use full context)
        num_rows: Number of rows to process (None = process all rows)
    """
    try:
        annotator = DeepSeekMetadiscourseAnnotator(
            context_words_before=context_before,
            context_words_after=context_after
        )

        # Load dataset
        df = annotator._load_dataset(specific_file=specific_file)

        # Limit rows if requested
        if num_rows is not None:
            df = df.head(num_rows)
            annotator.logger.info(f"Processing only first {num_rows} rows")

        # Extract model name for filename (remove special characters and make it short)
        model_short = annotator.model.replace("deepseek-", "").replace("-", "_")

        # Create output filename based on input and context settings
        if specific_file:
            base_name = Path(specific_file).stem
            if context_before or context_after:
                context_suffix = f"_ctx{context_before or 'full'}_{context_after or 'full'}"
                output_file = f"annotated_{base_name}{context_suffix}_{model_short}.csv"
            else:
                output_file = f"annotated_{base_name}_{model_short}.csv"
        else:
            if context_before or context_after:
                context_suffix = f"_ctx{context_before or 'full'}_{context_after or 'full'}"
                output_file = f"annotated_metadiscourse{context_suffix}_{model_short}.csv"
            else:
                output_file = f"annotated_metadiscourse_{model_short}.csv"

        # Create output base for temp files (still in data directory for temp files)
        if specific_file:
            base_name = Path(specific_file).stem
            if context_before or context_after:
                context_suffix = f"_ctx{context_before or 'full'}_{context_after or 'full'}"
                output_base = f"data/annotated_{base_name}{context_suffix}_{model_short}"
            else:
                output_base = f"data/annotated_{base_name}_{model_short}"
        else:
            if context_before or context_after:
                context_suffix = f"_ctx{context_before or 'full'}_{context_after or 'full'}"
                output_base = f"data/annotated_metadiscourse{context_suffix}_{model_short}"
            else:
                output_base = f"data/annotated_metadiscourse_{model_short}"

        # Run annotation
        annotated_df = annotator.annotate_batch(
            df,
            batch_size=10,
            output_file=output_base
        )

        # Save results
        annotator.save_results(annotated_df, output_file)

        print("Annotation completed successfully!")
        print(f"Processed {len(annotated_df)} rows")
        print(
            f"Success rate: {len(annotated_df[annotated_df['annotation_status'] == 'success']) / len(annotated_df) * 100:.1f}%")
        print(f"Results saved to: result/deepseek/{output_file}")
        print(f"Annotation model: {annotator.model}")

        # Print breakdown by thesis and section
        if 'thesis_code' in annotated_df.columns and 'section' in annotated_df.columns:
            print("\nProcessing breakdown by thesis and section:")
            stats = annotated_df.groupby(['thesis_code', 'section']).agg({
                'annotation_status': lambda x: (x == 'success').sum(),
                'sentence': 'count'
            }).rename(columns={'annotation_status': 'successful', 'sentence': 'total'})
            for (thesis, section), row in stats.iterrows():
                success_rate = (row['successful'] / row['total'] * 100) if row['total'] > 0 else 0
                print(f"  {thesis} - {section}: {row['successful']}/{row['total']} ({success_rate:.1f}%)")

    except Exception as e:
        logging.error(f"Annotation failed: {str(e)}")
        raise


if __name__ == "__main__":
    import sys

    # Parse command line arguments
    specific_file = None
    context_before = None
    context_after = None
    num_rows = None

    if len(sys.argv) > 1:
        # First argument is the file
        if not sys.argv[1].isdigit() and not sys.argv[1].startswith('--'):
            specific_file = sys.argv[1]
            start_idx = 2
        else:
            start_idx = 1

        # Parse context arguments
        for i in range(start_idx, len(sys.argv)):
            arg = sys.argv[i]
            if arg.startswith('--context-before='):
                context_before = int(arg.split('=')[1])
            elif arg.startswith('--context-after='):
                context_after = int(arg.split('=')[1])
            elif arg.startswith('--context='):
                # Set both to same value
                both = int(arg.split('=')[1])
                context_before = both
                context_after = both
            elif arg.startswith('--rows='):
                num_rows = int(arg.split('=')[1])

    main(specific_file, context_before, context_after, num_rows)

    # How to run
    # Use 30 words before and 30 words after
    # python src/annotation/llm/deepseek_restricted.py test_restricted.csv --rows=2 --context-before=30 --context-after=30