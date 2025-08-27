# run_optimizer.py - Main execution script
import sys
import time
import os
from typing import Dict, Any
import logging

# Add paths to import modules
sys.path.insert(0, os.path.dirname(__file__))  # Add current directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))  # Add src directory

from util.config import Config
from src.annotation.llm.optimizer.gpt_annotator import AnnotationOptimizer


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, Config.LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(Config.LOG_FILE),
            logging.StreamHandler(sys.stdout)
        ]
    )


def validate_environment():
    """Validate that all required files and configurations are present"""
    logger = logging.getLogger(__name__)

    logger.info("Validating environment configuration...")

    # Use config validation
    errors = Config.validate_config()

    if errors:
        for error in errors:
            logger.error(error)
        return False

    # Check for CSV files in each model directory
    for model in Config.MODEL_DIRECTORIES:
        model_dir = os.path.join(Config.BASE_DATA_PATH, model)
        csv_files = [f for f in os.listdir(model_dir) if f.endswith('.csv')]
        if not csv_files:
            logger.error(f"No CSV files found in {model_dir}")
            return False
        else:
            logger.info(f"Found {len(csv_files)} CSV file(s) in {model_dir}")

    logger.info("Environment validation passed")
    return True


def print_results_summary(optimized_df, error_analysis: Dict[str, Any]):
    """Print a summary of the optimization results"""
    print("\n" + "=" * 60)
    print("ANNOTATION OPTIMIZATION RESULTS SUMMARY")
    print("=" * 60)

    print(f"\nTotal cases processed: {len(optimized_df)}")
    print(f"Disagreement cases found: {len(error_analysis['disagreement_cases'])}")

    print(f"\nModel Agreement with Optimized Decisions:")
    for model, accuracy in error_analysis['model_accuracy'].items():
        print(f"  {model.capitalize()}: {accuracy:.2%}")

    print(f"\nTop 5 Disagreement Cases:")
    for i, case in enumerate(error_analysis['disagreement_cases'][:5], 1):
        print(f"\n{i}. Expression: '{case['expression']}'")
        print(f"   Models: {case['model_decisions']}")
        print(f"   Optimized: {case['optimized_decision']}")
        print(f"   Sentence: {case['sentence'][:100]}...")

    print(f"\nFiles generated:")
    print(f"  - Optimized annotations: {Config.OUTPUT_CSV_PATH}")
    print(f"  - Improved prompt: {Config.IMPROVED_PROMPT_PATH}")
    print(f"  - Log file: annotation_optimization.log")


def main():
    """Main execution function"""
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("Starting annotation optimization process...")

    # Validate environment
    if not validate_environment():
        logger.error("Environment validation failed. Please check the requirements.")
        sys.exit(1)

    try:
        # Initialize optimizer
        logger.info("Initializing annotation optimizer...")
        optimizer = AnnotationOptimizer(
            openai_api_key=Config.OPENAI_API_KEY,
            model=Config.OPENAI_MODEL
        )

        # Run optimization pipeline
        start_time = time.time()
        optimized_df, error_analysis = optimizer.run_full_pipeline(
            base_path=Config.BASE_DATA_PATH,
            output_path=Config.OUTPUT_CSV_PATH
        )

        end_time = time.time()
        processing_time = end_time - start_time

        logger.info(f"Optimization completed in {processing_time:.2f} seconds")

        # Print results summary
        print_results_summary(optimized_df, error_analysis)

        # Save detailed error analysis
        import json
        error_analysis_path = '/result/optimized/error_analysis.json'
        with open(error_analysis_path, 'w') as f:
            # Convert any non-serializable objects to strings
            serializable_analysis = {
                'model_accuracy': error_analysis['model_accuracy'],
                'disagreement_count': len(error_analysis['disagreement_cases']),
                'disagreement_cases': error_analysis['disagreement_cases'][:20]  # Save top 20
            }
            json.dump(serializable_analysis, f, indent=2)

        print(f"  - Detailed error analysis: {error_analysis_path}")

    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Optimization failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

# requirements.txt - Dependencies
"""
pandas>=1.5.0
openai>=1.0.0
python-dotenv>=1.0.0
"""

# .env.example - Environment variables template
"""
# Copy this file to .env and fill in your actual API key
OPENAI_API_KEY=sk-your_openai_api_key_here
"""