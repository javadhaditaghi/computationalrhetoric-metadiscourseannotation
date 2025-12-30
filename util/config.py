# util/config.py - Configuration file
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    OPENAI_MODEL = 'gpt-5'  # or 'gpt-3.5-turbo' for faster/cheaper processing

    # File paths (relative to project root)
    BASE_DATA_PATH = 'result'
    # OUTPUT_DIR = 'result/optimized'
    OUTPUT_CSV_PATH_WITHOUT_BOUNDARY = 'result/optimized/without_boundary/optimized_annotations.csv'
    OPTIMIZATION_PROMPT_PATH = 'src/annotation/prompt/adjunction_protocol.json'
    INTERNAL_EXTERNAL_PROMPT_PATH = 'src/annotation/prompt/internalexternal.txt'
    IMPROVED_PROMPT_PATH = 'src/annotation/prompt/improved_internalexternal.txt'
    ADJUDICATION_PROTOCOL_PATH = "src/annotation/prompt/adjunction_protocol.json"
    ADJUNCTION_PROTOCOL_PATH = "src/annotation/prompt/adjunction_protocol.json"
    MINIMAL_ADJUDICATION_PROMPT_PATH = "src/annotation/prompt/minimal_adjudication_prompt.json"
    ANNOTATION_FRAMEWORK_PATH = "src/annotation/prompt/guidelines.json"
    FAILED_ANNOTATION_PATH = "data/output/failed_annotation_error.csv"

    # Without boundary prompts
    ADJUDICATION_PROTOCOL_PATH_WITHOUT_BOUNDARY = 'src/annotation/prompt/without_boundary/adjunction_protocol_no_boundaries.json'
    ADJUNCTION_PROTOCOL_PATH_WITHOUT_BOUNDARY = 'src/annotation/prompt/without_boundary/adjunction_protocol_no_boundaries.json'
    OPTIMIZATION_PROMPT_PATH_WITHOUT_BOUNDARY = 'src/annotation/prompt/without_boundary/adjunction_protocol_no_boundaries.json'
    MINIMAL_ADJUDICATION_PROMPT_PATH_WITHOUT_BOUNDARY = "src/annotation/prompt/without_boundary/minimal_adjudication_prompt_no_boundaries.json"


    # Processing configuration
    BATCH_SIZE = 10  # Process annotations in batches to manage API rate limits
    MAX_RETRIES = 3  # Number of retries for API calls
    RETRY_DELAY = 5  # Seconds to wait between retries
    TEMPERATURE = 0  # OpenAI temperature for consistency
    MAX_TOKENS = 12000  # Maximum tokens for GPT responses

    # Model directories
    MODEL_DIRECTORIES = ['claude', 'gemini', 'deepseek']

    # Logging
    LOG_LEVEL = 'INFO'
    LOG_FILE = 'result/optimized/annotation_optimization.log'

    # Output columns for final CSV
    OUTPUT_COLUMNS = [
        'thesis_code',
        'section',
        'Sentence',
        'gemini_metadiscourse_annotation',
        'claude_metadiscourse_annotation',
        'deepseek_metadiscourse_annotation',
        'Optimized_final_decision'
    ]

    # Annotation roles
    VALID_ROLES = ['Metadiscourse', 'Propositional', 'Borderline']

    # Confidence range
    MIN_CONFIDENCE = 1
    MAX_CONFIDENCE = 5

    @classmethod
    def validate_config(cls):
        """Validate configuration settings"""
        errors = []

        if not cls.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY not found in environment variables")

        if not os.path.exists(cls.BASE_DATA_PATH):
            errors.append(f"Base data path does not exist: {cls.BASE_DATA_PATH}")

        for model_dir in cls.MODEL_DIRECTORIES:
            full_path = os.path.join(cls.BASE_DATA_PATH, model_dir)
            if not os.path.exists(full_path):
                errors.append(f"Model directory does not exist: {full_path}")

        return errors

    @classmethod
    def get_project_root(cls):
        """Get the project root directory"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.dirname(current_dir)  # Go up one level from util/