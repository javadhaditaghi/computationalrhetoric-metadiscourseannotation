import pandas as pd
import json
import os
import sys
import re
from typing import Dict, List, Tuple, Optional
from openai import OpenAI
from collections import Counter
import logging
from unidecode import unidecode
import string
from json_repair import repair_json
import argparse

# Add the project root to the path to import config
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
sys.path.insert(0, project_root)

from util.config import Config
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnnotationOptimizer:
    def __init__(self, openai_api_key: str, model: str = "gpt-5",
                 context_words_before: int = 30, context_words_after: int = 30,
                 use_retrieval: bool = True):
        """
        Initialize the Annotation Optimizer

        Args:
            openai_api_key: OpenAI API key
            model: GPT model to use for optimization
            context_words_before: Number of words to include from context_before column
            context_words_after: Number of words to include from context_after column
            use_retrieval: Whether to use smart retrieval (True) or full framework (False)
        """
        self.client = OpenAI(api_key=openai_api_key)
        self.model = model
        self.context_words_before = context_words_before
        self.context_words_after = context_words_after
        self.use_retrieval = use_retrieval

        # Load prompts and framework
        self.optimization_prompt = self._load_optimization_prompt()

        if use_retrieval:
            self.minimal_adjudication_prompt = self._load_minimal_adjudication_prompt()
            self.annotation_framework = self._load_annotation_framework()
            self.retrieval_map = self._initialize_retrieval_map()
            logger.info("Initialized with RETRIEVAL mode (token-optimized)")
        else:
            logger.info("Initialized with FULL FRAMEWORK mode")

    def _load_optimization_prompt(self) -> str:
        """Load the optimization prompt from file"""
        prompt_path = os.path.join(project_root, Config.OPTIMIZATION_PROMPT_PATH_WITHOUT_BOUNDARY)

        if not os.path.exists(prompt_path):
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

    def _load_minimal_adjudication_prompt(self) -> str:
        """Load the minimal adjudication prompt from file"""
        # If you have a separate minimal prompt file, load it here
        # For now, using the optimization prompt as fallback
        try:
            prompt_path = os.path.join(project_root, Config.MINIMAL_ADJUDICATION_PROMPT_PATH_WITHOUT_BOUNDARY)
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except (FileNotFoundError, AttributeError):
            logger.warning("Minimal adjudication prompt not found, using optimization prompt")
            return self.optimization_prompt

    def _load_annotation_framework(self) -> Dict:
        """
        Load the adjunction protocol JSON file.

        UPDATED: Now loads adjunction_protocol.json instead of guidelines.json

        Returns:
            Dictionary containing the adjunction protocol
        """
        try:
            # Try to load from Config path first
            if hasattr(Config, 'ADJUNCTION_PROTOCOL_PATH_WITHOUT_BOUNDARY'):
                framework_path = os.path.join(project_root, Config.ADJUNCTION_PROTOCOL_PATH_WITHOUT_BOUNDARY)
            else:
                # Fallback: try common locations
                possible_paths = [
                    os.path.join(project_root, "without_boundary/adjunction_protocol_no_boundaries.json"),
                    os.path.join(project_root, "prompts/without_boundary/adjunction_protocol_no_boundaries.json"),
                    os.path.join(project_root, "config/without_boundary/adjunction_protocol_no_boundaries.json"),
                    os.path.join(project_root, "data/without_boundary/adjunction_protocol_no_boundaries.json"),
                ]

                framework_path = None
                for path in possible_paths:
                    if os.path.exists(path):
                        framework_path = path
                        break

                if not framework_path:
                    logger.warning("adjunction_protocol.json not found in common locations")
                    return {}

            with open(framework_path, 'r', encoding='utf-8') as f:
                protocol = json.load(f)
                logger.info(f"Loaded adjunction protocol from {framework_path}")
                return protocol

        except FileNotFoundError as e:
            logger.warning(f"Adjunction protocol file not found: {e}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in adjunction protocol: {e}")
            return {}
        except Exception as e:
            logger.error(f"Error loading adjunction protocol: {e}")
            return {}

    def _initialize_retrieval_map(self) -> Dict:
        """
        Initialize the retrieval map for adjunction_protocol.json structure.

        COMPLETELY REWRITTEN for adjunction_protocol.json paths.

        The adjunction_protocol.json structure:
        - task_name, role
        - input_specification
        - adjudication_procedure (stages 1-5)
            - stage_1_pre_adjudication_analysis
            - stage_2_evidence_based_weighting
            - stage_3_dimensional_adjudication
            - stage_4_confidence_synthesis
            - stage_5_gold_annotation_synthesis
        - output_format
        - critical_rules
        - final_instructions

        Returns:
            Dictionary mapping disagreement types to JSON paths
        """
        return {
            # =================================================================
            # TYPE A: REFLEXIVITY DISAGREEMENTS
            # =================================================================
            "Type A (Reflexivity)": {
                "json_paths": [
                    # Core reflexivity principle and checks
                    "adjudication_procedure.stage_1_pre_adjudication_analysis.step_1_3_theoretical_framework_alignment.core_principles_to_check.reflexivity_principle",
                    # D1 reflexivity adjudication logic
                    "adjudication_procedure.stage_3_dimensional_adjudication.dimension_1_observable_realization.step_1_reflexivity_adjudication",
                    # Disagreement type definition
                    "adjudication_procedure.stage_1_pre_adjudication_analysis.step_1_1_disagreement_pattern_identification.disagreement_types.type_a_reflexivity",
                    # Tier 1 reflexivity criterion
                    "adjudication_procedure.stage_2_evidence_based_weighting.tier_1_foundational_correctness.criteria.reflexivity_correctly_identified"
                ],
                "description": "Reflexivity principle, adjudication logic, and Tier 1 disqualification criteria"
            },

            # =================================================================
            # TYPE C: SCOPE DISAGREEMENTS - MEDIUM PRIORITY
            # =================================================================
            "Type C (Scope)": {
                "json_paths": [
                    # Full D2 scope adjudication (includes all steps)
                    "adjudication_procedure.stage_3_dimensional_adjudication.dimension_2_functional_scope",
                    # Scope-function principle from framework alignment
                    "adjudication_procedure.stage_1_pre_adjudication_analysis.step_1_3_theoretical_framework_alignment.core_principles_to_check.scope_function_principle",
                    # Disagreement type definition
                    "adjudication_procedure.stage_1_pre_adjudication_analysis.step_1_1_disagreement_pattern_identification.disagreement_types.type_c_scope"
                ],
                "description": "Scope adjudication steps, scope-function principle, and MICRO/MESO/MACRO tests"
            },

            # =================================================================
            # TYPE D: LEVEL 1 CLASSIFICATION - HIGH PRIORITY
            # =================================================================
            "Type D (Level 1)": {
                "json_paths": [
                    # Full Level 1 adjudication
                    "adjudication_procedure.stage_3_dimensional_adjudication.dimension_3_metadiscourse_classification.level_1_md_vs_propositional_vs_borderline",
                    # Text-context-discourse principle
                    "adjudication_procedure.stage_1_pre_adjudication_analysis.step_1_3_theoretical_framework_alignment.core_principles_to_check.text_context_discourse_principle",
                    # Disagreement type definition
                    "adjudication_procedure.stage_1_pre_adjudication_analysis.step_1_1_disagreement_pattern_identification.disagreement_types.type_d_level1",
                    # Tier 1 framework alignment
                    "adjudication_procedure.stage_2_evidence_based_weighting.tier_1_foundational_correctness.criteria.theoretical_framework_alignment"
                ],
                "description": "Level 1 (MD/Propositional/Borderline) adjudication with removal test logic"
            },

            # =================================================================
            # TYPE E: INTERACTIVE TYPE DISAGREEMENTS
            # =================================================================
            "Type E (Type - Interactive)": {
                "json_paths": [
                    # Level 2 adjudication
                    "adjudication_procedure.stage_3_dimensional_adjudication.dimension_3_metadiscourse_classification.level_2_interactive_vs_interactional",
                    # Level 3 adjudication (specific types)
                    "adjudication_procedure.stage_3_dimensional_adjudication.dimension_3_metadiscourse_classification.level_3_specific_type",
                    # Disagreement type definition
                    "adjudication_procedure.stage_1_pre_adjudication_analysis.step_1_1_disagreement_pattern_identification.disagreement_types.type_e_type"
                ],
                "description": "Interactive/Interactional distinction and Level 3 specific type resolution"
            },

            # =================================================================
            # TYPE E: INTERACTIONAL TYPE DISAGREEMENTS
            # =================================================================
            "Type E (Type - Interactional)": {
                "json_paths": [
                    # Same as Interactive - covers both categories
                    "adjudication_procedure.stage_3_dimensional_adjudication.dimension_3_metadiscourse_classification.level_2_interactive_vs_interactional",
                    "adjudication_procedure.stage_3_dimensional_adjudication.dimension_3_metadiscourse_classification.level_3_specific_type",
                    "adjudication_procedure.stage_1_pre_adjudication_analysis.step_1_1_disagreement_pattern_identification.disagreement_types.type_e_type"
                ],
                "description": "Interactive/Interactional distinction and Level 3 specific type resolution"
            },

            # =================================================================
            # TYPE F: LEVEL 1 BORDERLINE
            # =================================================================
            "Type F (Borderline - Level 1)": {
                "json_paths": [
                    # Borderline system principle
                    "adjudication_procedure.stage_1_pre_adjudication_analysis.step_1_3_theoretical_framework_alignment.core_principles_to_check.borderline_system_principle",
                    # Level 1 borderline adjudication (genuine vs pseudo)
                    "adjudication_procedure.stage_3_dimensional_adjudication.dimension_3_metadiscourse_classification.level_1_md_vs_propositional_vs_borderline.step_3_borderline_adjudication",
                    # Disagreement type definition
                    "adjudication_procedure.stage_1_pre_adjudication_analysis.step_1_1_disagreement_pattern_identification.disagreement_types.type_f_borderline"
                ],
                "description": "Level 1 borderline (MD/Propositional dual functionality) adjudication"
            },

            # =================================================================
            # TYPE F: LEVEL 2 BORDERLINE
            # =================================================================
            "Type F (Borderline - Level 2)": {
                "json_paths": [
                    # Borderline system principle
                    "adjudication_procedure.stage_1_pre_adjudication_analysis.step_1_3_theoretical_framework_alignment.core_principles_to_check.borderline_system_principle",
                    # Level 2 borderline check in Level 3 resolution
                    "adjudication_procedure.stage_3_dimensional_adjudication.dimension_3_metadiscourse_classification.level_3_specific_type.step_3_resolve_type_disagreement.priority_1_check_level_2_borderline",
                    # Level 3 synthesis for Level 2 borderline
                    "adjudication_procedure.stage_3_dimensional_adjudication.dimension_3_metadiscourse_classification.level_3_specific_type.step_4_synthesize_level_3.if_level_2_borderline",
                    # Disagreement type definition
                    "adjudication_procedure.stage_1_pre_adjudication_analysis.step_1_1_disagreement_pattern_identification.disagreement_types.type_f_borderline"
                ],
                "description": "Level 2 borderline (multiple MD types) with PRIMARY/SECONDARY/TERTIARY ranking"
            },

            # =================================================================
            # TYPE G: CONFIDENCE DISAGREEMENTS - LOW PRIORITY (NEW)
            # =================================================================
            "Type G (Confidence)": {
                "json_paths": [
                    # Full confidence synthesis stage
                    "adjudication_procedure.stage_4_confidence_synthesis",
                    # Tier 3 confidence calibration
                    "adjudication_procedure.stage_2_evidence_based_weighting.tier_3_confidence_calibration",
                    # Disagreement type definition
                    "adjudication_procedure.stage_1_pre_adjudication_analysis.step_1_1_disagreement_pattern_identification.disagreement_types.type_g_confidence"
                ],
                "description": "Confidence calibration when only confidence ratings differ"
            },

            # =================================================================
            # GENERAL: EVIDENCE WEIGHTING (Always included for disagreements)
            # =================================================================
            "Evidence Weighting": {
                "json_paths": [
                    # Tier 1 foundational correctness (disqualifying criteria)
                    "adjudication_procedure.stage_2_evidence_based_weighting.tier_1_foundational_correctness",
                    # Tier 2 reasoning quality
                    "adjudication_procedure.stage_2_evidence_based_weighting.tier_2_reasoning_quality"
                ],
                "description": "Tier 1 and Tier 2 evidence-based weighting criteria"
            },

            # =================================================================
            # GENERAL: ROOT CAUSE ANALYSIS
            # =================================================================
            "Root Cause Analysis": {
                "json_paths": [
                    "adjudication_procedure.stage_1_pre_adjudication_analysis.step_1_2_root_cause_identification"
                ],
                "description": "Root cause identification for why disagreements occurred"
            },

            # =================================================================
            # GENERAL: GOLD ANNOTATION SYNTHESIS
            # =================================================================
            "Gold Synthesis": {
                "json_paths": [
                    "adjudication_procedure.stage_5_gold_annotation_synthesis.synthesis_principles",
                    "adjudication_procedure.stage_5_gold_annotation_synthesis.required_metadata"
                ],
                "description": "Final gold annotation synthesis principles and required metadata"
            }
        }

    def _retrieve_framework_sections(self, disagreement_types: List[str]) -> Dict:
        """
        Retrieve relevant framework sections based on disagreement types.

        ENHANCED VERSION:
        - Automatically includes Evidence Weighting for any disagreement
        - Better section naming from paths
        - More robust error handling

        Args:
            disagreement_types: List of disagreement type keys

        Returns:
            Dictionary of retrieved framework sections
        """
        retrieved_sections = {}

        if not self.annotation_framework:
            logger.warning("Annotation framework not loaded, cannot retrieve sections")
            return retrieved_sections

        # Always include evidence weighting if there are any disagreements
        types_to_retrieve = list(disagreement_types) if disagreement_types else []
        if types_to_retrieve and "Evidence Weighting" not in types_to_retrieve:
            types_to_retrieve.append("Evidence Weighting")

        for dtype in types_to_retrieve:
            if dtype not in self.retrieval_map:
                logger.debug(f"Disagreement type '{dtype}' not in retrieval map")
                continue

            config = self.retrieval_map[dtype]

            for json_path in config["json_paths"]:
                try:
                    section_content = self._get_nested_value(self.annotation_framework, json_path)

                    # Create a meaningful, unique section name from the path
                    path_parts = json_path.split('.')

                    # Extract meaningful parts (skip generic prefixes)
                    meaningful_parts = []
                    for part in path_parts:
                        # Skip stage prefixes but keep step info
                        if part.startswith('stage_') and len(part) > 8:
                            meaningful_parts.append(part.split('_', 2)[-1] if '_' in part else part)
                        elif part.startswith('step_'):
                            continue  # Skip step numbers
                        elif part not in ['adjudication_procedure', 'core_principles_to_check', 'criteria']:
                            meaningful_parts.append(part)

                    # Use last 2-3 meaningful parts
                    section_name = '_'.join(meaningful_parts[-3:]) if meaningful_parts else path_parts[-1]

                    # Ensure uniqueness by adding dtype prefix if needed
                    if section_name in retrieved_sections:
                        section_name = f"{dtype.replace(' ', '_').replace('(', '').replace(')', '')}_{section_name}"

                    # Only add if not already present
                    if section_name not in retrieved_sections:
                        retrieved_sections[section_name] = section_content
                        logger.debug(f"Retrieved '{section_name}' from '{json_path}'")

                except (KeyError, IndexError, TypeError) as e:
                    logger.warning(f"Could not retrieve path '{json_path}': {e}")
                    continue
                except Exception as e:
                    logger.error(f"Unexpected error retrieving '{json_path}': {e}")
                    continue

        logger.info(
            f"Retrieved {len(retrieved_sections)} framework sections for {len(types_to_retrieve)} disagreement types")
        return retrieved_sections

    def _get_nested_value(self, data: Dict, path: str) -> any:
        """
        Extract value from nested JSON using dot notation path

        Args:
            data: JSON dictionary
            path: Dot-separated path (e.g., "theoretical_framework.core_principle")

        Returns:
            Value at the specified path
        """
        keys = path.split('.')
        value = data

        for key in keys:
            if key.isdigit():  # Array index
                value = value[int(key)]
            else:  # Dictionary key
                value = value[key]

        return value

    def _normalize_classification(self, value: str) -> str:
        """
        Normalize classification values (scope, level 1/2/3)

        Handles:
        - Case insensitivity (uppercase for consistency)
        - Whitespace normalization
        - Hyphen/underscore normalization
        - Leading/trailing punctuation

        Args:
            value: Classification value to normalize

        Returns:
            Normalized classification string (uppercase, underscores)

        Examples:
            "METADISCOURSE" → "METADISCOURSE"
            "Metadiscourse" → "METADISCOURSE"
            "MICRO-SCOPE" → "MICRO_SCOPE"
            "Frame Marker" → "FRAME_MARKER"
        """
        if not value or pd.isna(value):
            return ""

        # Handle Unicode
        normalized = unidecode(value)

        # Uppercase
        normalized = normalized.upper()

        # Strip punctuation and whitespace
        normalized = normalized.strip(string.punctuation + string.whitespace)

        # Replace spaces and hyphens with underscores
        normalized = re.sub(r'[\s\-]+', '_', normalized)

        # Collapse multiple underscores
        normalized = re.sub(r'_+', '_', normalized)

        # Strip leading/trailing underscores
        normalized = normalized.strip('_')

        return normalized

    def _get_expression(self, ann: Dict, expr_number: int = 1) -> Optional[str]:
        """
        Flexibly get expression from annotation, handling both 'expression' and 'expression_1' formats
        Returns None if expression is null/None/"null"/"None"

        Args:
            ann: Annotation dictionary
            expr_number: Which expression to get (1, 2, or 3)

        Returns:
            Expression string or None
        """
        if not ann:
            return None

        # Try different possible keys for the expression
        possible_keys = [
            f"expression_{expr_number}",  # e.g., "expression_1"
            f"expression{expr_number}",  # e.g., "expression1" (no underscore)
            "expression" if expr_number == 1 else None,  # Only for first expression
        ]

        for key in possible_keys:
            if key and key in ann:
                value = ann.get(key)

                # Check if value is valid (not null/None/etc.)
                if self._is_valid_value(value):
                    return value

        return None

    def _get_analysis(self, ann: Dict, analysis_number: int = 1) -> Optional[Dict]:
        """
        Flexibly get analysis from annotation, handling both 'analysis' and 'analysis_1' formats

        Args:
            ann: Annotation dictionary
            analysis_number: Which analysis to get (1, 2, or 3)

        Returns:
            Analysis dictionary or None
        """
        if not ann:
            return None

        # Try different possible keys for the analysis
        possible_keys = [
            f"analysis_{analysis_number}",  # e.g., "analysis_1"
            f"analysis{analysis_number}",  # e.g., "analysis1" (no underscore)
            "analysis" if analysis_number == 1 else None,  # Only for first analysis
        ]

        for key in possible_keys:
            if key and key in ann:
                return ann.get(key)

        return None

    def _count_expressions(self, ann: Dict) -> int:
        """
        Count how many NON-NULL expressions are in the annotation
        Handles: None, null, "null", "None", empty strings

        Args:
            ann: Annotation dictionary

        Returns:
            Number of non-null expressions (1, 2, or 3)
        """
        if not ann:
            return 0

        count = 0
        for i in range(1, 4):  # Check for up to 3 expressions
            expr = self._get_expression(ann, i)

            # Check if expression is actually valid (not null/None/empty)
            if expr and self._is_valid_value(expr):
                count += 1

        return count

    def _is_valid_value(self, value) -> bool:
        """
        Check if a value is valid (not null, None, "null", "None", or empty)

        Args:
            value: Value to check

        Returns:
            True if valid, False otherwise
        """
        if value is None:
            return False

        if pd.isna(value):
            return False

        # Convert to string and check
        str_value = str(value).strip().lower()

        if str_value in ['', 'none', 'null', 'nan']:
            return False

        return True

    def _get_analysis_value(self, ann: Dict, analysis_number: int, *keys) -> any:
        """
        Get a nested value from a specific analysis, with flexible key handling

        Args:
            ann: Annotation dictionary
            analysis_number: Which analysis (1, 2, or 3)
            *keys: Path to the value within the analysis

        Returns:
            Value at the specified path or None

        Example:
            _get_analysis_value(ann, 1, "dimension_2_functional_scope", "classification")
        """
        analysis = self._get_analysis(ann, analysis_number)

        if not analysis:
            return None

        try:
            value = analysis
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return None

    def _detect_disagreement_types(self, claude_ann: Dict, gemini_ann: Dict,
                                   deepseek_ann: Dict) -> List[str]:
        """
        Detect disagreement types between the three annotations.

        UPDATED: Now includes Type G (Confidence) detection from adjunction_protocol.

        Args:
            claude_ann: Claude's annotation
            gemini_ann: Gemini's annotation
            deepseek_ann: DeepSeek's annotation

        Returns:
            List of disagreement type keys
        """
        disagreement_types = []
        annotations = [ann for ann in [claude_ann, gemini_ann, deepseek_ann] if ann]

        if len(annotations) < 2:
            return disagreement_types

        # Determine maximum number of expressions across all models
        max_expressions = max([self._count_expressions(ann) for ann in annotations])

        # =====================================================================
        # TYPE A: Reflexivity disagreement
        # =====================================================================
        reflexivity_values = [
            self._get_analysis_value(ann, 1, "dimension_1_observable_realization", "reflexivity", "is_reflexive")
            for ann in annotations
        ]
        valid_reflexivity = [v for v in reflexivity_values if v is not None]
        if len(set(valid_reflexivity)) > 1:
            disagreement_types.append("Type A (Reflexivity)")

        # =====================================================================
        # TYPE C, D, E: Check scope and classification for EACH analysis
        # =====================================================================
        for analysis_num in range(1, max_expressions + 1):
            # TYPE C: Scope disagreement
            scopes = [
                self._get_analysis_value(ann, analysis_num, "dimension_2_functional_scope", "classification")
                for ann in annotations
            ]
            normalized_scopes = [self._normalize_classification(s) for s in scopes]

            non_empty_scopes = [s for s in normalized_scopes if s]
            if len(non_empty_scopes) >= 2 and len(set(non_empty_scopes)) > 1:
                if "Type C (Scope)" not in disagreement_types:
                    disagreement_types.append("Type C (Scope)")
                logger.debug(f"Scope disagreement detected in analysis_{analysis_num}")

            # TYPE D: Level 1 classification disagreement
            level_1 = [
                self._get_analysis_value(ann, analysis_num, "dimension_3_metadiscourse_classification",
                                         "level_1_primary_classification")
                for ann in annotations
            ]
            normalized_level_1 = [self._normalize_classification(l) for l in level_1]

            non_empty_level_1 = [l for l in normalized_level_1 if l]
            if len(non_empty_level_1) >= 2 and len(set(non_empty_level_1)) > 1:
                if "Type D (Level 1)" not in disagreement_types:
                    disagreement_types.append("Type D (Level 1)")
                logger.debug(f"Level 1 disagreement detected in analysis_{analysis_num}")

            # TYPE E: Type disagreement (Level 2 and Level 3)
            level_2 = [
                self._get_analysis_value(ann, analysis_num, "dimension_3_metadiscourse_classification",
                                         "level_2_functional_category")
                for ann in annotations
            ]
            normalized_level_2 = [self._normalize_classification(l) for l in level_2]
            level_2_unique = set([l for l in normalized_level_2 if l])

            if len(level_2_unique) > 1:
                if "INTERACTIVE" in level_2_unique:
                    if "Type E (Type - Interactive)" not in disagreement_types:
                        disagreement_types.append("Type E (Type - Interactive)")
                if "INTERACTIONAL" in level_2_unique:
                    if "Type E (Type - Interactional)" not in disagreement_types:
                        disagreement_types.append("Type E (Type - Interactional)")
            else:
                # Check Level 3 types if Level 2 agrees
                level_3 = [
                    self._get_analysis_value(ann, analysis_num, "dimension_3_metadiscourse_classification",
                                             "level_3_specific_type")
                    for ann in annotations
                ]
                normalized_level_3 = [self._normalize_classification(l) for l in level_3]

                non_empty_level_3 = [l for l in normalized_level_3 if l]
                if len(non_empty_level_3) >= 2 and len(set(non_empty_level_3)) > 1:
                    # Determine which category based on Level 2
                    first_valid_l2 = next((l for l in normalized_level_2 if l), None)
                    if first_valid_l2 == "INTERACTIVE":
                        if "Type E (Type - Interactive)" not in disagreement_types:
                            disagreement_types.append("Type E (Type - Interactive)")
                    elif first_valid_l2 == "INTERACTIONAL":
                        if "Type E (Type - Interactional)" not in disagreement_types:
                            disagreement_types.append("Type E (Type - Interactional)")

        # =====================================================================
        # TYPE F: Borderline disagreement (check first analysis)
        # =====================================================================
        borderline_l1 = [
            self._get_analysis_value(ann, 1, "borderline_classification",
                                     "level_1_borderline_md_propositional", "is_level_1_borderline")
            for ann in annotations
        ]
        valid_bl1 = [bl for bl in borderline_l1 if bl is not None]
        if any(borderline_l1) or (len(set(valid_bl1)) > 1):
            disagreement_types.append("Type F (Borderline - Level 1)")

        borderline_l2 = [
            self._get_analysis_value(ann, 1, "borderline_classification",
                                     "level_2_borderline_md_features", "is_level_2_borderline")
            for ann in annotations
        ]
        valid_bl2 = [bl for bl in borderline_l2 if bl is not None]
        if any(borderline_l2) or (len(set(valid_bl2)) > 1):
            disagreement_types.append("Type F (Borderline - Level 2)")

        # =====================================================================
        # TYPE G: Confidence disagreement (NEW - from adjunction_protocol)
        # Only flag if classifications agree but confidence differs significantly
        # =====================================================================
        if not disagreement_types:
            confidence_values = [
                self._get_analysis_value(ann, 1, "confidence_ratings", "overall_confidence")
                for ann in annotations
            ]
            valid_confidences = [c for c in confidence_values if c is not None and isinstance(c, (int, float))]

            if len(valid_confidences) >= 2:
                conf_range = max(valid_confidences) - min(valid_confidences)
                if conf_range > 1:  # More than 1 point difference
                    disagreement_types.append("Type G (Confidence)")
                    logger.debug(f"Confidence disagreement detected: range={conf_range}")

        return disagreement_types

    def test_normalization_with_real_data(self, max_cases: int = 10):
        """
        Test normalization with actual annotation data from the three models
        Shows before/after normalization and whether disagreements are found

        Args:
            max_cases: Maximum number of cases to test
        """
        print("\n" + "=" * 80)
        print("TESTING NORMALIZATION WITH REAL ANNOTATION DATA")
        print("=" * 80 + "\n")

        try:
            # Load the CSV files
            dataframes = self.load_csv_files()

            # Merge annotations
            merged_df = self.merge_annotations(dataframes, max_rows=max_cases)

            print(f"Testing {len(merged_df)} real annotation cases...\n")
            print("=" * 80 + "\n")

            # Track statistics
            stats = {
                'total_cases': 0,
                'scope_disagreements_before': 0,
                'scope_disagreements_after': 0,
                'level1_disagreements_before': 0,
                'level1_disagreements_after': 0,
                'level2_disagreements_before': 0,
                'level2_disagreements_after': 0,
                'false_positives_prevented': 0
            }

            for idx, row in merged_df.iterrows():
                stats['total_cases'] += 1

                print(f"{'=' * 80}")
                print(f"CASE {idx + 1}")
                print(f"{'=' * 80}")
                print(f"Sentence: {row['sentence'][:100]}...")
                print()

                # Parse annotations
                claude_ann = self.parse_json_annotation(row['claude_metadiscourse_annotation'])
                claude_ann = self.validate_and_clean_annotation(claude_ann) if claude_ann else None

                gemini_ann = self.parse_json_annotation(row['gemini_metadiscourse_annotation'])
                gemini_ann = self.validate_and_clean_annotation(gemini_ann) if gemini_ann else None

                deepseek_ann = self.parse_json_annotation(row['deepseek_metadiscourse_annotation'])
                deepseek_ann = self.validate_and_clean_annotation(deepseek_ann) if deepseek_ann else None

                annotations = [ann for ann in [claude_ann, gemini_ann, deepseek_ann] if ann]

                if len(annotations) < 2:
                    print("⚠️  Skipping: Less than 2 valid annotations\n")
                    continue

                # Helper function
                def get_value(ann, *keys):
                    try:
                        value = ann
                        for key in keys:
                            value = value[key]
                        return value
                    except (KeyError, TypeError):
                        return None

                # Get expression counts
                expression_counts = [self._count_expressions(ann) for ann in annotations]
                max_expr_count = max(expression_counts) if expression_counts else 1
                model_names = ["Claude", "Gemini", "DeepSeek"][:len(annotations)]

                # TEST SCOPE, LEVEL 1, AND LEVEL 2 FOR EACH ANALYSIS
                for analysis_num in range(1, max_expr_count + 1):
                    # Check if at least 2 models have this analysis
                    analysis_exists = [
                        self._get_analysis(ann, analysis_num) is not None
                        for ann in annotations
                    ]

                    if sum(analysis_exists) < 2:
                        continue  # Skip if less than 2 models have this analysis

                    # Show header for this analysis
                    if max_expr_count > 1:
                        print(f"{'=' * 80}")
                        print(f"ANALYSIS {analysis_num} (for expression_{analysis_num}):")
                        print(f"{'=' * 80}")

                    # TEST SCOPE NORMALIZATION
                    print("📏 SCOPE COMPARISON:")
                    print("-" * 80)

                    scopes = [
                        self._get_analysis_value(ann, analysis_num, "dimension_2_functional_scope", "classification")
                        for ann in annotations
                    ]

                    if any(scopes):
                        print("Original values:")
                        for name, scope in zip(model_names, scopes):
                            if scope:
                                print(f"  {name:10s}: '{scope}'")

                        normalized_scopes = [self._normalize_classification(s) for s in scopes]

                        print("\nNormalized values:")
                        for name, orig, norm in zip(model_names, scopes, normalized_scopes):
                            if norm:
                                changed = " ✓ (changed)" if orig != norm else ""
                                print(f"  {name:10s}: '{norm}'{changed}")

                        unique_before = set([s for s in scopes if s])
                        unique_after = set([s for s in normalized_scopes if s])

                        print("\nResult:")
                        if len(unique_before) > 1 and len(unique_after) == 1:
                            print("  ✅ FALSE POSITIVE PREVENTED - Normalization fixed formatting")
                            stats['false_positives_prevented'] += 1
                            if analysis_num == 1:
                                stats['scope_disagreements_before'] += 1
                        elif len(unique_before) > 1 and len(unique_after) > 1:
                            print("  ⚠️  REAL DISAGREEMENT - Models chose different scopes")
                            if analysis_num == 1:
                                stats['scope_disagreements_before'] += 1
                                stats['scope_disagreements_after'] += 1
                        else:
                            print("  ✅ ALL AGREE")
                    else:
                        print("  ℹ️  No scope data available")

                    print()

                    # TEST LEVEL 1 CLASSIFICATION NORMALIZATION
                    print("🏷️  LEVEL 1 CLASSIFICATION COMPARISON:")
                    print("-" * 80)

                    level_1 = [
                        self._get_analysis_value(ann, analysis_num, "dimension_3_metadiscourse_classification",
                                                 "level_1_primary_classification")
                        for ann in annotations
                    ]

                    if any(level_1):
                        print("Original values:")
                        for name, l1 in zip(model_names, level_1):
                            if l1:
                                print(f"  {name:10s}: '{l1}'")

                        normalized_level_1 = [self._normalize_classification(l) for l in level_1]

                        print("\nNormalized values:")
                        for name, orig, norm in zip(model_names, level_1, normalized_level_1):
                            if norm:
                                changed = " ✓ (changed)" if orig != norm else ""
                                print(f"  {name:10s}: '{norm}'{changed}")

                        unique_before = set([l for l in level_1 if l])
                        unique_after = set([l for l in normalized_level_1 if l])

                        print("\nResult:")
                        if len(unique_before) > 1 and len(unique_after) == 1:
                            print("  ✅ FALSE POSITIVE PREVENTED - Normalization fixed formatting")
                            stats['false_positives_prevented'] += 1
                            if analysis_num == 1:
                                stats['level1_disagreements_before'] += 1
                        elif len(unique_before) > 1 and len(unique_after) > 1:
                            print("  ⚠️  REAL DISAGREEMENT - Models classified differently")
                            if analysis_num == 1:
                                stats['level1_disagreements_before'] += 1
                                stats['level1_disagreements_after'] += 1
                        else:
                            print("  ✅ ALL AGREE")
                    else:
                        print("  ℹ️  No Level 1 data available")

                    print()

                    # TEST LEVEL 2 CLASSIFICATION NORMALIZATION
                    print("🔤 LEVEL 2 CLASSIFICATION COMPARISON:")
                    print("-" * 80)

                    level_2 = [
                        self._get_analysis_value(ann, analysis_num, "dimension_3_metadiscourse_classification",
                                                 "level_2_functional_category")
                        for ann in annotations
                    ]

                    if any(level_2):
                        print("Original values:")
                        for name, l2 in zip(model_names, level_2):
                            if l2:
                                print(f"  {name:10s}: '{l2}'")

                        normalized_level_2 = [self._normalize_classification(l) for l in level_2]

                        print("\nNormalized values:")
                        for name, orig, norm in zip(model_names, level_2, normalized_level_2):
                            if norm:
                                changed = " ✓ (changed)" if orig != norm else ""
                                print(f"  {name:10s}: '{norm}'{changed}")

                        unique_before = set([l for l in level_2 if l])
                        unique_after = set([l for l in normalized_level_2 if l])

                        print("\nResult:")
                        if len(unique_before) > 1 and len(unique_after) == 1:
                            print("  ✅ FALSE POSITIVE PREVENTED - Normalization fixed formatting")
                            stats['false_positives_prevented'] += 1
                            if analysis_num == 1:
                                stats['level2_disagreements_before'] += 1
                        elif len(unique_before) > 1 and len(unique_after) > 1:
                            print("  ⚠️  REAL DISAGREEMENT - Models classified differently")
                            if analysis_num == 1:
                                stats['level2_disagreements_before'] += 1
                                stats['level2_disagreements_after'] += 1
                        else:
                            print("  ✅ ALL AGREE")
                    else:
                        print("  ℹ️  No Level 2 data available")

                    print()

                print("=" * 80 + "\n")

            # Print summary statistics
            print("=" * 80)
            print("SUMMARY STATISTICS")
            print("=" * 80)
            print(f"\nTotal cases tested: {stats['total_cases']}")

            print(f"\n📏 SCOPE DISAGREEMENTS:")
            print(f"   Before normalization: {stats['scope_disagreements_before']}")
            print(f"   After normalization:  {stats['scope_disagreements_after']}")
            print(
                f"   False positives prevented: {stats['scope_disagreements_before'] - stats['scope_disagreements_after']}")

            print(f"\n🏷️  LEVEL 1 DISAGREEMENTS:")
            print(f"   Before normalization: {stats['level1_disagreements_before']}")
            print(f"   After normalization:  {stats['level1_disagreements_after']}")
            print(
                f"   False positives prevented: {stats['level1_disagreements_before'] - stats['level1_disagreements_after']}")

            print(f"\n🔤 LEVEL 2 DISAGREEMENTS:")
            print(f"   Before normalization: {stats['level2_disagreements_before']}")
            print(f"   After normalization:  {stats['level2_disagreements_after']}")
            print(
                f"   False positives prevented: {stats['level2_disagreements_before'] - stats['level2_disagreements_after']}")

            print(f"\n✅ TOTAL FALSE POSITIVES PREVENTED: {stats['false_positives_prevented']}")

            if stats['false_positives_prevented'] > 0:
                total_disagreements_before = (stats['scope_disagreements_before'] +
                                              stats['level1_disagreements_before'] +
                                              stats['level2_disagreements_before'])
                if total_disagreements_before > 0:
                    prevention_rate = (stats['false_positives_prevented'] / total_disagreements_before) * 100
                    print(f"   Prevention rate: {prevention_rate:.1f}%")

            print("\n" + "=" * 80 + "\n")

            return stats

        except Exception as e:
            print(f"❌ Error during real data testing: {e}")
            logger.error(f"Real data test failed: {e}", exc_info=True)
            return None

    def diagnose_annotation_structure(self, max_samples: int = 3):
        """
        Diagnose the structure of annotations to understand key variations
        """
        print("\n" + "=" * 80)
        print("ANNOTATION STRUCTURE DIAGNOSIS")
        print("=" * 80 + "\n")

        try:
            dataframes = self.load_csv_files()

            for model_name, df in dataframes.items():
                print(f"\n{'=' * 80}")
                print(f"MODEL: {model_name.upper()}")
                print(f"{'=' * 80}\n")

                sample_annotations = df['metadiscourse_annotation'].dropna().head(max_samples)

                for idx, ann_str in enumerate(sample_annotations, 1):
                    print(f"--- Sample {idx} ---\n")

                    try:
                        ann = self.parse_json_annotation(ann_str)

                        if ann:
                            # Check expression keys
                            expr_keys = [k for k in ann.keys() if 'expression' in k.lower()]
                            print(f"Expression keys found: {expr_keys}")

                            # Show all expressions and their validity
                            valid_count = 0
                            for key in expr_keys:
                                value = ann[key]
                                is_valid = self._is_valid_value(value)

                                if value is not None:
                                    validity_marker = "✓ valid" if is_valid else "✗ null/empty"
                                    print(f"  {key}: '{value}' {validity_marker}")
                                    if is_valid:
                                        valid_count += 1
                                else:
                                    print(f"  {key}: None ✗ null/empty")

                            print(f"\nValid expressions: {valid_count}/{len(expr_keys)}")

                            # Verify with _count_expressions()
                            counted = self._count_expressions(ann)
                            match = "✓" if counted == valid_count else "✗ MISMATCH"
                            print(f"_count_expressions() returns: {counted} {match}")

                            # Check analysis keys
                            analysis_keys = [k for k in ann.keys() if 'analysis' in k.lower()]
                            print(f"\nAnalysis keys found: {analysis_keys}")

                            # Show structure of each analysis found
                            for analysis_key in analysis_keys:
                                analysis = ann[analysis_key]
                                if isinstance(analysis, dict):
                                    print(f"\nStructure of '{analysis_key}':")
                                    for key in analysis.keys():
                                        print(f"  - {key}")

                            print()
                        else:
                            print("❌ Failed to parse annotation\n")

                    except Exception as e:
                        print(f"❌ Error: {e}\n")

            print("=" * 80 + "\n")

        except Exception as e:
            print(f"❌ Diagnosis failed: {e}")

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
        Includes automatic repair for malformed JSON

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

            # Try to parse directly first
            try:
                return json.loads(json_content)
            except json.JSONDecodeError as e:
                # Attempt repair using json-repair library
                logger.warning(f"Initial parse failed: {e.msg}. Attempting repair...")

                try:
                    repaired = repair_json(json_content)
                    parsed = json.loads(repaired)
                    logger.info("✓ JSON repair successful!")
                    return parsed
                except Exception as e2:
                    logger.error(f"✗ JSON repair failed: {e2}")
                    logger.debug(f"Problematic content: {json_str[:200]}...")
                    return None

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
        first_brace = text.find('{')
        last_brace = text.rfind('}')

        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            potential_json = text[first_brace:last_brace + 1]
            # Basic validation - count braces to ensure it's complete
            open_braces = potential_json.count('{')
            close_braces = potential_json.count('}')
            if open_braces == close_braces:
                return potential_json

        # Pattern 5: Remove common prefixes/suffixes
        cleaned_text = text

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

        cleaned_text = re.sub(r'^```json?\s*\n?', '', cleaned_text, flags=re.IGNORECASE | re.MULTILINE)
        cleaned_text = re.sub(r'\n?```\s*$', '', cleaned_text, flags=re.MULTILINE)

        cleaned_text = cleaned_text.strip()
        if cleaned_text.startswith('{') and cleaned_text.endswith('}'):
            return cleaned_text

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

        # For the new format, just return as-is if it has the expected structure
        # The annotation should follow the metadiscourse framework structure
        return annotation

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

                try:
                    extracted_json = self._extract_json_from_markdown(str(annotation))
                    logger.info(f"Extracted JSON: {extracted_json[:200]}...")

                    parsed = json.loads(extracted_json)
                    logger.info(f"✓ Successfully parsed")

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
        Merges based on thesis_code, section, and sentence only

        Note: The 'expression' column from CSV is NOT used for merging.
        Instead, expression_1, expression_2, expression_3 are extracted from
        inside each model's JSON annotation.

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
            # Select columns: thesis_code, section, sentence, annotation
            # NOTE: We do NOT include 'expression' column - it's not reliable for merging
            model_df = dataframes[model][
                ['thesis_code', 'section', 'sentence', 'metadiscourse_annotation']
            ].copy()

            # Apply same row limit to other models for consistency
            if max_rows is not None:
                model_df = model_df.head(max_rows)

            model_df = model_df.rename(columns={'metadiscourse_annotation': f'{model}_metadiscourse_annotation'})

            # Merge on thesis_code, section, sentence ONLY
            # This ensures all three models' annotations for the SAME SENTENCE are in one row
            base_df = base_df.merge(
                model_df,
                on=['thesis_code', 'section', 'sentence'],
                how='outer'
            )

        # Select required columns - NO 'expression' column
        required_columns = [
            'thesis_code',
            'section',
            'sentence',
            'claude_metadiscourse_annotation',
            'gemini_metadiscourse_annotation',
            'deepseek_metadiscourse_annotation'
        ]

        # Add context columns if they exist in the dataset
        if 'context_before' in base_df.columns:
            required_columns.append('context_before')
        if 'context_after' in base_df.columns:
            required_columns.append('context_after')

        return base_df[required_columns]

    def optimize_annotation(self, claude_ann: Dict, gemini_ann: Dict, deepseek_ann: Dict,
                            sentence: str, context_before: str = "",
                            context_after: str = "", row_data: Dict = None) -> Dict:
        """
        Use GPT to optimize annotations with smart retrieval or full framework

        Args:
            claude_ann: Claude's annotation
            gemini_ann: Gemini's annotation
            deepseek_ann: DeepSeek's annotation
            sentence: The sentence being annotated
            context_before: Context before the sentence
            context_after: Context after the sentence

        Returns:
            Optimized annotation dictionary
        """
        # Extract limited context
        limited_context_before = self._extract_limited_context(context_before, self.context_words_before)
        limited_context_after = self._extract_limited_context(context_after, self.context_words_after)

        # Build prompt based on retrieval mode
        if self.use_retrieval and hasattr(self, 'annotation_framework') and self.annotation_framework:
            # RETRIEVAL MODE: Minimal prompt + retrieved sections
            disagreement_types = self._detect_disagreement_types(claude_ann, gemini_ann, deepseek_ann)
            retrieved_framework = self._retrieve_framework_sections(disagreement_types)

            prompt = f"""{self.minimal_adjudication_prompt}

## DISAGREEMENTS DETECTED
{', '.join(disagreement_types) if disagreement_types else "No major disagreements detected"}

## RELEVANT ANNOTATION FRAMEWORK SECTIONS
{json.dumps(retrieved_framework, indent=2, ensure_ascii=False)}

## CASE TO ADJUDICATE

Sentence: {sentence}
Context Before: {limited_context_before if limited_context_before else "N/A"}
Context After: {limited_context_after if limited_context_after else "N/A"}

Annotation 1 (Claude):
{json.dumps(claude_ann, indent=2, ensure_ascii=False) if claude_ann else "N/A"}

Annotation 2 (Gemini):
{json.dumps(gemini_ann, indent=2, ensure_ascii=False) if gemini_ann else "N/A"}

Annotation 3 (Deepseek):
{json.dumps(deepseek_ann, indent=2, ensure_ascii=False) if deepseek_ann else "N/A"}

TASK: Adjudicate systematically following the adjudication process. Output ONLY gold-standard JSON.
"""

            logger.info(f"Using RETRIEVAL mode. Disagreements: {disagreement_types}")
            logger.info(f"Retrieved {len(retrieved_framework)} framework sections")

        else:
            # STANDARD MODE: Use optimization prompt
            json_input = {
                "prompt_instructions": self.optimization_prompt,
                "sentence": sentence,
                "context_before": limited_context_before if limited_context_before else None,
                "context_after": limited_context_after if limited_context_after else None,
                "claude_annotation": claude_ann if claude_ann else None,
                "gemini_annotation": gemini_ann if gemini_ann else None,
                "deepseek_annotation": deepseek_ann if deepseek_ann else None
            }

            json_input_str = json.dumps(json_input, indent=2, ensure_ascii=False)

            prompt = f"""Here is the task in JSON format:

{json_input_str}

Based on the JSON input above, analyze the three model annotations and provide your optimized final decision in JSON format.
"""

            logger.info("Using STANDARD mode (full optimization prompt)")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                # temperature=Config.TEMPERATURE,
                # max_tokens=Config.MAX_TOKENS
            )

            response_text = response.choices[0].message.content.strip()
            extracted_json = self._extract_json_from_markdown(response_text)

            return json.loads(extracted_json)

        except Exception as e:
            logger.error(f"Error in GPT optimization: {e}")
            return self._fallback_optimization(claude_ann, gemini_ann, deepseek_ann, row_data)

    def _fallback_optimization(self, claude_ann: Dict, gemini_ann: Dict,
                               deepseek_ann: Dict, row_data: Dict = None) -> Dict:
        """
        Fallback when GPT optimization fails.
        Logs the failed case to a separate CSV file for review.

        Args:
            claude_ann: Claude's annotation
            gemini_ann: Gemini's annotation
            deepseek_ann: DeepSeek's annotation
            row_data: Original row data (sentence, thesis_code, etc.) for logging

        Returns:
            Fallback annotation dictionary
        """
        # Log this failure to the error CSV
        self._log_failed_annotation(claude_ann, gemini_ann, deepseek_ann, row_data)

        # Return consistent structure with actual annotations
        return {
            "thesis_code": row_data.get("thesis_code", "FALLBACK") if row_data else "FALLBACK",
            "section": row_data.get("section", "Unknown") if row_data else "Unknown",
            "expression_1": "Unknown",
            "expression_2": None,
            "expression_3": None,
            "analysis_1": {
                "dimension_1_observable_realization": {
                    "reflexivity": {"is_reflexive": None}
                },
                "dimension_2_functional_scope": {
                    "classification": None
                },
                "dimension_3_metadiscourse_classification": {
                    "level_1_primary_classification": "BORDERLINE_MD_PROPOSITIONAL",
                    "level_2_functional_category": None,
                    "level_3_specific_type": None
                }
            },
            "confidence_ratings": {
                "overall_confidence": 0
            },
            "borderline_classification": {
                "level_1_borderline_md_propositional": {"is_level_1_borderline": None},
                "level_2_borderline_md_features": {"is_level_2_borderline": None}
            },
            "_fallback_metadata": {
                "is_fallback": True,
                "reason": "GPT optimization failed",
                "timestamp": pd.Timestamp.now().isoformat()
            }
        }

    def _log_failed_annotation(self, claude_ann: Dict, gemini_ann: Dict,
                               deepseek_ann: Dict, row_data: Dict = None) -> None:
        """
        Log failed annotation case to CSV file for later review.

        Args:
            claude_ann: Claude's annotation
            gemini_ann: Gemini's annotation
            deepseek_ann: DeepSeek's annotation
            row_data: Original row data
        """
        error_file_path = os.path.join(project_root, Config.FAILED_ANNOTATION_PATH)

        # Ensure directory exists
        os.makedirs(os.path.dirname(error_file_path), exist_ok=True)

        # Prepare error record
        error_record = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "thesis_code": row_data.get("thesis_code", "Unknown") if row_data else "Unknown",
            "section": row_data.get("section", "Unknown") if row_data else "Unknown",
            "sentence": row_data.get("sentence", "Unknown") if row_data else "Unknown",
            "claude_annotation": json.dumps(claude_ann) if claude_ann else "None",
            "gemini_annotation": json.dumps(gemini_ann) if gemini_ann else "None",
            "deepseek_annotation": json.dumps(deepseek_ann) if deepseek_ann else "None",
            "failure_reason": "GPT optimization failed"
        }

        # Check if file exists to determine if we need headers
        file_exists = os.path.exists(error_file_path)

        # Append to CSV
        error_df = pd.DataFrame([error_record])
        error_df.to_csv(
            error_file_path,
            mode='a',  # Append mode
            header=not file_exists,  # Only write header if file is new
            index=False
        )

        logger.warning(f"⚠️ Failed annotation logged to {error_file_path}")

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
            if idx % 10 == 0 or idx < 10:
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

            # Prepare row_data for error logging
            row_data = {
                "thesis_code": row.get("thesis_code", "Unknown"),
                "section": row.get("section", "Unknown"),
                "sentence": row.get("sentence", "Unknown"),
                "context_before": context_before,
                "context_after": context_after
            }

            # Optimize annotation with context
            optimized = self.optimize_annotation(
                claude_ann, gemini_ann, deepseek_ann,
                row['sentence'],
                context_before, context_after,
                row_data=row_data
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

        # Basic analysis for now
        logger.info("Error analysis completed (basic version)")

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
        return "Improved prompt generation placeholder"

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
            output_path = Config.OUTPUT_CSV_PATH_WITHOUT_BOUNDARY

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

        logger.info("Pipeline completed successfully!")
        return optimized_df, error_analysis


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Annotation Optimizer')
    parser.add_argument('input_file', nargs='?', default=None,
                        help='Input CSV file (optional)')
    parser.add_argument('--rows', type=int, default=None,
                        help='Maximum rows to process')
    parser.add_argument('--context-before', type=int, default=30,
                        help='Words of context before sentence')
    parser.add_argument('--context-after', type=int, default=30,
                        help='Words of context after sentence')
    parser.add_argument('--skip-diagnosis', action='store_true',
                        help='Skip annotation structure diagnosis')
    parser.add_argument('--skip-test', action='store_true',
                        help='Skip normalization test')
    parser.add_argument('--auto-run', action='store_true',
                        help='Run full pipeline without confirmation')

    args = parser.parse_args()

    # Initialize optimizer with CLI arguments
    optimizer = AnnotationOptimizer(
        openai_api_key=Config.OPENAI_API_KEY,
        model=Config.OPENAI_MODEL,
        context_words_before=args.context_before,
        context_words_after=args.context_after,
        use_retrieval=True
    )

    # Optional: Diagnosis
    if not args.skip_diagnosis:
        print("\n🔍 Diagnosing annotation structure...")
        optimizer.diagnose_annotation_structure(max_samples=min(10, args.rows or 10))

    # Optional: Normalization test
    if not args.skip_test:
        print("\n🧪 Testing normalization with real data...")
        stats = optimizer.test_normalization_with_real_data(
            max_cases=min(10, args.rows or 10)
        )

    # Run pipeline
    if args.auto_run:
        run_pipeline = True
    else:
        print("\nWould you like to proceed with the full pipeline? (y/n): ", end='')
        run_pipeline = input().lower() == 'y'

    if run_pipeline:
        optimized_df, error_analysis = optimizer.run_full_pipeline(
            debug_parsing=False,
            max_rows=args.rows
        )
        print(f"\n✅ Optimization completed!")
        print(f"Total cases processed: {len(optimized_df)}")