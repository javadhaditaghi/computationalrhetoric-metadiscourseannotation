import spacy
import pandas as pd
import os
import argparse
import re
from pathlib import Path


def load_scispacy_model():
    """
    Load a scispacy model.
    Make sure you have installed scispacy and downloaded a model:
    pip install scispacy
    pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_sm-0.5.3.tar.gz
    """
    try:
        nlp = spacy.load("en_core_sci_sm")
        return nlp
    except OSError:
        print("Error: scispacy model 'en_core_sci_sm' not found.")
        print("Please install it with:")
        print(
            "pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_sm-0.5.3.tar.gz")
        return None


def clean_number_structures(text):
    """
    Remove structures like 'enter+number+enter' from text.
    Returns cleaned text and count of removed structures.

    Examples of patterns to remove:
    - \n1\n
    - \n23\n
    - \n456\n
    """
    # Pattern to match: newline + one or more digits + newline
    pattern = r'\n\d+\n'

    # Find all matches to count them
    matches = re.findall(pattern, text)
    count_removed = len(matches)

    # Remove the structures
    cleaned_text = re.sub(pattern, ' ', text)

    # Clean up multiple spaces that might result from removal
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    return cleaned_text, count_removed


def get_context_words(tokens, start_idx, end_idx, context_size=150, direction='before'):
    """
    Extract context words before or after a sentence.

    Args:
        tokens: List of spacy tokens
        start_idx: Start index of the sentence
        end_idx: End index of the sentence
        context_size: Number of words to extract
        direction: 'before' or 'after'

    Returns:
        String of context words
    """
    if direction == 'before':
        # Get tokens before the sentence
        context_start = max(0, start_idx - context_size)
        context_tokens = tokens[context_start:start_idx]
    else:  # direction == 'after'
        # Get tokens after the sentence
        context_end = min(len(tokens), end_idx + context_size)
        context_tokens = tokens[end_idx:context_end]

    # Join tokens with spaces, preserving original spacing where possible
    context_text = ""
    for i, token in enumerate(context_tokens):
        if i == 0:
            context_text = token.text
        else:
            # Add space if there was whitespace after the previous token
            if context_tokens[i - 1].whitespace_:
                context_text += " " + token.text
            else:
                context_text += token.text

    return context_text.strip()


def process_text_file(file_path, output_path, context_window=150):
    """
    Process a text file and create a dataset with sentences and context.

    Args:
        file_path: Path to input text file
        output_path: Path to output CSV file
        context_window: Number of words for context before and after
    """
    # Load scispacy model
    nlp = load_scispacy_model()
    if nlp is None:
        return False

    # Read the text file
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return False

    # Clean number structures from the entire text first
    print("Cleaning number structures (\\nnumber\\n)...")
    cleaned_text, total_structures_removed = clean_number_structures(text)
    print(f"Removed {total_structures_removed} number structures from the text")

    # Process the cleaned text with scispacy
    print("Processing text with scispacy...")
    doc = nlp(cleaned_text)

    # Extract sentences and create dataset
    dataset = []
    tokens = list(doc)  # Convert to list for indexing
    sentence_structures_cleaned = 0  # Counter for structures cleaned from individual sentences

    print(f"Found {len(list(doc.sents))} sentences")

    for sent in doc.sents:
        # Clean any remaining number structures in the sentence
        original_sentence = sent.text.strip()
        cleaned_sentence, structures_in_sentence = clean_number_structures(original_sentence)
        sentence_structures_cleaned += structures_in_sentence
        # Find token indices for this sentence
        sent_start_idx = None
        sent_end_idx = None

        for i, token in enumerate(tokens):
            if token.idx == sent.start_char:
                sent_start_idx = i
            if token.idx + len(token.text) == sent.end_char:
                sent_end_idx = i + 1
                break

        if sent_start_idx is None or sent_end_idx is None:
            # Fallback: find approximate indices
            for i, token in enumerate(tokens):
                if token in sent:
                    if sent_start_idx is None:
                        sent_start_idx = i
                    sent_end_idx = i + 1

        # Get context before and after
        context_before = get_context_words(tokens, sent_start_idx, sent_end_idx,
                                           context_window, 'before')
        context_after = get_context_words(tokens, sent_start_idx, sent_end_idx,
                                          context_window, 'after')

        # Extract thesis code (you can modify this logic based on your needs)
        thesis_code = extract_thesis_code(file_path)

        # Create row
        row = {
            'sentence': cleaned_sentence,
            'context_before': context_before,
            'context_after': context_after,
            'section': '',  # Empty as requested
            'thesis_code': thesis_code
        }

        dataset.append(row)

    # Print cleaning summary
    total_cleaned = total_structures_removed + sentence_structures_cleaned
    print(f"\n=== CLEANING SUMMARY ===")
    print(f"Number structures removed from full text: {total_structures_removed}")
    print(f"Number structures removed from individual sentences: {sentence_structures_cleaned}")
    print(f"Total number structures cleaned: {total_cleaned}")
    print("========================\n")

    # Create DataFrame and save to CSV
    df = pd.DataFrame(dataset)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Dataset saved to {output_path}")
    print(f"Created {len(df)} rows")

    return True


def extract_thesis_code(file_path):
    """
    Extract thesis code from filename or path.
    Modify this function based on your naming convention.
    """
    filename = os.path.basename(file_path)
    # Remove .txt extension
    thesis_code = filename.replace('.txt', '')
    return thesis_code


def main():
    parser = argparse.ArgumentParser(description='Process text files with scispacy to create dataset')
    parser.add_argument('--input_file', '-i', required=True,
                        help='Input text file path (e.g., dataset_creation/data/thesis1.txt)')
    parser.add_argument('--output_file', '-o',
                        help='Output CSV file path (default: auto-generated in dataset_creation/result/preprocessed/)')
    parser.add_argument('--context_window', '-c', type=int, default=150,
                        help='Context window size in words (default: 150)')

    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.input_file):
        print(f"Error: Input file {args.input_file} does not exist")
        return

    # Generate output filename if not provided
    if args.output_file is None:
        input_filename = os.path.basename(args.input_file).replace('.txt', '')
        args.output_file = f"dataset_creation/result/preprocessed/{input_filename}_processed.csv"

    print(f"Input file: {args.input_file}")
    print(f"Output file: {args.output_file}")
    print(f"Context window: {args.context_window} words")

    # Process the file
    success = process_text_file(args.input_file, args.output_file, args.context_window)

    if success:
        print("Processing completed successfully!")
    else:
        print("Processing failed!")


if __name__ == "__main__":
    main()