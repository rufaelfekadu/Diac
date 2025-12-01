import re
import sys
from collections import defaultdict


def preprocess_text(text):
    """Clean and tokenize Arabic text."""
    # Remove non-Arabic characters except whitespace
    text = re.sub(r"[^\u0600-\u06FF\s]", "", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def calculate_wer(reference, hypothesis):
    """
    Calculate Word Error Rate (WER) between reference and hypothesis.
    WER = (S + D + I) / N, where:
    S = substitutions, D = deletions, I = insertions, N = number of words in reference
    """
    # Preprocess and tokenize
    ref_words = preprocess_text(reference).split()
    hyp_words = preprocess_text(hypothesis).split()

    # Initialize the distance matrix
    d = [[0 for _ in range(len(hyp_words) + 1)] for _ in range(len(ref_words) + 1)]

    # Fill first row and column
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j

    # Compute Levenshtein distance
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitution = d[i - 1][j - 1] + 1
                insertion = d[i][j - 1] + 1
                deletion = d[i - 1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    # The last element contains the distance
    distance = d[len(ref_words)][len(hyp_words)]

    # Calculate WER
    if len(ref_words) == 0:
        return 0 if len(hyp_words) == 0 else float("inf")
    return distance / len(ref_words)


def calculate_cer(reference, hypothesis):
    """
    Calculate Character Error Rate (CER) between reference and hypothesis.
    CER = (S + D + I) / N, where:
    S = substitutions, D = deletions, I = insertions, N = number of characters in reference
    """
    # Preprocess
    ref_chars = list(preprocess_text(reference).replace(" ", ""))
    hyp_chars = list(preprocess_text(hypothesis).replace(" ", ""))

    # Initialize the distance matrix
    d = [[0 for _ in range(len(hyp_chars) + 1)] for _ in range(len(ref_chars) + 1)]

    # Fill first row and column
    for i in range(len(ref_chars) + 1):
        d[i][0] = i
    for j in range(len(hyp_chars) + 1):
        d[0][j] = j

    # Compute Levenshtein distance
    for i in range(1, len(ref_chars) + 1):
        for j in range(1, len(hyp_chars) + 1):
            if ref_chars[i - 1] == hyp_chars[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitution = d[i - 1][j - 1] + 1
                insertion = d[i][j - 1] + 1
                deletion = d[i - 1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    # The last element contains the distance
    distance = d[len(ref_chars)][len(hyp_chars)]

    # Calculate CER
    if len(ref_chars) == 0:
        return 0 if len(hyp_chars) == 0 else float("inf")
    return distance / len(ref_chars)


def process_files(ref_file, hyp_file):
    """Process the reference and hypothesis files and calculate WER."""
    try:
        with open(ref_file, "r", encoding="utf-8") as f_ref:
            ref_lines = f_ref.readlines()

        with open(hyp_file, "r", encoding="utf-8") as f_hyp:
            hyp_lines = f_hyp.readlines()

        total_wer = 0
        total_cer = 0
        processed_lines = 0

        # Process each line
        for i, (ref, hyp) in enumerate(zip(ref_lines, hyp_lines)):
            wer = calculate_wer(ref, hyp)
            cer = calculate_cer(ref, hyp)
            print(f"Line {i+1} WER: {wer:.4f}, CER: {cer:.4f}")
            total_wer += wer
            total_cer += cer
            processed_lines += 1

        # Calculate average WER across all lines
        if processed_lines > 0:
            avg_wer = total_wer / processed_lines
            avg_cer = total_cer / processed_lines
            print(f"\nAverage WER: {avg_wer:.4f}, Average CER: {avg_cer:.4f}")
        else:
            print("No lines processed.")

    except Exception as e:
        print(f"Error processing files: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python wer.py reference_file hypothesis_file")
        sys.exit(1)

    reference_file = sys.argv[1]
    hypothesis_file = sys.argv[2]

    process_files(reference_file, hypothesis_file)
