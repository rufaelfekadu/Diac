import argparse
import os
import pandas as pd
import csv

os.environ["PYTHONIOENCODING"] = "utf-8"


def load_file(file_path):
    """
    Load file - either TSV with reference and ASR or single-column text file
    """
    try:
        # First try to read as TSV with two columns
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            rows = list(reader)

        if len(rows) > 0 and len(rows[0]) == 2:
            df = pd.DataFrame(rows, columns=["reference", "asr"])
        elif len(rows) > 0 and len(rows[0]) == 1:
            df = pd.DataFrame(rows, columns=["reference"])
        else:
            raise ValueError(
                "File format not recognized. Expected TSV with 2 columns or single-column text file."
            )
    except Exception as e:
        print(f"Error loading file: {e}")
        return None
    return df


def save_references(df, output_dir):
    """
    Save reference and ASR transcriptions to separate files
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save references
    ref_path = os.path.join(output_dir, "ref.txt")
    with open(ref_path, "w", encoding="utf-8") as f:
        f.write("\n".join(df["reference"].astype(str)))

    # Save ASR transcriptions if they exist
    if "asr" in df.columns:
        asr_path = os.path.join(output_dir, "asr.txt")
        with open(asr_path, "w", encoding="utf-8") as f:
            f.write("\n".join(df["asr"].astype(str)))

    return ref_path, asr_path


def main():
    parser = argparse.ArgumentParser(
        description="Prepare reference data from TSV or text file"
    )
    parser.add_argument(
        "--input_file",
        required=True,
        help="Path to TSV file with reference and ASR transcriptions, or single-column text file",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="output",
        help="Output directory for processed files",
    )

    args = parser.parse_args()

    # Load the file
    df = load_file(args.input_file)
    if df is None:
        return 1

    # Print some statistics
    print(f"Loaded {len(df)} entries from {args.input_file}")
    print(f"Example entry:\nReference: {df['reference'].iloc[0]}")
    if not df["asr"].iloc[0] == "":
        print(f"ASR: {df['asr'].iloc[0]}")
    else:
        print("No ASR transcriptions found (single-column file)")

    # Save references and ASR transcripts
    ref_path, asr_path = save_references(df, args.output_dir)
    print(f"References saved to: {ref_path}")
    print(f"ASR transcriptions saved to: {asr_path}")

    return 0


if __name__ == "__main__":
    exit(main())
