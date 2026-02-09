import pandas as pd
import argparse
import sys

def analyze_columns(csv_path):
    try:
        # Load the dataset
        df = pd.read_csv(csv_path)
        print(f"✅ Successfully loaded: {csv_path}")
        print(f"Total rows: {len(df)}\n")
    except FileNotFoundError:
        print(f"❌ Error: The file '{csv_path}' was not found.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        sys.exit(1)

    # The columns we want to analyze
    target_columns = ["action", "tool"]

    for col in target_columns:
        print(f"--- Analysis for column: '{col}' ---")
        
        if col not in df.columns:
            print(f"⚠️  Warning: Column '{col}' not found in CSV. Skipping.")
            print("-" * 30 + "\n")
            continue

        # Get unique values and their counts
        # dropna=False ensures we also count missing/empty values if they exist
        value_counts = df[col].value_counts(dropna=False)
        unique_values = df[col].unique()

        print(f"Total Unique Values: {len(unique_values)}")
        print("\nTop 10 most frequent values:")
        print(value_counts.head(10))
        
        # If you want to see ALL values, uncomment the line below:
        # print(value_counts)

        print("-" * 30 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze unique values in 'action' and 'tool' columns.")
    # Add an argument so you can run it like: python analyze.py my_data.csv
    parser.add_argument("csv_file", nargs="?", default="dataset.csv", help="Path to your CSV file")
    
    args = parser.parse_args()
    analyze_columns(args.csv_file)