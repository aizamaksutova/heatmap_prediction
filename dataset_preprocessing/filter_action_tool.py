import pandas as pd
import argparse
import sys
from pathlib import Path

def filter_dataset(input_path, output_path, target_action, target_tool):
    # 1. Load the dataset
    input_file = Path(input_path)
    if not input_file.exists():
        print(f"❌ Error: Input file '{input_path}' not found.")
        return

    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return

    # 2. Validation: Check if columns exist
    if 'action' not in df.columns or 'tool' not in df.columns:
        print("❌ Error: The dataset is missing 'action' or 'tool' columns.")
        print(f"   Available columns: {list(df.columns)}")
        return

    # 3. Apply the filter
    # We strip whitespace to avoid issues with " grasping" vs "grasping"
    mask = (
        (df['action'].astype(str).str.strip() == target_action) & 
        (df['tool'].astype(str).str.strip() == target_tool)
    )
    
    filtered_df = df[mask]
    count = len(filtered_df)

    # 4. Save or Report
    if count == 0:
        print(f"⚠️  No matching rows found for Action='{target_action}' and Tool='{target_tool}'.")
        print("   (Check your spelling or capitalization matches exactly)")
    else:
        # Create output directory if it doesn't exist
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        filtered_df.to_csv(output_path, index=False)
        print(f"✅ Success! Found {count} matching rows.")
        print(f"   Saved new dataset to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter dataset by Action and Tool")
    
    # Required arguments
    parser.add_argument("--input", required=True, help="Path to source CSV")
    parser.add_argument("--output", required=True, help="Path to save the filtered CSV")
    parser.add_argument("--action", required=True, help="The action name to filter by")
    parser.add_argument("--tool", required=True, help="The tool name to filter by")

    args = parser.parse_args()

    filter_dataset(args.input, args.output, args.action, args.tool)