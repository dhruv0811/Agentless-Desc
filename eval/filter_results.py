import json
import os
import glob
from typing import Dict, Any

def filter_entries_with_predicted_files(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filter entries to include only those with non-empty predicted_files lists.
    
    Args:
        data: Dictionary containing the data with instance_metrics
        
    Returns:
        Dictionary with filtered instance_metrics
    """
    # Check if we have the expected structure
    if 'instance_metrics' not in data or not isinstance(data['instance_metrics'], list):
        print("Error: Invalid data structure - instance_metrics array not found")
        return {}
    
    # Filter entries where predicted_files is not empty
    filtered = [
        entry for entry in data['instance_metrics'] 
        if entry.get('predicted_files') and len(entry['predicted_files']) > 0
    ]
    
    # Create a new result object with the filtered entries
    result = {
        'total_instances': data.get('total_instances', 0),
        'total_predictions': data.get('total_predictions', 0),
        'total_gold_files': data.get('total_gold_files', 0),
        'total_correct': data.get('total_correct', 0),
        'instance_metrics': filtered
    }
    
    return result

def process_folder(input_folder: str, output_folder: str = None) -> None:
    """
    Process all JSON files in the input folder and create filtered versions
    in the output folder.
    
    Args:
        input_folder: Path to folder containing JSON files
        output_folder: Path to folder where filtered files will be saved.
                      If None, uses input_folder + '_filtered'
    """
    # Create output folder if it doesn't exist
    if output_folder is None:
        output_folder = input_folder + '_filtered'
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output directory: {output_folder}")
    
    # Process all JSON files in the input folder
    json_files = glob.glob(os.path.join(input_folder, "*.json"))
    if not json_files:
        print(f"No JSON files found in {input_folder}")
        return
    
    file_count = 0
    for file_path in json_files:
        try:
            # Get the filename without the path
            file_name = os.path.basename(file_path)
            output_path = os.path.join(output_folder, f"filtered_{file_name}")
            
            # Load and filter the data
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            filtered_data = filter_entries_with_predicted_files(data)
            
            # Skip writing if filtered data is empty
            if not filtered_data:
                print(f"Warning: No valid data found in {file_name}, skipping")
                continue
                
            # Write filtered data to output file
            with open(output_path, 'w') as f:
                json.dump(filtered_data, f, indent=2)
            
            # Count original and filtered entries
            original_count = len(data.get('instance_metrics', []))
            filtered_count = len(filtered_data.get('instance_metrics', []))
            
            print(f"Processed {file_name}: {filtered_count}/{original_count} entries with non-empty predicted_files")
            file_count += 1
            
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
    
    print(f"Processing complete. {file_count} files processed.")

# Command line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Filter entries with non-empty predicted_files')
    parser.add_argument('--input_folder', default='./orig_results', help='Folder containing JSON files to process')
    parser.add_argument('-o', '--output_folder', default='./filtered_results', help='Output folder for filtered files')
    
    args = parser.parse_args()
    
    process_folder(args.input_folder, args.output_folder)