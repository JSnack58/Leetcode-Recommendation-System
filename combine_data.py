import os
import glob

def combine_jsonl_files(input_dir, output_file):
    """
    Combines all .jsonl files in the given directory into a single .jsonl file.
    """
    # Find all .jsonl files in the input directory
    search_pattern = os.path.join(input_dir, "*.jsonl")
    file_list = glob.glob(search_pattern)
    
    if not file_list:
        print(f"No .jsonl files found in '{input_dir}'.")
        return

    print(f"Found {len(file_list)} files to combine.")
    
    # Open the output file in write mode
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for file_path in file_list:
            # print(f"Processing: {os.path.basename(file_path)}")
            with open(file_path, 'r', encoding='utf-8') as infile:
                for line in infile:
                    outfile.write(line)
                    
    print(f"Successfully combined {len(file_list)} files into '{output_file}'.")

if __name__ == "__main__":
    input_directory = "contest data"
    output_filename = "combined_contest_data.jsonl"
    
    combine_jsonl_files(input_directory, output_filename)
