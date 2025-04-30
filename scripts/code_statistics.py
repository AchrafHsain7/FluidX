import os
import argparse

def count_lines_in_py_files(root_dir):
    """
    Recursively count all Python code lines in .py files starting from the root directory.
    
    Args:
        root_dir (str): The root directory to start scanning from
        
    Returns:
        dict: Dictionary with statistics about Python files and lines
    """
    total_lines = 0
    total_files = 0
    file_counts = {}
    
    # Walk through all subdirectories
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Filter out only Python files
        py_files = [f for f in filenames if f.endswith('.py')]
        print(py_files)
        for py_file in py_files:
            file_path = os.path.join(dirpath, py_file)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    # Count lines in the file
                    lines = len(f.readlines())
                    total_lines += lines
                    total_files += 1
                    
                    # Store individual file statistics
                    rel_path = os.path.relpath(file_path, root_dir)
                    file_counts[rel_path] = lines
                    
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    
    return {
        'total_files': total_files,
        'total_lines': total_lines,
        'file_counts': file_counts
    }

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Count lines in Python files recursively.')
    parser.add_argument('root_dir', nargs='?', default='.', 
                        help='Root directory to start scanning (default: current directory)')
    parser.add_argument('-d', '--detailed', action='store_true',
                        help='Show detailed breakdown by file')
    args = parser.parse_args()
    
    # Count lines
    stats = count_lines_in_py_files(args.root_dir)
    
    # Print results
    print(f"\nPython Code Statistics for: {os.path.abspath(args.root_dir)}")
    print(f"Total Python files found: {stats['total_files']}")
    print(f"Total lines of Python code: {stats['total_lines']}")
    
    # Print detailed breakdown if requested
    if args.detailed and stats['file_counts']:
        print("\nBreakdown by file:")
        print("-" * 80)
        for file_path, line_count in sorted(stats['file_counts'].items(), 
                                           key=lambda x: x[1], reverse=True):
            print(f"{line_count:8d} lines: {file_path}")

if __name__ == "__main__":
    main()