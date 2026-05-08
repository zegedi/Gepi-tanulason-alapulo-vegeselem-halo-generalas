import sys
import os

def process_file_and_generate_obj(input_filename):
    """
    Processes an input file line by line, applies specific rules,
    and writes the filtered and modified content to a new .obj file.
    """
    # Determine the output filename
    # os.path.splitext separates the filename into (root, ext)
    root, _ = os.path.splitext(input_filename)
    output_filename = root + ".obj"

    print(f"Processing '{input_filename}'...")
    print(f"Output will be written to '{output_filename}'...")

    try:
        with open(input_filename, 'r') as infile, open(output_filename, 'w') as outfile:
            for line_number, line in enumerate(infile, 1): # Start line_number from 1
                stripped_line = line.strip()

                if not stripped_line: # Skip empty lines
                    continue

                parts = stripped_line.split()

                if not parts: # Skip lines that are just whitespace
                    continue

                identifier = parts[0]

                if identifier == 'v':
                    # Check for 3 floating point numbers after 'v'
                    numbers_found = []
                    for part in parts[1:]:
                        try:
                            num = float(part)
                            numbers_found.append(num)
                        except ValueError:
                            # If it's not a float, stop collecting numbers for this line
                            break

                    # Ensure we have exactly 3 numbers
                    while len(numbers_found) < 3:
                        numbers_found.append(0.0) # Add missing zeros

                    # Format the output line
                    # Take only the first 3 numbers, just in case there were more than 3 floats
                    output_line = f"v {" ".join(map(str, numbers_found))}\n"
                    outfile.write(output_line)

                elif identifier == 'f':
                    # Write out 'f' lines as is, preserving their original content
                    # Ensure it ends with a newline
                    outfile.write(line if line.endswith('\n') else line + '\n')

                # For any other identifier, do not output the line (as per requirement 4)

    except FileNotFoundError:
        print(f"Error: Input file '{input_filename}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

    print("Processing complete.")

if __name__ == '__main__':
    
    base_file_name = r"C:\Users\EGV1BP\Documents\FEM+AI\fem_ai_model_v8_hole_in_square\tests_vertex\result_"

    for result_index in range(1, 2001):

        process_file_and_generate_obj(f"{base_file_name}{result_index}.txt")

    # if len(sys.argv) < 2:
    #     print("Usage: python your_script_name.py <input_filename>")
    #     print("Please provide the input filename as an argument.")
    #     sys.exit(1)

    # input_file = sys.argv[1]
    # process_file_and_generate_obj(input_file)