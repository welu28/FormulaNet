import os
import pickle
import random
import sys

from holstep_parser import graph_from_hol_stmt

def generate_one_file(file_path, output_path, converter):
    '''
    Process a single HolStep file and save 1 output file.

    Parameters
    ----------
    file_path : str
        Path to the input .txt file
    output_path : str
        Path to save the processed data (e.g., 'holstep000')
    converter : function
        Function to convert statements to graphs/trees
    '''
    outputs = []

    with open(file_path, 'r') as f:
        next(f)  #N line
        conj_symbol = next(f)
        conj_token = next(f)
        assert conj_symbol[0] == 'C'
        conjecture = converter(conj_symbol[2:], conj_token[2:])

        for line in f:
            if line and line[0] in '+-':
                statement = converter(line[2:], next(f)[2:])
                flag = 1 if line[0] == '+' else 0
                outputs.append((flag, conjecture, statement))

    with open(output_path, 'wb') as f:
        pickle.dump(outputs, f, pickle.HIGHEST_PROTOCOL)

    print(f'Done. Saved {len(outputs)} examples to {output_path}')


if __name__ == '__main__':
    # Example usage:
    # python test_generate_one.py data/hol_raw_data/train/file1.txt data/test_out/holstep000

    if len(sys.argv) < 3:
        print("Usage: python test_generate_one.py <input_file> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    generate_one_file(input_file, output_file, graph_from_hol_stmt)