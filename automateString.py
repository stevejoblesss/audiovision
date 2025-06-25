print("Converting..")
def add_quotes_to_lines(input_file: str, output_file: str = None):
    output_file = output_file or input_file
    
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(" ".join(f'"{line.strip()}",' for line in lines))

input_filename = "LIST OF WANTED PIECES.txt"
output_filename = "outputted pieces.txt"
add_quotes_to_lines(input_filename, output_filename)
print("Conversion Completed")