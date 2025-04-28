import os
import pandas as pd

def generate_students_excel(dataset_dir='dataset', output_file='students_list.xlsx'):
    # Get absolute path of dataset directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, dataset_dir)

    # List all folders in dataset directory
    folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]

    # Parse folder names to extract student name and roll number
    data = []
    for folder in folders:
        # Assuming folder name format: "StudentName RollNumber"
        parts = folder.rsplit(' ', 1)
        if len(parts) == 2:
            name, roll_number = parts
            data.append({'Name': name, 'Roll Number': roll_number})
        else:
            # If format unexpected, put whole folder name as name and roll number empty
            data.append({'Name': folder, 'Roll Number': ''})

    # Create DataFrame and write to Excel
    df = pd.DataFrame(data)
    output_path = os.path.join(base_dir, output_file)
    df.to_excel(output_path, index=False)
    print(f"Excel file generated at: {output_path}")

def generate_students_excel(dataset_dir='dataset', output_file='students_list.xlsx'):
    import os
    import pandas as pd

    # Get absolute path of dataset directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, dataset_dir)

    # List all folders in dataset directory
    folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]

    # Parse folder names to extract student name and roll number
    data = []
    for folder in folders:
        # Assuming folder name format: "StudentName RollNumber"
        parts = folder.rsplit(' ', 1)
        if len(parts) == 2:
            name, roll_number = parts
            data.append({'Name': name, 'Roll Number': roll_number})
        else:
            # If format unexpected, put whole folder name as name and roll number empty
            data.append({'Name': folder, 'Roll Number': ''})

    # Create DataFrame and write to Excel
    df = pd.DataFrame(data)
    output_path = os.path.join(base_dir, output_file)
    df.to_excel(output_path, index=False)
    print(f"Excel file generated at: {output_path}")

if __name__ == '__main__':
    generate_students_excel()
