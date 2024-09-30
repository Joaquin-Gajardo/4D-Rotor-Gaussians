import json
from pathlib import Path


def extract_file_paths(json_file, output_file):
    if not json_file.exists():
        print(f"Warning: {json_file} not found.")
        return

    with open(json_file, 'r') as f:
        data = json.load(f)
    
    with open(output_file, 'w') as f:
        for frame in data['frames']:
            path = frame['file_path'].replace('images/', '', 1)
            f.write(f"{path}\n")

    print(f"File paths from {json_file} have been extracted to {output_file}")

def main():
    # Path and file names
    base_path = Path("data/WAT")
    train_json = 'transforms_train.json'
    test_json = 'transforms_test.json'
    train_output = 'train_list.txt'
    test_output = 'test_list.txt'

    for scene in base_path.iterdir():
        if scene.is_dir():

            extract_file_paths(scene / train_json, scene / train_output)

            extract_file_paths(scene / test_json, scene / test_output)

if __name__ == "__main__":
    
    main()