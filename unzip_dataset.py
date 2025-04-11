import zipfile
import os

     # Define the path to the zip file and the destination directory
zip_file_path = 'data/ISL.zip'
destination_dir = 'data/ISL'

     # Create the destination directory if it doesn't exist
os.makedirs(destination_dir, exist_ok=True)

     # Unzip the file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
     zip_ref.extractall(destination_dir)

print(f"Extracted {zip_file_path} to {destination_dir}")