import subprocess
import gspread
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from datetime import datetime
import os
import sys
import shutil
from autoseg.config import read_config
from autoseg.utils import get_artifact_base_path

# Define the available scripts with their aliases
scripts = {
    "train": "train.py",
    "predict": "predict.py",
    "postproc": "post_processing/main.py",
    "evaluate": "eval/evaluate.py",
}

# Check if config path is provided as a command-line argument
if len(sys.argv) < 2:
    print("Usage: python main.py <config_path> [script_aliases...]")
    print("Example: python main.py baselines/unet/1 train predict postproc evaluate")
    sys.exit(1)

config_path = sys.argv[1]
script_aliases = sys.argv[2:]

if not script_aliases:
    print("No valid script aliases provided. Please specify which scripts to run.")
    sys.exit(1)

# Get the model name from the config
model_name = read_config(config_path)["model"]["name"]
row_index = None


def initialize_sheet(sheet):
    global row_index
    # Initialize the sheet with headers if it's empty
    if not sheet.get_all_values():
        headers = [
            "Name",
            "script1 start",
            "script1 end",
            "script2 start",
            "script2 end",
            "script3 start",
            "script3 end",
            "Results Link",
        ]
        sheet.append_row(headers)

    # Find or create row for this test
    test_row = next(
        (row for row in sheet.get_all_values() if row[0] == model_name), None
    )
    if test_row:
        row_index = sheet.get_all_values().index(test_row) + 1
    else:
        row_index = len(sheet.get_all_values()) + 1
        sheet.update_cell(row_index, 1, model_name)


def update_sheet(sheet, script_alias, time, is_start):
    script_number = list(scripts.keys()).index(script_alias) + 1
    col = script_number * 2 if is_start else script_number * 2 + 1
    sheet.update_cell(row_index, col, time.isoformat())


def run_script(script_path, sheet, script_alias):
    print(f"Running {script_path}...")
    start_time = datetime.now()
    update_sheet(sheet, script_alias, start_time, is_start=True)

    try:
        subprocess.run(["python", script_path, config_path], check=True)
    finally:
        end_time = datetime.now()
        update_sheet(sheet, script_alias, end_time, is_start=False)

    print(f"{script_path} completed")


def zip_folder(folder_path, output_path):
    shutil.make_archive(output_path, "zip", folder_path)
    return output_path + ".zip"


def upload_to_drive(file_path, folder_id):
    creds = Credentials.from_service_account_file(
        "service_account.json", scopes=["https://www.googleapis.com/auth/drive.file"]
    )
    drive_service = build("drive", "v3", credentials=creds)

    file_metadata = {"name": os.path.basename(file_path), "parents": [folder_id]}
    media = MediaFileUpload(file_path, resumable=True)
    file = (
        drive_service.files()
        .create(body=file_metadata, media_body=media, fields="id,webViewLink")
        .execute()
    )

    return file.get("webViewLink")


def update_sheet_with_link(sheet, link):
    sheet.update_cell(row_index, 10, link)


# Replace with your actual Google Sheet URL
sheet_url = "https://docs.google.com/spreadsheets/d/1KDPX08F_CjH9GCaCom_WpnSfqSdDkYehZQ8OVxHx9Ow/edit?usp=sharing"

# Connect to the Google Sheet
client = gspread.service_account(filename="service_account.json")
sheet = client.open_by_url(sheet_url).worksheet("Sheet1")

# Initialize the sheet and get the row index
initialize_sheet(sheet)

# Google Drive folder ID
folder_id = "1j2U1h-JOjouaq4jxaLP-CpwyRERkQlYG"

# Echo "Starting" in bash before running the scripts
subprocess.run(["echo", "Starting"], check=True)

# Run specified scripts
evaluate_ran = False
for alias in script_aliases:
    if alias in scripts:
        run_script(scripts[alias], sheet, alias)
        if alias == "evaluate":
            evaluate_ran = True
    else:
        print(f"Warning: Script alias '{alias}' is not valid and will be skipped.")

# Only zip and upload if evaluate.py was run
if evaluate_ran:
    # Get the base path and model name
    base_path = get_artifact_base_path()
    model_name = read_config(config_path)["model"]["name"]

    # Create the full path for the results folder
    results_folder = os.path.join(base_path, model_name, "results")

    # Zip the results folder
    zip_file_path = zip_folder(results_folder, os.path.join(base_path, model_name))

    # Upload the zipped folder to Google Drive and update the sheet
    results_link = upload_to_drive(zip_file_path, folder_id)
    if results_link:
        update_sheet_with_link(sheet, results_link)
        print(
            f"Results folder uploaded from {results_folder} and link added to the sheet."
        )
    else:
        print(f"Failed to upload results folder from {results_folder}.")

    # Remove the zip file after uploading
    os.remove(zip_file_path)
else:
    print("Evaluate script was not run, skipping results upload.")

print("All specified scripts executed successfully.")
