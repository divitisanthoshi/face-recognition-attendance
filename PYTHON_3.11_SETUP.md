# Setting up Python 3.11 Environment for Smart Attendance System

This guide helps you set up a Python 3.11 environment to avoid compatibility issues with some packages when using Python 3.12.

## Steps

1. Download and install Python 3.11:
   - Visit https://www.python.org/downloads/release/python-3110/
   - Download and install the appropriate installer for your OS.

2. Create a new virtual environment using Python 3.11:
   ```bash
   python3.11 -m venv .venv311
   ```

3. Activate the virtual environment:
   - On Windows CMD:
     ```
     .venv311\Scripts\activate
     ```
   - On PowerShell:
     ```
     .venv311\Scripts\Activate.ps1
     ```

4. Upgrade pip, setuptools, and wheel inside the environment:
   ```bash
   python -m pip install --upgrade pip setuptools wheel
   ```

5. Install the project dependencies:
   ```bash
   pip install -r requirements.txt
   ```

This setup will help you avoid build errors related to Python 3.12 compatibility issues.

If you need further assistance, please let me know.
