# ML Environment

## Setup

First time setup:
1. Open terminal in this folder.
2. Run `python -m venv venv`
3. Activate the environment:
   - Windows (PowerShell): `.\venv\Scripts\Activate.ps1`
   - Windows (Command Prompt): `.\venv\Scripts\activate.bat`
   - Git Bash: `source venv/Scripts/activate`
4. Install dependencies: `pip install -r requirements.txt`

## Running

1. Activate the environment (if not already active):
   - Windows (PowerShell): `.\venv\Scripts\Activate.ps1`
   - Windows (Command Prompt): `.\venv\Scripts\activate.bat`
   - Git Bash: `source venv/Scripts/activate`
2. Run the Streamlit app:
   ```bash
   streamlit run main.py
   ```
