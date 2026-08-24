# **Running The App**

## 

## From Anaconda Prompt



1. Open Anaconda Prompt from the Start menu.

2. Activate the environment:

conda activate vfm\_agri\_env

3. Navigate to the project folder (quotes matter because of the space in the path):

cd /d "D:\\ShaunakKathavate Github\\vegetable\_and\_fruits\_time\_series\_and\_clustering" # Insert your path between the quotes

4. Run the app:

python app\\run\_app.py

5. Wait for it to print something like:

You can now view your Streamlit app in your browser.

URL: http://localhost:8501

6. Open that URL in your browser.

7. To stop the app later, go back to that window and press Ctrl+C.





## From the VS Code terminal



*VS Code's terminal is usually PowerShell here, and it may default to a different Python than vfm\_agri\_env — so activate explicitly rather than assuming.*



1. Open a terminal in VS Code (Ctrl+`).

2. Make sure it's a PowerShell prompt (bottom-right of the terminal panel shows the shell type). If it's opened inside app/ or elsewhere, go to the project root:

cd "D:\\ShaunakKathavate Github\\vegetable\_and\_fruits\_time\_series\_and\_clustering" # Insert your path between the quotes


3. Activate the conda environment:

conda activate vfm\_agri\_env

-> If PowerShell says something like "conda is not recognized" or blocks activation with a script-execution error, conda isn't initialized for PowerShell yet. In that case skip activation and just call the interpreter directly instead (step 4b below).

4. Run the app — two ways, pick whichever worked above:

-> a) If conda activate succeeded:

      python app\\run\_app.py

-> b) If conda isn't available in this terminal, call the environment's Python directly, no activation needed:

      \& "C:\\Users\\ASUS\\anaconda3\\envs\\vfm\_agri\_env\\python.exe" app\\run\_app.py

5. Open http://localhost:8501 in your browser once you see the "You can now view your Streamlit app" message.

6. To stop it, click into that terminal and press Ctrl+C.



Either way, don't use VS Code's ▶ "Run Python File" button for this one — it may launch with whatever interpreter is currently selected for the editor (which could be a different, unrelated Python install), not vfm\_agri\_env.



If you hit an actual error message with either method, paste it here and I'll dig in rather than guessing further.

