import subprocess
import platform
import os
import sys
import time

def run_simulation():
    python_exec = sys.executable
    current_os = platform.system()
    project_root = os.path.abspath(os.getcwd())
    
    # We clean the commands to be single-line strings to avoid newline errors
    commands = [
        ("SERVER", f'"{python_exec}" -m server.server_app'),
        ("CLIENT-1", f'"{python_exec}" -m client.client_app --id 1'),
        ("CLIENT-2", f'"{python_exec}" -m client.client_app --id 2 --personalize'),
        ("CLIENT-3", f'"{python_exec}" -m client.client_app --id 3 --personalize'),
    ]

    print(f"🚀 Detected OS: {current_os}")
    print(f"📂 Project Root: {project_root}")

    for name, cmd in commands:
        if current_os == "Windows":
            # Windows 'start' uses the first quoted string as the Title
            # We use /k to keep the window open after execution
            subprocess.Popen(f'start "{name}" cmd /k "{cmd}"', shell=True)
        
        elif current_os == "Linux":
            # Using double quotes for PYTHONPATH to handle spaces in folder names
            full_cmd = f'export PYTHONPATH=$PYTHONPATH:"{project_root}"; {cmd}'
            
            print(f"📡 Launching {name} in Konsole...")
            subprocess.Popen([
                "konsole", 
                "--title", name,        # Sets the window/tab name
                "--noclose",            # Keeps window open on exit
                "--workdir", project_root, 
                "-e", "bash", "-c", full_cmd
            ])
        
        # Give the server a head start
        if name == "SERVER":
            time.sleep(5)

    print("✅ All terminals launched. Monitor windows for logs.")

if __name__ == "__main__":
    run_simulation()