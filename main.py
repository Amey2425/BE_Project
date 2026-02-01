import subprocess
import platform
import os
import sys
import time

def run_simulation():
    python_exec = sys.executable
    current_os = platform.system()
    project_root = os.path.abspath(os.getcwd())
    
    commands = [
        ("SERVER", f'"{python_exec}" -m server.server_app'),
        ("CLIENT-1", f'"{python_exec}" -m client.client_app --id 1'),
        ("CLIENT-2", f'"{python_exec}" -m client.client_app --id 2 --personalize'),
        ("CLIENT-3", f'"{python_exec}" -m client.client_app --id 3 --personalize'),
    ]

    for name, cmd in commands:
        if current_os == "Windows":
            # Set PYTHONPATH so modules are found correctly on Windows
            full_cmd = f'set PYTHONPATH={project_root} && {cmd}'
            subprocess.Popen(f'start "{name}" cmd /k "{full_cmd}"', shell=True)
        
        elif current_os == "Linux":
            full_cmd = f'export PYTHONPATH=$PYTHONPATH:"{project_root}"; {cmd}'
            # Attempt to use x-terminal-emulator as a more generic option than konsole
            term = "x-terminal-emulator" if subprocess.run(["which", "x-terminal-emulator"], capture_output=True).returncode == 0 else "konsole"
            
            subprocess.Popen([
                term, "-e", "bash", "-c", f"{full_cmd}; exec bash"
            ])
        
        if name == "SERVER":
            time.sleep(2)

    print("All terminals launched. Monitor windows for logs.")

if __name__ == "__main__":
    run_simulation()