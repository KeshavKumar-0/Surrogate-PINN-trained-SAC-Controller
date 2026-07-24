"""run_plant module.

Handles core functionality and definitions."""
import subprocess
import sys
import time
import os

def spawn_process(script_path, log_file_path):
    """Executes spawn_process operations."""
    print(f'Spawning {script_path}...')
    log_file = open(log_file_path, 'w')
    process = subprocess.Popen([sys.executable, '-u', script_path], stdout=log_file, stderr=subprocess.STDOUT)
    return (process, log_file)

def main():
    """Executes main operations."""
    print('Initializing Asynchronous Urea Control Architecture...')
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.makedirs('logs', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    processes = []
    log_files = []
    try:
        p, l = spawn_process('telemetry/monitor.py', 'logs/monitor.log')
        processes.append(p)
        log_files.append(l)
        time.sleep(0.1)
        p, l = spawn_process('rl_agent/learner.py', 'logs/learner.log')
        processes.append(p)
        log_files.append(l)
        p, l = spawn_process('rl_agent/actor_node.py', 'logs/actor.log')
        processes.append(p)
        log_files.append(l)
        time.sleep(2)
        p, l = spawn_process('surrogate/pinn_export.py', 'logs/plant.log')
        processes.append(p)
        log_files.append(l)
        print('\nSystem Online. Architecture running in background.')
        print('Press Ctrl+C to trigger graceful shutdown and model serialization.\n')
        while True:
            for p in processes:
                if p.poll() is not None:
                    print(f'Process {p.pid} exited unexpectedly! Shutting down.')
                    raise KeyboardInterrupt
            time.sleep(1)
    except KeyboardInterrupt:
        print('\nSIGINT caught. Initiating graceful shutdown. Waiting for processes to save...')
        for p in processes:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print(f'Process {p.pid} hung. Killing.')
                p.kill()
        print('All processes terminated cleanly.')
    finally:
        for l in log_files:
            l.close()
if __name__ == '__main__':
    main()