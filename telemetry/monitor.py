"""monitor module.

Handles core functionality and definitions."""
import sys
import os
import time
import json
import zmq
import csv
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from communication.ipc_bus import create_telemetry_sub_socket, PORTS
CSV_FILE = 'logs/telemetry_history.csv'
TXT_FILE = 'live_dashboard.txt'
CSV_HEADERS = ['timestamp', 'source', 'mass', 'temperature', 'c_urea', 'c_water', 'c_biuret', 'input_heat', 'pressure', 'reward']

def clear_console():
    """Executes clear_console operations."""
    sys.stdout.write('\x1b[H\x1b[2J')
    sys.stdout.flush()

def format_dashboard(metrics):
    """Executes format_dashboard operations."""
    lines = ['=================================================================', '                  UREA PLANT TELEMETRY MONITOR                   ', '=================================================================', f" Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}", f" Source   : {metrics.get('source', 'UNKNOWN').upper()}", '-----------------------------------------------------------------', ' PHYSICAL REACTOR STATES:', f"   Total Mass (M)          : {metrics.get('mass', 0.0):10.4f} kg", f"   Temperature (T)         : {metrics.get('temperature', 0.0):10.2f} °C", f"   Urea Mass Frac (C_u)    : {metrics.get('c_urea', 0.0):10.6f}", f"   Water Mass Frac (C_w)   : {metrics.get('c_water', 0.0):10.6f}", f"   Biuret Mass Frac (C_b)  : {metrics.get('c_biuret', 0.0):10.6f}", '-----------------------------------------------------------------', ' CONTROL ACTUATORS:', f"   Steam Valve Input (Q)   : {metrics.get('input_heat', 0.0):10.2f} kW", f"   Vacuum Pressure (P)     : {metrics.get('pressure', 0.0):10.4f} bar", '-----------------------------------------------------------------', ' SAFETY CONTROLLER METRICS:']
    temp = metrics.get('temperature', 0.0)
    biuret = metrics.get('c_biuret', 0.0)
    if temp >= 135.0:
        lines.append('   [CRITICAL ALERT] Temperature exceeds 135°C threshold!')
    else:
        lines.append('   Thermal Status          : OPERATIONAL (Nominal)')
    if biuret >= 0.008:
        lines.append('   [CRITICAL ALERT] Biuret contamination exceeds 0.008 fraction!')
    else:
        lines.append('   Purity Status           : NOMINAL')
    lines.append('================================================================')
    return '\n'.join(lines)

def main():
    """Executes main operations."""
    try:
        socket = create_telemetry_sub_socket()
    except Exception as e:
        print(f'Error creating telemetry socket: {e}')
        sys.exit(1)
    socket.setsockopt_string(zmq.SUBSCRIBE, '')
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)
    os.makedirs('logs', exist_ok=True)
    csv_exists = os.path.isfile(CSV_FILE)
    csv_file = open(CSV_FILE, mode='a', newline='')
    csv_writer = csv.writer(csv_file)
    if not csv_exists:
        csv_writer.writerow(CSV_HEADERS)
        csv_file.flush()
    print(f"Telemetry daemon listening on port {PORTS['telemetry']}...")
    last_ui_update = 0
    update_interval = 0.1
    try:
        while True:
            socks = dict(poller.poll(timeout=100))
            if socket in socks and socks[socket] == zmq.POLLIN:
                message = socket.recv_string()
                try:
                    metrics = json.loads(message)
                    current_time = time.time()
                    row = [current_time, metrics.get('source', 'UNKNOWN'), metrics.get('mass', 0.0), metrics.get('temperature', 0.0), metrics.get('c_urea', 0.0), metrics.get('c_water', 0.0), metrics.get('c_biuret', 0.0), metrics.get('input_heat', 0.0), metrics.get('pressure', 0.0), metrics.get('reward', 0.0)]
                    csv_writer.writerow(row)
                    csv_file.flush()
                    if current_time - last_ui_update >= update_interval:
                        dashboard_str = format_dashboard(metrics)
                        with open(TXT_FILE, 'w') as f:
                            f.write(dashboard_str)
                        clear_console()
                        sys.stdout.write(dashboard_str + '\n')
                        sys.stdout.flush()
                        last_ui_update = current_time
                except json.JSONDecodeError:
                    sys.stderr.write('Malformed JSON frame dropped.\n')
                    sys.stderr.flush()
    except KeyboardInterrupt:
        print('\nShutting down Telemetry Monitor cleanly.')
    finally:
        csv_file.close()
        socket.close()
        context.term()
if __name__ == '__main__':
    main()