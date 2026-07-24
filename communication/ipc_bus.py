"""ipc_bus module.

Handles core functionality and definitions."""
import zmq
PORTS = {'telemetry': 5555, 'transitions': 5556, 'weights': 5557, 'control': 5558}

def get_context():
    """Executes get_context operations."""
    return zmq.Context.instance()

def create_push_socket():
    """Executes create_push_socket operations."""
    context = get_context()
    socket = context.socket(zmq.PUSH)
    socket.setsockopt(zmq.LINGER, 0)
    socket.connect(f"tcp://localhost:{PORTS['transitions']}")
    return socket

def create_pull_socket():
    """Executes create_pull_socket operations."""
    context = get_context()
    socket = context.socket(zmq.PULL)
    socket.bind(f"tcp://*:{PORTS['transitions']}")
    return socket

def create_weight_pub_socket():
    """Executes create_weight_pub_socket operations."""
    context = get_context()
    socket = context.socket(zmq.PUB)
    socket.bind(f"tcp://*:{PORTS['weights']}")
    return socket

def create_weight_sub_socket():
    """Executes create_weight_sub_socket operations."""
    context = get_context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.connect(f"tcp://localhost:{PORTS['weights']}")
    socket.setsockopt_string(zmq.SUBSCRIBE, '')
    return socket

def create_telemetry_pub_socket():
    """Executes create_telemetry_pub_socket operations."""
    context = get_context()
    socket = context.socket(zmq.PUB)
    socket.connect(f"tcp://localhost:{PORTS['telemetry']}")
    return socket

def create_telemetry_sub_socket():
    """Executes create_telemetry_sub_socket operations."""
    context = get_context()
    socket = context.socket(zmq.SUB)
    socket.bind(f"tcp://*:{PORTS['telemetry']}")
    socket.setsockopt_string(zmq.SUBSCRIBE, '')
    return socket

def poll_socket_with_timeout(socket, timeout_ms=500):
    """Executes poll_socket_with_timeout operations."""
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)
    socks = dict(poller.poll(timeout_ms))
    return socket in socks and socks[socket] == zmq.POLLIN