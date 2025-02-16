import websocket
import threading
import json

class TerrariaWebSocketClient:
    def __init__(self, url="ws://localhost:5000/game"):
        self.url = url
        self.player_data = {}
        self.stop_event = threading.Event()
        self._ws_thread = None

    # maybe we read every 250 milliseconds?
    def on_message(self, ws, message):
        """Callback function when a new message is received from the WebSocket server."""
        try:
            data = json.loads(message)  # Convert JSON string to Python dictionary
            self.player_data = data
            print(self.player_data)
        except json.JSONDecodeError:
            print("Received non-JSON data:", message)

    def on_error(self, ws, error):
        """Callback function when an error occurs."""
        print("WebSocket Error:", error)

    def on_close(self, ws, close_status_code, close_msg):
        """Callback function when the WebSocket connection is closed."""
        print("WebSocket Closed")

    def on_open(self, ws):
        """Callback function when the WebSocket connection is successfully opened."""
        print("Connected to Terraria WebSocket server!")

    def start(self):
        """Start the WebSocket connection in a separate thread."""
        if self._ws_thread is None:
            self._ws_thread = threading.Thread(target=self._run_ws)
            self._ws_thread.daemon = True  # Ensures the thread stops when the main program exits
            self._ws_thread.start()

    def _run_ws(self):
        """Start the WebSocket app and keep it running."""
        ws = websocket.WebSocketApp(self.url,
                                    on_message=self.on_message,
                                    on_error=self.on_error,
                                    on_close=self.on_close)
        ws.on_open = self.on_open
        ws.run_forever()

    def get_player_data(self):
        """Get the latest player data."""
        return self.player_data

    def stop(self):
        """Stop the WebSocket listener."""
        self._stop_event.set()
        if self._ws_thread:
            self._ws_thread.join()
