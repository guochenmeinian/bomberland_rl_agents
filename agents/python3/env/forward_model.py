from typing import Dict, List
import websockets
import json


class ForwardModel:
    def __init__(self, connection_string: str):
        self._connection_string = connection_string
        self._next_state_callback = None
        self.connection = None

    async def close(self):
        if self.connection is not None:
            await self.connection.close()

    def set_next_state_callback(self, next_state_callback):
        self._next_state_callback = next_state_callback

    async def connect(self):
        self.connection = await websockets.connect(self._connection_string)
        if self.connection.open:
            return self.connection

    async def _handle_messages(self, connection: str):
        while True:
            try:
                raw_data = await connection.recv()
                data = json.loads(raw_data)
                await self._on_data(data)
            except websockets.exceptions.ConnectionClosed:
                print('Connection with server closed')
                break

    async def _on_data(self, data):
        data_type = data.get("type")

        if data_type == "info":
            # no operation
            pass
        elif data_type == "next_game_state":
            payload = data.get("payload")
            await self._on_next_state(payload)
        elif data_type == "game_state":
            # no-op
            return
        else:
            print(f"unknown packet \"{data_type}\": {data}")

    async def _on_next_state(self, payload):
        if self._next_state_callback != None:
            await self._next_state_callback(payload)

    """
    sample moves payload:
    [
        {
            "action": {"move": "right", "type": "move"},
            "agent_number": 0,
        }, {
            "action": {"move": "left", "type": "move"},
            "agent_number": 1,
        }
    ]

    REMARKS:
    `sequence_id` is used to for you match up an evaluated
    next_state call since payloads can come back in any order
    It should ideally be unique
    """
    async def send_next_state(self, sequence_id: int, game_state: Dict, actions: List[Dict]):
        game_state.pop("connection", None)
        
        # 👇 打印所有 entity type
        # print("[DEBUG] entity types:", {e.get("type") for e in game_state["entities"]})
        # print("[DEBUG] actions:", actions)

        packet = {"actions": actions,
                  "type": "evaluate_next_state", "state": game_state, "sequence_id": sequence_id}
        await self.connection.send(json.dumps(packet))
