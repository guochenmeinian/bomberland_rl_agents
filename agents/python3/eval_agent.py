import asyncio
import json
import os
import torch
from agent.ppo_agent import PPOAgent
from utils.obs_utils import state_to_observations, action_index_to_game_action, get_alive_mask
from utils.save_model import find_latest_checkpoint, load_checkpoint
from config import Config
from env.game_state import GameState
from utils.obs_utils import bombs_positions_and_count


uri = os.environ.get('GAME_CONNECTION_STRING') or "ws://127.0.0.1:3000/?role=agent&agentId=agentId&name=defaultName"

class Agent:
    def __init__(self):
        self.device = Config.device
        self.agent_model = PPOAgent(Config)

        checkpoint_path = "checkpoints/ppo_checkpoint_ep6600.pt"
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            state_dict = checkpoint["model"] 
            self.agent_model.model.load_state_dict(state_dict)
            print(f"✅ Loaded checkpoint from: {checkpoint_path}")


            for name, param in self.agent_model.model.named_parameters():
                if torch.isnan(param).any():
                    print(f"❌ NaN detected in model parameter: {name}")
                    raise ValueError("Model contains NaN, aborting!")
        else:
            print(f"⚠️ Warning: checkpoint not found at {checkpoint_path}")
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
        self.client = GameState(uri)
        self.client.set_game_tick_callback(self._on_game_tick)

        loop = asyncio.get_event_loop()
        connection = loop.run_until_complete(self.client.connect())
        tasks = [
            asyncio.ensure_future(self.client._handle_messages(connection)),
        ]
        loop.run_until_complete(asyncio.wait(tasks))

    async def _on_game_tick(self, tick_number, game_state):
        try:
            agent_id = game_state.get("connection").get("agent_id")
            
            # 提取self state和map
            self_states, full_map, unit_ids, alive_unit_ids = state_to_observations(game_state, agent_id=agent_id)

            current_bomb_infos, current_bomb_count = bombs_positions_and_count(game_state, unit_ids)

            alive_mask = get_alive_mask(unit_ids, alive_unit_ids)
            # 直接PPO推理
            action_indices, _, _, detonate_targets, _ = self.agent_model.select_actions(
                self_states, full_map, alive_mask, current_bomb_infos, current_bomb_count, unit_ids, game_state
            )
            action_indices = action_indices[0]  # batch size = 1, 取第一个
           

            # 生成动作
            actions_to_send = action_index_to_game_action(action_indices, game_state, detonate_targets, agent_id=agent_id)

            for action in actions_to_send:
                try:
                    act_type = action["action"]["type"]
                    unit_id = action["action"]["unit_id"]

                    if act_type == "move":
                        move_dir = action["action"]["move"]
                        await self.client.send_move(move_dir, unit_id)
                    elif act_type == "bomb":
                        await self.client.send_bomb(unit_id)
                    elif act_type == "detonate":
                        coords = action["action"].get("coordinates")
                        if coords and len(coords) == 2:
                            x, y = coords
                            await self.client.send_detonate(x, y, unit_id)
                        else:
                            print(f"[警告] detonate动作没有有效的coordinates字段: {action}")
                    else:
                        continue

                except KeyError as e:
                    print(f"[警告] 动作有问题，缺少字段 {e}，跳过这个动作: {action}")
                except Exception as e:
                    print(f"[警告] 动作处理异常 {e}，跳过这个动作: {action}")


        except Exception as e:
            import traceback
            print("[❌ 错误发生了]")
            print(f"错误类型 (type): {type(e).__name__}")
            print(f"错误信息 (message): {e}")
            print(f"详细Traceback (调用栈):")
            traceback.print_exc()

def main():
    for i in range(0, 10):
        while True:
            try:
                Agent()
            except Exception as e:
                print(f"[❌ Agent启动错误] {e}")
                import time
                time.sleep(5)
                continue
            break

if __name__ == "__main__":
    main()
