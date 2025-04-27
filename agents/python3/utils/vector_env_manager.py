# utils/vector_env_manager.py
import asyncio
from env.safe_gym import SafeGym
import copy

class VectorGymManager:
    def __init__(self, fwd_model_uri, num_envs):
        self.fwd_model_uri = fwd_model_uri
        self.num_envs = num_envs
        self.envs = [SafeGym(fwd_model_uri) for _ in range(num_envs)]
        self.current_states = [None] * num_envs

    async def connect_all(self):
        """连接所有环境"""
        for env in self.envs:
            await env.connect()

    async def reset_all(self):
        """重置所有环境"""
        self.current_states = []
        for env in self.envs:
            state = await env.reset_game()
            await asyncio.sleep(0.2)  # 🧹 防止同步bug，还是保留sleep
            env.make("bomberland-env", state["payload"])
            self.current_states.append(state)

    async def step_all(self, all_actions):
        """
        同时对所有环境执行动作
        Args:
            all_actions: list of actions for each env
        Returns:
            next_states: list of next states
            dones: list of done flags
            infos: list of info dicts
        """
        next_states = []
        dones = []
        infos = []

        for env, actions in zip(self.envs, all_actions):
            try:
                next_state, done, info = await env.step(actions)
                await asyncio.sleep(0.2)
            except Exception as e:
                print(f"[step_all 错误] {e}")
                next_state, done, info = None, True, {}

            next_states.append(next_state)
            dones.append(done)
            infos.append(info)

        self.current_states = next_states
        return next_states, dones, infos

    async def close_all(self):
        """关闭所有环境"""
        for env in self.envs:
            try:
                await env.close()
            except Exception as e:
                print(f"[close_all 错误] {e}")

    def get_current_states(self):
        """返回当前所有环境的状态"""
        return self.current_states

