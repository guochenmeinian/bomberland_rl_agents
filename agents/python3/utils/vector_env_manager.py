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
        # for env in self.envs:
        #     await env.connect()
        await asyncio.gather(*[env.connect() for env in self.envs])

    async def reset_all(self):
        """重置所有环境"""
        results = await asyncio.gather(*[env.reset_game() for env in self.envs])
        
        # 设置 _prev_state
        self.current_states = []
        for state in results:
            if isinstance(state, dict):
                state["_prev_state"] = copy.deepcopy(state)
            self.current_states.append(state)

        # 创建环境
        await asyncio.gather(*[
            env.make("bomberland-env", state["payload"])
            for env, state in zip(self.envs, self.current_states)
        ])


    async def step_all(self, all_actions):
        results = await asyncio.gather(*[
            env.step(actions) if actions else (None, True, {}) 
            for env, actions in zip(self.envs, all_actions)
        ])
        next_states, dones, infos = zip(*results)

        updated_states = []
        for i, (new_state, done) in enumerate(zip(next_states, dones)):
            prev_state = self.current_states[i]
            if isinstance(new_state, dict):
                if isinstance(prev_state, dict):
                    if "_prev_state" in prev_state:
                        new_state["_prev_state"] = prev_state["_prev_state"]
                    else:
                        new_state["_prev_state"] = prev_state.copy()
                    if "_meta" in prev_state:
                        new_state["_meta"] = prev_state["_meta"]
            updated_states.append(new_state)

        self.current_states = updated_states
        return list(next_states), list(dones), list(infos)

    async def close_all(self):
        """关闭所有环境"""
        # for env in self.envs:
        #     try:
        #         await env.close()
        #     except Exception as e:
        #         print(f"[close_all 错误] {e}")
        await asyncio.gather(*[env.close() for env in self.envs])

    def get_current_states(self):
        """返回当前所有环境的状态"""
        return self.current_states

