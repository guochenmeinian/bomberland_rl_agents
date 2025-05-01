# utils/vector_env_manager.py
import asyncio
from env.safe_gym import SafeGym
import copy
import numpy as np

def to_serializable(obj):
    """递归地将 numpy 类型转换为 JSON 可序列化的类型"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    else:
        return obj


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
        results = await asyncio.gather(*[env.reset_game() for env in self.envs], return_exceptions=True)

        self.current_states = []
        for i, state in enumerate(results):
            if isinstance(state, Exception):
                print(f"[reset_all] 环境 {i} reset_game 异常: {state}")
                self.current_states.append(None)
                continue

            if isinstance(state, dict):
                state["_prev_state"] = copy.deepcopy(state)
                self.current_states.append(state)
            else:
                print(f"[reset_all] 环境 {i} 返回非法状态: {state}")
                self.current_states.append(None)

        # 安全地创建地图
        for i, (env, state) in enumerate(zip(self.envs, self.current_states)):
            if state is not None and "payload" in state:
                env.make("bomberland-env", state["payload"])
            else:
                print(f"[reset_all] 警告：跳过第 {i} 个环境的 make 创建")


    async def step_all(self, all_actions):
        """执行所有环境的一步动作，并更新当前状态"""
        results = await asyncio.gather(*[
            env.step(to_serializable(actions)) if actions else asyncio.sleep(0, result=(None, True, {}))
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
            else:
                print(f"[step_all] 警告：第 {i} 个环境返回非法 new_state: {new_state}")
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

