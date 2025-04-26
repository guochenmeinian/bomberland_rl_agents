import asyncio
import traceback
from env.gym import Gym
import json
import random
from env.game_state import GameState

# 由于这是一个长期和需要稳定运行的 RL 训练，我们需要一个稳定的环境。
# 为了实现这一点，我们将使用一个名为 SafeGym 的类，它将封装 Gym 类，并提供一些额外的功能，如重试机制和状态转换。
class SafeGym:
    def __init__(self, uri):
        self.gym = Gym(uri)
        self.connected = False
        self.env = None
        self.current_state = None
        self.game_state = None
        self.admin_uri = uri.replace("?role=admin", "?role=admin&agentId=admin")

    async def connect(self, retries=5):
        for i in range(retries):
            try:
                print(f"尝试连接 ({i+1}/{retries})...")
                await self.gym.connect()
                self.connected = True
                
                # 初始化 GameState 对象用于获取游戏状态
                self.game_state = GameState(self.admin_uri)
                await self.game_state.connect()
                
                # 设置回调函数来接收游戏状态
                self.game_state.set_game_tick_callback(self._on_game_tick)
                
                # 启动消息处理循环
                loop = asyncio.get_event_loop()
                loop.create_task(self.game_state._handle_messages(self.game_state.connection))
                
                print("连接成功！")
                return
            except Exception as e:
                print(f"连接失败 ({i+1}/{retries}): {str(e)}")
                await asyncio.sleep(2)
        raise ConnectionError("无法连接到服务器")
    
    async def _on_game_tick(self, tick_number, state):
        # 这个回调函数会在每个游戏帧被调用
        # 我们可以在这里更新当前状态
        self.current_state = {"type": "game_state", "payload": state}
    
    async def reset_game(self):
        print("尝试重置游戏地图")
        
        # 1. 先获取当前状态作为备份
        backup_state = self.current_state
        
        # 2. 发送重置消息
        reset_message = {
            "type": "request_game_reset",
            "world_seed": random.randint(0, 9007199254740991),
            "prng_seed": random.randint(0, 9007199254740991)
        }
        
        # 使用 ForwardModel 发送重置消息
        await self.gym._client_fwd.connection.send(json.dumps(reset_message))
        
        # 3. 增加等待时间
        max_wait_time = 5  # 增加到10秒
        wait_time = 0
        wait_interval = 0.2  # 增加间隔时间
        
        # 4. 设置一个标志，表示我们正在等待新状态
        old_state = self.current_state
        
        # 5. 等待状态更新
        while wait_time < max_wait_time:
            await asyncio.sleep(wait_interval)
            wait_time += wait_interval
            
            # 检查状态是否已更新（与旧状态不同）
            if self.current_state is not None and self.current_state != old_state:
                print(f"成功获取新状态，用时 {wait_time:.1f} 秒")
                return self.current_state
        
        # 6. 如果超时，尝试直接从游戏状态获取
        print("通过回调获取状态超时，尝试直接从游戏状态获取...")
        
        if self.game_state._state is not None:
            print("成功从 GameState 获取状态")
            self.current_state = {"type": "game_state", "payload": self.game_state._state}
            return self.current_state
        
        # 7. 如果仍然无法获取，使用备份状态或默认状态
        if backup_state is not None:
            print("使用备份状态")
            return backup_state
        
        # 8. 最后才使用默认状态
        print("警告：无法获取任何状态，使用默认状态")
        from train import load_initial_state
        default_state = load_initial_state()
        self.current_state = {"type": "game_state", "payload": default_state}
        
        print("游戏地图重置完成")
        return self.current_state
    
    def set_state(self, new_state):
        self._state = new_state

    
    def make(self, name, initial_state):
        print("创建环境...")
        self.env = self.gym.make(name, initial_state)
        print("环境创建成功")
        return self.env
    
    async def reset(self):
        if self.connected and self.env:
            print("重置环境...")
            await self.env.reset()
            print("环境重置成功")
            
    async def step(self, actions):
        try:
            if not self.env:
                raise ValueError("环境未初始化，请先调用make方法")
                
            print(f"执行动作: {actions}")
            
            # 使用标准步进方法
            observation, done, info = await self.env.step(actions)
            
            # 更新当前状态
            if observation is not None:
                self.current_state = {"type": "game_state", "payload": observation}
                
                # 打印状态信息
                for action in actions:
                    if 'action' in action and 'unit_id' in action['action']:
                        unit_id = action['action']['unit_id']
                        if unit_id in observation["unit_state"]:
                            unit_pos = observation["unit_state"][unit_id]["coordinates"]
                            print(f"单位 {unit_id} 位置: {unit_pos}")
                
                print(f"游戏时钟: {observation['tick']}")
                print(f"事件: {info}")
                
            return observation, done, info

        except Exception as e:
            print(f"Step 错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, True, {}
            
    async def close(self):
        if self.connected:
            print("关闭环境连接...")
            if self.game_state:
                # 关闭 GameState 连接
                await self.game_state.connection.close()
            await self.gym.close()
            print("环境连接已关闭")