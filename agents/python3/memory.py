class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.logprobs = []

    def store(self, state, action, reward, done, logprob):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.logprobs.append(logprob)

    def clear(self):
        self.__init__()
