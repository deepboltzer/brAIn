class AttackData():
    """Simple utility class to hold data."""
    def __init__(self):
        self._last_obs = []
        self._last_act = []
        self._last_rew = 0
        self._last_done = False
        self._last_info = {}
    
    @property
    def last_obs(self):
        return self._last_obs
    
    @last_obs.setter
    def last_obs(self, obs):
        self._last_obs = obs
    
    @property
    def last_act(self):
        return self._last_act
    
    @last_act.setter
    def last_act(self, act):
        self._last_act = act
    
    @property
    def last_rew(self):
        return self._last_rew
    
    @last_rew.setter
    def last_rew(self, rew):
        self._last_rew = rew
    
    @property
    def last_done(self):
        return self._last_done
    
    @last_done.setter
    def last_done(self, done):
        self._last_done = done
    
    @property
    def last_info(self):
        return self._last_info
    
    @last_info.setter
    def last_info(self, info):
        self._last_info = info