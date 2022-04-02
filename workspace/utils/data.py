class AttackData():
    """Simple utility class to hold data."""
    def __init__(self):
        self._last_obs = None
        self._last_act = None
        self._last_rew = None
        self._last_done = None
        self._last_info = None
    
    @property
    def last_obs(self):
        return self._last_obs
    
    @last_obs.setter
    def set_last_obs(self, obs):
        self._last_obs = obs
    
    @property
    def last_act(self):
        return self._last_act
    
    @last_act.setter
    def set_last_act(self, act):
        self._last_act = act
    
    @property
    def last_rew(self):
        return self._last_rew
    
    @last_rew.setter
    def set_last_rew(self, rew):
        self._last_rew = rew
    
    @property
    def last_done(self):
        return self._last_done
    
    @last_done.setter
    def set_last_done(self, done):
        self._last_done = done
    
    @property
    def last_info(self):
        return self._last_info
    
    @last_info.setter
    def set_last_info(self, info):
        self._last_info = info