
import inspect
from src.utils.db import ConfigurableMixin


class BaseModel(ConfigurableMixin):
    def __init__(self, model, config):
        print('xxxxxxxxx', model, config)
        self.config = config
        self.config["model"] = model.__name__

        self.db_id = None
        if self.conn is not None:
            self.db_id = self.register_model(
                model_type=self.__class__.__name__,
                path=inspect.getfile(self.__class__),
                config=self.config
            )



class TransformerPpo(ConfigurableMixin):
    def __init__(self,model,policy,env,model_config={},policy_config={}, run_config={}):
        self.config = {
            "model_config":model_config,
            "policy_config":policy_config,
            "model":model.__name__,
            "policy":policy if type(policy)== str else policy.__name__
        }
        self.db_id = None
        if self.conn is not None:
            self.db_id = self.register_model(
                model_type=self.__class__.__name__,
                path=inspect.getfile(self.__class__),
                config=self.config
            )
        
        agent = model(
            policy=policy,
            env=env,
            **model_config,
            **run_config
        )
        self.agent = agent
        
    
