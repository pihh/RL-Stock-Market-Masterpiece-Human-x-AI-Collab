
import inspect
from src.utils.db import ConfigurableMixin

class BaseModel(ConfigurableMixin):
    def __init__(self,model, config, db_conn=None):
        self.config = config
        self.config["model"]=model.__name__ 
        self.db_id = None
        if db_conn is not None:
            self.db_id = self.register_model(
                
                model_type=self.__class__.__name__,
                path=inspect.getfile(self.__class__),
                config=self.config
            )