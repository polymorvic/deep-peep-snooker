from pydantic import BaseModel


class SnookerModel(BaseModel, arbitrary_types_allowed=True):
    pass

