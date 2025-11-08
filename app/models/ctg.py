from pydantic import BaseModel

class CTGData(BaseModel):
    time: int
    fhr: int
    contraction: float
