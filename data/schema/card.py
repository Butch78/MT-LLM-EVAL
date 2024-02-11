from pydantic import BaseModel


class Card(BaseModel):
    chunk: str
    content: str


class RougeScore(BaseModel):
    rouge1: float
    rouge2: float
    rougeL: float


class BertScore(BaseModel):
    P: float
    R: float
    F1: float


class Evalutation(BaseModel):
    chunk: str
    content: str
    rouge1: float
    rouge2: float
    rougeL: float
    bert_score_P: float
    bert_score_R: float
    bert_score_F1: float
    similarity_score: float
