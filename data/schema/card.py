from pydantic import BaseModel


class Section(BaseModel):
    section_heading: str
    text: str
    character_count: int = 0
    token_count: int = 0


class Flashcard(BaseModel):
    section_heading: str
    text: str
    content: str
    character_count: int = 0
    token_count: int = 0
    flash_card_content: str = ""


class RougeScore(BaseModel):
    rouge1: float
    rouge2: float
    rougeL: float


class BertScore(BaseModel):
    P: float
    R: float
    F1: float


class Evalutation(BaseModel):
    character_count: int
    token_count: int
    section_heading: str
    section: str
    content: str
    rouge1: float
    rouge2: float
    rougeL: float
    bleu_score: float
    bert_score_P: float
    bert_score_R: float
    bert_score_F1: float
    similarity_score: float
