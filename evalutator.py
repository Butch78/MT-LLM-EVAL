from rouge_score import rouge_scorer
from bert_score import score
from sentence_transformers import SentenceTransformer, util

# from bart_score import BARTScorer
from data.schema.card import Evalutation, RougeScore, BertScore


class Evaluator:
    def __init__(self, chunk: str, content: str):
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )
        self.bert_scorer = score
        # self.bart_scorer = BARTScorer()
        self.chunk = chunk
        self.content = content

    def get_rouge_score(self, reference_summary, candidate_summary) -> RougeScore:
        """
        Rough Score is a measure of the overlap between the candidate summary and the reference summary

        Args:
            reference_summary (str): The reference summary
            candidate_summary (str): The candidate summary

        """
        scores = self.rouge_scorer.score(reference_summary, candidate_summary)
        print("Rouge Scores: ", scores)
        return RougeScore(
            rouge1=scores["rouge1"].fmeasure,
            rouge2=scores["rouge2"].fmeasure,
            rougeL=scores["rougeL"].fmeasure,
        )

    def get_bert_score(self, reference, candidate) -> BertScore:
        P, R, F1 = self.bert_scorer(
            [candidate], [reference], lang="en", rescale_with_baseline=True
        )
        print("P: ", P)
        print("R: ", R)
        print("F1: ", F1)
        return BertScore(P=P.item(), R=R.item(), F1=F1.item())

    # def get_bart_score(self, reference, candidate):
    #     score = self.bart_scorer.score(candidate, reference)
    #     print("Bart Score: ", score)
    #     return score

    def get_similarity_score(self, reference, candidate):
        model = SentenceTransformer("stsb-roberta-large")
        reference_embedding = model.encode(reference, convert_to_tensor=True)
        candidate_embedding = model.encode(candidate, convert_to_tensor=True)
        cosine_score = util.pytorch_cos_sim(reference_embedding, candidate_embedding)
        print("Cosine Score: ", cosine_score)
        return cosine_score.item()

    def evaluate(self) -> Evalutation:
        # bart_score = self.get_bart_score(
        #     reference=self.chunk, candidate=self.content
        # )

        rouge_score = self.get_rouge_score(
            reference_summary=self.chunk, candidate_summary=self.content
        )
        bert_score = self.get_bert_score(
            reference=self.chunk, candidate=self.content
        )

        return Evalutation(
            chunk=self.chunk,
            content=self.content,
            rouge1=rouge_score.rouge1,
            rouge2=rouge_score.rouge2,
            rougeL=rouge_score.rougeL,
            bert_score_P=bert_score.P,
            bert_score_R=bert_score.R,
            bert_score_F1=bert_score.F1,
            # bart_score=bart_score,
            similarity_score=self.get_similarity_score(
                reference=self.chunk, candidate=self.content
            ),
        )
