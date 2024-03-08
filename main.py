import os
from openai import OpenAI
import polars as pl

from evalutator import Evaluator


# The PDFs documents are stored in the source directory
source_dir = "./data/source"
# The extracted text from the PDFs will be stored in the raw directory
raw_dir = "./data/raw"


def fetch_fn(section: str) -> str:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        temperature=1.5,
        #  https://platform.openai.com/docs/guides/text-generation/json-mode
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": """You are an Expert ANKI Card Maker, You are able to make great content for your users from any text, It is only relevant to the topic at hand. 
                    You are able to comprehend the whole content and find the most important and difficult part of the content, Using the most relevalent and important part of the content,
                    You output the ANKI cards in the following JSON format. {"Front": "Question About Content", "Back": "Answer to Question about Content"}""",
            },
            {
                "role": "user",
                "content": section,
            },
        ],
    )
    # Using ChatCompletionMessage parse the response into a dictionary
    # Then use the dictionary to create a Card object
    # Then return the Card object

    return response.choices[0].message.content


def create_flashcards(parquet_file_path: str):
    """
    This function takes a file path, reads the parquert and uses the OpenAI API to create flashcards from the text and
      then writes the flashcards to a parquet file
    """
    pl.read_parquet(parquet_file_path).with_columns(
        pl.col("text").map_elements(fetch_fn, return_dtype=pl.Utf8).alias("content")
    ).write_parquet("./data/final/gpt_4_1106_15_flashcards_chapter_2_sections.parquet")


def evalute_flashcards(chapter_name: str, parquet_file_path: str, n_threads: int = 4):
    df = pl.read_parquet(parquet_file_path)

    evaluations = []
    for card in df.rows(named=True):
        evaluator = Evaluator(card=card)
        evaluations.append(evaluator.evaluate())

    #  Export the evaluations to a parquet file
    df = pl.DataFrame(evaluations)

    #  Get the count of the evaluations
    df.write_parquet(f"./data/interim/evaluations_{chapter_name}.parquet")


# Call the create_flashcards function to create flashcards from the chunks of text
create_flashcards(
    "./data/final/chapter_2_sections_filtered.parquet",
)

# # Call the evalute_flashcards function to evaluate the flashcards


# GTP-4-1106
# evalute_flashcards(
#     "gpt_4_1106_chapter_1",
#     "./data/interim/gpt_4_1106_flashcards_chapter_1_sections.parquet",
# )
# evalute_flashcards(
#     "gpt_4_1106_chapter_2",
#     "./data/interim/gpt_4_1106_flashcards_chapter_2_sections.parquet",
# )
# evalute_flashcards(
#     "gpt_4_1106_chapter_3",
#     "./data/interim/gpt_4_1106_flashcards_chapter_3_sections.parquet",
# )


# # GPT-3-1106
# evalute_flashcards(
#     "chapter_1",
#     "./data/interim/flashcards_chapter_1_sections.parquet",
# )
# evalute_flashcards(
#     "chapter_2",
#     "./data/interim/flashcards_chapter_2_sections.parquet",
# )
# evalute_flashcards(
#     "chapter_3",
#     "./data/interim/flashcards_chapter_3_sections.parquet",
# )
