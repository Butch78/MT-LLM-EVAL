import os
from PyPDF2 import PdfReader
from semantic_text_splitter import CharacterTextSplitter
from openai import OpenAI
import polars as pl
import concurrent.futures


source_dir = "./data/source"
raw_dir = "./data/raw"

CHUNK_SIZE = 500


def pdf_to_chunks():
    # Get a list of all pdf files in the source directory
    text = ""
    pdf_files = [f for f in os.listdir(source_dir) if f.endswith(".pdf")]

    # Process each pdf file
    for pdf_file in pdf_files:
        pdf_path = os.path.join(source_dir, pdf_file)

        # Read the PDf file
        reader = PdfReader(pdf_path)
        number_of_pages = len(reader.pages)
        print("Number of pages: " + str(number_of_pages))
        for page in reader.pages:
            text += page.extract_text()

        # Semantically Split the text into chunks
        splitter = CharacterTextSplitter(trim_chunks=False)
        chunks = splitter.chunks(text, chunk_capacity=CHUNK_SIZE)
        print("Number of chunks: " + str(len(chunks)))

        # Save the extracted text to a file with the same name as the pdf but with a parquet extension in the /data/raw folder
        parquet_file = pdf_file.replace(".pdf", f"_{CHUNK_SIZE}.parquet")
        parquet_path = os.path.join(raw_dir, parquet_file)


        df = pl.DataFrame({"chunk": chunks})
        df.write_parquet(parquet_path)


def create_flashcards(chapter_name: str, parquet_file_path: str, n_threads: int = 4):
    """
    This function takes a chunk of text and uses the OpenAI API to create flashcards from the text
    """
    client = OpenAI()

    df = pl.read_parquet(parquet_file_path)
    chunks = df["chunks"].to_list()
    flash_cards = []

    def fetch_fn(chunk: str) -> str:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            #  https://platform.openai.com/docs/guides/text-generation/json-mode
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": """You are an Expert ANKI Card Maker, You are able to make great content for your users from any text, It is only relevant to the topic at hand. You output the ANKI cards in the following JSON format. {"Front": "Question About Content", "Back": "Answer to Question about Content"}""",
                },
                {
                    "role": "user",
                    "content": chunk,
                },
            ],
        )
        # Using ChatCompletionMessage parse the response into a dictionary
        # Then use the dictionary to create a Card object
        # Then return the Card object

        return response.choices[0].message.content

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
        flashcards = list(executor.map(fetch_fn, chunks))

    dicts = [card.model_dump() for card in flash_cards]
    df = pl.DataFrame(dicts)
    df.write_parquet(f"./data/interim/{chapter_name}.parquet")

    return flashcards


# Call the pdf_to_chunks function
pdf_to_chunks()


# create_flashcards_from_chunks(
#     "chapter_1", "./data/raw/Effective Software Testing Chapter 1.txt"
# )
# evalute_flashcards("chapter_1_2000.parquet")
