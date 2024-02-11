import os
from PyPDF2 import PdfReader
from semantic_text_splitter import CharacterTextSplitter
from openai import OpenAI
import polars as pl
import concurrent.futures

from data.schema.card import Card
from evalutator import Evaluator

# The PDFs documents are stored in the source directory
source_dir = "./data/source"
# The extracted text from the PDFs will be stored in the raw directory
raw_dir = "./data/raw"

# Any smaller and it would create to many cards per chapter
CHUNK_SIZE = 2000


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
        print(f"{pdf_path} Number of pages: " + str(number_of_pages))
        for page in reader.pages:
            text += page.extract_text()

        # Save the extracted text to a text file with the same name as the pdf but with a txt extension in the /data/raw folder
        text_file = pdf_file.replace(".pdf", ".txt")
        text_path = os.path.join(raw_dir, text_file)
        with open(text_path, "w") as file:
            file.write(text)

        # Semantically Split the text into chunks
        splitter = CharacterTextSplitter(trim_chunks=True)
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
    chunks = df["chunk"].to_list()
    flash_cards = []

    def fetch_fn(chunk: str) -> str:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            #  https://platform.openai.com/docs/guides/text-generation/json-mode
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": """You are an Expert ANKI Card Maker, You are able to make great content for your users from any text, It is only relevant to the topic at hand. 
                    You are able to comprehend the whole content and find the most important and difficult part of the content, Using the ost relevalent and important part of the content,
                    You output the ANKI cards in the following JSON format. {"Front": "Question About Content", "Back": "Answer to Question about Content"}""",
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

        return Card(chunk=chunk, content=response.choices[0].message.content)

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
        flash_cards = list(executor.map(fetch_fn, chunks))

    print(f"Number of flashcards: {len(flash_cards)}")
    print(f"First flashcard: {flash_cards[0].model_dump()}")

    dicts = [card.model_dump() for card in flash_cards]
    df = pl.DataFrame(dicts)
    df.write_parquet(f"./data/interim/flashcards_{chapter_name}_{CHUNK_SIZE}.parquet")

    return flash_cards

    # Call the pdf_to_chunks function
    # pdf_to_chunks()


def evalute_flashcards(chapter_name: str, parquet_file_path: str, n_threads: int = 4):
    df = pl.read_parquet(parquet_file_path)
    evaluations = []
    for card in df.rows(named=True):
        evaluator = Evaluator(chunk=card["chunk"], content=card["content"])
        evaluations.append(evaluator.evaluate())

    #  Export the evaluations to a parquet file
    df = pl.DataFrame(evaluations)

    #  Get the count of the evaluations
    df.write_parquet(
        f"./data/interim/evaluations_{chapter_name}_{str(len(evaluations))}.parquet"
    )

    print(f"Number of evaluations: {len(evaluations)}")


# Call the pdf_to_chunks function to extract the text from the pdfs and save them to parquet files
pdf_to_chunks()

# Call the create_flashcards function to create flashcards from the chunks of text
create_flashcards(
    "chapter_1",
    f"./data/raw/Effective Software Testing Chapter 1_{CHUNK_SIZE}.parquet",
)

# Call the evalute_flashcards function to evaluate the flashcards
evalute_flashcards(
    "chapter_1", f"./data/interim/flashcards_chapter_1_{CHUNK_SIZE}.parquet"
)
