import os
from PyPDF2 import PdfReader
from semantic_text_splitter import CharacterTextSplitter
from openai import OpenAI
from evalutator import Evaluator


source_dir = "./data/source"
raw_dir = "./data/raw"


def chunk_text(text):
    splitter = CharacterTextSplitter(trim_chunks=False)
    chunks = splitter.chunks(text, chunk_capacity=2000)
    return chunks


def extract_text(pdf_path):
    text = ""
    reader = PdfReader(pdf_path)
    number_of_pages = len(reader.pages)
    print("Number of pages: " + str(number_of_pages))
    for page in reader.pages:
        text += page.extract_text()

    return text


def pdf_to_text():
    # Get a list of all pdf files in the source directory
    pdf_files = [f for f in os.listdir(source_dir) if f.endswith(".pdf")]

    # Process each pdf file
    for pdf_file in pdf_files:
        pdf_path = os.path.join(source_dir, pdf_file)
        extracted_text = extract_text(pdf_path)

        # Save the extracted text to a file with the same name as the pdf but with a txt extension in the /data/raw folder
        txt_file = pdf_file.replace(".pdf", ".txt")
        txt_path = os.path.join(raw_dir, txt_file)

        with open(txt_path, "w") as f:
            f.write(extracted_text)


def create_flashcards(chunk: str):
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
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


# Create a function that takes a list of chunks and creates flashcards for each chunk and then saves the flashcards to a file
def create_flashcards_from_chunks(text_file: str):
    with open(text_file, "r") as f:
        text = f.read()
    chunks = chunk_text(text)
    flash_cards = []
    for chunk in chunks:
        card = Evaluator(chunk=chunk, content=create_flashcards(chunk)).evaluate()
        flash_cards.append(card)

    # Save the flashcards to a file
    flashcards_path = "./data/raw/testflashcards.txt"
    with open(flashcards_path, "w") as f:
        # rest of your code
        for card in flash_cards:
            f.write(str(card.model_dump()) + "\n")


create_flashcards_from_chunks("./data/raw/Effective Software Testing Chapter 1.txt")
