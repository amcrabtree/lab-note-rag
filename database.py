import os
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import argparse

EMBEDDING_MODEL = "BAAI/bge-base-en"  


def load_markdown_files(directory):
    texts = []
    filenames = []
    for file in os.listdir(directory):
        if file.endswith(".md"):
            with open(os.path.join(directory, file), "r", encoding="utf-8") as f:
                texts.append(f.read())
                filenames.append(file)
    return texts, filenames


def save_vector_store(markdown_dir: str, output_dir: str) -> None:
    # Load and process markdown files
    texts, filenames = load_markdown_files(markdown_dir)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = [chunk for text in texts for chunk in text_splitter.split_text(text)]

    # Initialize embedding model
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = FAISS.from_texts(chunks, embeddings)
    vector_store.save_local(output_dir)
    return None


# Argparse
parser = argparse.ArgumentParser(description="Turn PDF lab notebook into Markdown.")
parser.add_argument("--markdown_dir", "-m",
                    help="Path to directory containing lab notebooks as .md files.")
parser.add_argument("--output_dir", "-o", default="notebook_vector_database",
                    help="Path to output directory containing database.")


if __name__ == "__main__":
    args = parser.parse_args()

    save_vector_store(args.markdown_dir, args.output_dir)
    print(f"Database saved to:  {args.output_dir}")