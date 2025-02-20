# lab-note-rag
A RAG for lab notebooks where the user can ask questions about previous experiments. 

Run in jupyter notebook or in CLI. 

### Install libraries

```sh
cd lab-note-rag
pip install -r requirements.txt
```

### Run in Jupyter notebook

After you open Jupyter Notebook, select `GPT-4o extraction.ipynb`

```sh
cd lab-note-rag
pip install notebook ipywidgets
jupyter notebook
```

### Run in CLI

1. First, convert all lab notebooks to markdown documents (.md files). If you'd like, use your own prompt for extracting text by using the `--prompt` argument. 

```sh
python3 pdf_to_text_gpt4o.py \
  --notebook "test/sample_notebook.pdf" \
  --output_file "test/sample_notebook.md" \
  --api_key "sk-proj-8Q5VuL4........" \
  --book_id "LF Book 1"
```


2. Convert all .md files of extracted text into a vector database for the chatbot to use RAG (Retrieval Augmented Generation). 

```sh
python3 database.py \
  --markdown_dir "./test" \
  --output_dir "./test/notebook_database"
```


3. Chat away. 

```sh
ollama pull <model-name> 

python3 lab_rag_query.py \
  --database "./test/notebook_database"
```
