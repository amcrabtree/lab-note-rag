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
pip install notebook
jupyter notebook
```

### Run in CLI

Optionally, use your own prompt for extracting text by using the `--prompt` argument. 

```sh
python3 pdf_to_text_gpt4o.py \
  --notebook "test/sample_notebook.pdf" \
  --output_file "test/sample_notebook.md" \
  --api_key "sk-proj-8Q5VuL4........" 
```
