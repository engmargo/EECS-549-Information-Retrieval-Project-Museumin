# Information-Retrieval-Project-Museumin
A National Wide Museum Search Engine

## Running the Project

1. Clone the data from `./data_1101/`, home html file from `./web/`, all image and JS file from `./static/`, 
2. Clone pipeline components, `document_preprocessor.py`,`indexing.py`,`vector_ranker.py`,`ranker.py`,`l2r.py`,`relevance.py`,`blocks.py`, and fastapi files, `models.py`,`app.py`
3. Ensure fastapi library is installed.
4. Run the main script with `fastapi dev app.py`

## Inputs Note
Output list is randomly organized when query is empty and multiple selctions exist.

## Demo
<img width="1139" alt="Screenshot 2024-12-06 at 8 24 13 PM" src="https://github.com/user-attachments/assets/b0dceb4a-7358-4e4f-8c93-0f6ba15639c0">

## Technologies Used
- Langchain
- HuggingFace
- Scikit-Learn
- FastAPI
- JavaScript
- HTML
