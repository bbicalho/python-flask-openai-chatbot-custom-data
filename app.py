from flask import Flask, render_template, request, jsonify, make_response, redirect, url_for
import os
import json
from datetime import datetime
import shutil

import requests
import re
import urllib.request
from bs4 import BeautifulSoup
from collections import deque
from html.parser import HTMLParser
from urllib.parse import urlparse
import os
import pandas as pd
import tiktoken
import openai
import numpy as np
from openai.embeddings_utils import distances_from_embeddings, cosine_similarity
from ast import literal_eval
from dotenv import load_dotenv

from PyPDF2 import PdfReader

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(app.root_path, 'receivedfiles')
ALLOWED_EXTENSIONS = {'pdf', 'txt'}
EXECUTIONS_DATA_FILE = os.path.join(app.root_path, 'training-executions-data.json')
PROMPTS_VALUES_FILE = os.path.join(app.root_path, 'prompts-values.json')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  * 10 # Set maximum file size to 160MB

# Helper function to read received files information
def get_received_files_info():
    files_info = []
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        if filename.lower().endswith('.pdf') or filename.lower().endswith('.txt'):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
            file_date = datetime.fromtimestamp(os.path.getmtime(file_path))
            files_info.append({'filename': filename, 'size': round(file_size, 2), 'datetime': file_date})
    return files_info

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/chatbot', methods=['GET'])
def chatbot():
    return render_template('chatbot2.html')

@app.route('/process_answer', methods=['POST'])
def process_answer():
    user_question = request.json['question']
    # Call your process_answer function here and get the bot's answer
    # Replace the example response with the actual bot's answer

    df=pd.read_csv('processed/embeddings.csv', index_col=0)
    df['embeddings'] = df['embeddings'].apply(literal_eval).apply(np.array)

    bot_answer = answer_question(df, question=user_question)
    # bot_answer = "This is an example response from the bot."

    return jsonify({'answer': bot_answer})


@app.route('/receivedfiles', methods=['GET', 'POST'])
def received_files():
    if request.method == 'POST':
        # Check if the post request has file parts
        if 'file' not in request.files:
            return 'No file part in the request', 400

        files = request.files.getlist('file')

        # If the user does not select any files, return an error
        if not files:
            return 'No selected files', 400

        # Save the files to the receivedfiles folder
        for file in files:
            if file.filename == '':
                return 'No selected file', 400
            file.save(os.path.join('receivedfiles', file.filename))

        return redirect(url_for('received_files'))

    files_info = get_received_files_info()
    return render_template('received_files.html', files_info=files_info)

@app.route('/chatbotprompts', methods=['GET', 'POST'])
def chatbot_prompts():
    if request.method == 'POST':
        data = {
            'prompt-role-purpose': request.form.get('prompt-role-purpose'),
            'prompt-dont-know-text': request.form.get('prompt-dont-know-text')
        }
        with open(PROMPTS_VALUES_FILE, 'w') as file:
            json.dump(data, file)
    with open(PROMPTS_VALUES_FILE, 'r', encoding="utf-8") as file:
        prompts_values = json.load(file)
    return render_template('chatbot_prompts.html', prompts_values=prompts_values)



@app.route('/trainchatbot', methods=['GET', 'POST'])
def train_chatbot():

    # Set Cache-Control and Pragma headers to prevent caching

    if request.method == 'POST':
        received_files = os.listdir('receivedfiles')
        if not received_files:
            return jsonify({'error': 'No files available to train'})
        
        start_chatbot_training()
        return redirect(url_for('train_chatbot'))  # Redirect back to the same page after POST
    
    recent_files_info = get_received_files_info()

    with open(EXECUTIONS_DATA_FILE, 'r') as file:
        training_data = json.load(file)

    response = make_response(render_template('train_chatbot.html', training_data=training_data, files_info=recent_files_info))
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    
    # Set an Expires header with a past date to ensure the content is stale immediately
    response.headers['Expires'] = '0'

    return response

def start_chatbot_training():
    received_files_path = app.config['UPLOAD_FOLDER']
    received_files = [f for f in os.listdir(received_files_path) if f.lower().endswith(('.pdf', '.txt'))]

    # Calculate the elapsed time
    start_time = datetime.now()
    # Your chatbot training function here...
    # Replace the following line with the actual chatbot training code
    # (This is a dummy line to simulate training time)
    import time
    # time.sleep(5)  # Simulate training time of 5 seconds
    
    move_files_to_pdf_folder()
    input_folder_path = 'pdf-folder'  # Replace with the path to the folder containing PDF files
    output_folder_path = 'text/pdf'  # Replace with the path to the output folder
    pdf_to_txt(input_folder_path, output_folder_path)
    process_pdf_txt()
    do_embeddings()
        
    end_time = datetime.now()
    elapsed_time = (end_time - start_time).seconds

    # Add a new row to training data
    new_execution = {
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "files": received_files,
        "elapsed_time": elapsed_time
    }
    with open(EXECUTIONS_DATA_FILE, 'r+') as file:
        data = json.load(file)
        data.append(new_execution)
        file.seek(0)
        json.dump(data, file, indent=4)
        file.truncate()





def pdf_to_txt(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get a list of all PDF files in the input folder
    pdf_files = [file for file in os.listdir(input_folder) if file.lower().endswith('.pdf')]

    for pdf_file in pdf_files:
        pdf_path = os.path.join(input_folder, pdf_file)
        txt_path = os.path.join(output_folder, pdf_file.replace('.pdf', '.txt'))

        with open(pdf_path, 'rb') as pdf_file:
            pdf_reader = PdfReader(pdf_file)

            with open(txt_path, 'w', encoding='utf-8') as txt_file:
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    txt_file.write(page.extract_text())

        print(f"Converted {pdf_file} to {os.path.basename(txt_path)}")

# # Example usage
# input_folder_path = 'pdf-folder'  # Replace with the path to the folder containing PDF files
# output_folder_path = 'text/pdf'  # Replace with the path to the output folder
# pdf_to_txt(input_folder_path, output_folder_path)


def move_files_to_pdf_folder():
    
    source_folder = "receivedfiles"
    destination_folder = "pdf-folder"

    # Check if the destination folder exists and create it if not
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Get a list of all PDF and TXT files in the source folder
    file_list = [file for file in os.listdir(source_folder) if file.lower().endswith(('.pdf', '.txt'))]

    # Move each file to the destination folder
    for file in file_list:
        source_path = os.path.join(source_folder, file)
        destination_path = os.path.join(destination_folder, file)
        shutil.move(source_path, destination_path)

# Call the function to move the files
# move_files_to_pdf_folder()

def remove_newlines(serie):
    serie = serie.str.replace('\n', ' ')
    serie = serie.str.replace('\\n', ' ')
    serie = serie.str.replace('  ', ' ')
    serie = serie.str.replace('  ', ' ')
    return serie


# Function to split the text into chunks of a maximum number of tokens
max_tokens = 1500
openai.api_key = os.environ.get('API_KEY')

def split_into_many(text, max_tokens = max_tokens):

    tokenizer = tiktoken.get_encoding("cl100k_base")
    # Split the text into sentences
    sentences = text.split('. ')

    # Get the number of tokens for each sentence
    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]
    
    chunks = []
    tokens_so_far = 0
    chunk = []

    # Loop through the sentences and tokens joined together in a tuple
    for sentence, token in zip(sentences, n_tokens):

        # If the number of tokens so far plus the number of tokens in the current sentence is greater 
        # than the max number of tokens, then add the chunk to the list of chunks and reset
        # the chunk and tokens so far
        if tokens_so_far + token > max_tokens:
            chunks.append(". ".join(chunk) + ".")
            chunk = []
            tokens_so_far = 0

        # If the number of tokens in the current sentence is greater than the max number of 
        # tokens, go to the next sentence
        if token > max_tokens:
            continue

        # Otherwise, add the sentence to the chunk and add the number of tokens to the total
        chunk.append(sentence)
        tokens_so_far += token + 1
        
    # Add the last chunk to the list of chunks
    if chunk:
        chunks.append(". ".join(chunk) + ".")

    return chunks
    
def process_pdf_txt():

    # Create a list to store the text files
    texts=[]

    domain = 'pdf'

    # Get all the text files in the text directory
    for file in os.listdir("text/" + domain + "/"):

        # Open the file and read the text
        with open("text/" + domain + "/" + file, "r", encoding="UTF-8") as f:
            text = f.read()

            # Omit the first 11 lines and the last 4 lines, then replace -, _, and #update with spaces.
            texts.append((file[11:-4].replace('-',' ').replace('_', ' ').replace('#update',''), text))

    # Create a dataframe from the list of texts
    df = pd.DataFrame(texts, columns = ['fname', 'text'])

    # Check if the destination folder exists and create it if not
    if not os.path.exists('processed'):
        os.makedirs('processed')

    # Set the text column to be the raw text with the newlines removed
    df['text'] = df.fname + ". " + remove_newlines(df.text)
    df.to_csv('processed/scraped.csv')
    df.head()

def do_embeddings():

    max_tokens = 1500

    # Load the cl100k_base tokenizer which is designed to work with the ada-002 model
    tokenizer = tiktoken.get_encoding("cl100k_base")

    df = pd.read_csv('processed/scraped.csv', index_col=0)
    df.columns = ['title', 'text']

    # Tokenize the text and save the number of tokens to a new column
    df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))

    # Visualize the distribution of the number of tokens per row using a histogram
    # df.n_tokens.hist()

    shortened = []
    # Loop through the dataframe
    for row in df.iterrows():

        # If the text is None, go to the next row
        if row[1]['text'] is None:
            continue

        # If the number of tokens is greater than the max number of tokens, split the text into chunks
        if row[1]['n_tokens'] > max_tokens:
            shortened += split_into_many(row[1]['text'])
        
        # Otherwise, add the text to the list of shortened texts
        else:
            shortened.append( row[1]['text'] )

    ################################################################################
    ### Step 9
    ################################################################################

    df = pd.DataFrame(shortened, columns = ['text'])
    df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))
    # df.n_tokens.hist()

    ################################################################################
    ### Step 10
    ################################################################################

    # Note that you may run into rate limit issues depending on how many files you try to embed
    # Please check out our rate limit guide to learn more on how to handle this: https://platform.openai.com/docs/guides/rate-limits

    df['embeddings'] = df.text.apply(lambda x: openai.Embedding.create(input=x, engine='text-embedding-ada-002')['data'][0]['embedding'])
    df.to_csv('processed/embeddings.csv')

    # df.head()
    

def create_context(
    question, df, max_len=3800, size="ada"
):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    # Get the embeddings for the question
    q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']

    # Get the distances from the embeddings
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')


    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():
        
        # Add the length of the text to the current length
        cur_len += row['n_tokens'] + 4
        
        # If the context is too long, break
        if cur_len > max_len:
            break
        
        # Else add it to the text that is being returned
        returns.append(row["text"])

    # Return the context
    return "\n\n###\n\n".join(returns)

def answer_question(
    df,
    model="text-davinci-003",
    question="Am I allowed to publish model outputs to Twitter, without a human review?",
    max_len=1800,
    size="ada",
    debug=False,
    max_tokens=1500,
    stop_sequence=None
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context = create_context(
        question,
        df,
        max_len=max_len,
        size=size,
    )
    # If debug, print the raw model response
    if debug:
        print("Context:\n" + context)
        print("\n\n")

    try:
        
        # {"prompt-role-purpose": "Responda à pergunta com base no contexto abaixo e, se a pergunta não puder ser respondida com base no contexto, responda ", "prompt-dont-know-text": "pô brother, aí tu me fudeu...."}
        file_path = "prompts-values.json"

        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        prompt_dont_know_text = data.get("prompt-dont-know-text", "")
        prompt_role_purpose = data.get("prompt-role-purpose", "")
        print(prompt_dont_know_text)
        print(prompt_role_purpose)

        # Create a completions using the questin and context
        response = openai.Completion.create(
            prompt=f"{prompt_role_purpose} \"{prompt_dont_know_text}\"\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:",
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            model=model,
        )
        return response["choices"][0]["text"].strip()
    except Exception as e:
        print(e)
        return ""

if __name__ == '__main__':
    app.run()

