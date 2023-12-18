from flask import Flask, render_template, request, jsonify, make_response, redirect, url_for
# import traceback
import os
import sys
import json
from datetime import datetime
import shutil
import pytz

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
# ALLOWED_EXTENSIONS = {'pdf', 'txt'}
# PROMPTS_VALUES_FILE = os.path.join(app.root_path, 'prompts-values.json')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  * 10 # Set maximum file size to 160MB


brasilia_tz = pytz.timezone('America/Sao_Paulo')
current_datetime = datetime.now(brasilia_tz)
file_path = 'log_file-'+current_datetime.strftime('%Y-%m-%d-%H-%M-%S%Z')+'.txt'

# questions_array = []
json_file_path = 'super-adm-link-settings.json'

@app.route('/superadm')
def home():
    return render_template('admin.html')


@app.route('/menu/<id>', methods=['GET'])
def menu(id):
    # print(id)

    
    can_upload,can_train,can_prompt,can_delete = get_link_config(id)

    return render_template('index.html',user_id=id, can_upload=can_upload, can_train=can_train, can_prompt=can_prompt, can_delete=can_delete)


def get_link_config(id):
    # get menu permissions and list only allowed menus
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as file:
            links = json.load(file)
            # return jsonify(links)
    else:
        links = {}

    if id in links:
        can_upload = links[id]['ativaarquivo'+id]
        can_train = links[id]['ativatreinar'+id]
        can_prompt = links[id]['ativaprompt'+id]
        can_delete = links[id]['ativadelete'+id] 
    else:
        can_upload = True
        can_train = True
        can_prompt = True
        can_delete = True

    return can_upload,can_train,can_prompt,can_delete



def get_config_by_id(id):
    # get menu permissions and list only allowed menus
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as file:
            links = json.load(file)
            # return jsonify(links)
    else:
        links = {}

    if id in links:
        can_upload = links[id]['ativaarquivo'+id]
        can_train = links[id]['ativatreinar'+id]
        can_prompt = links[id]['ativaprompt'+id]
        can_delete = links[id]['ativadelete'+id] 
        model = links[id]['model']
    else:
        can_upload = True
        can_train = True
        can_prompt = True
        can_delete = True
        model = 'gpt-3.5-turbo-16k'

    return {"can_upload":can_upload,"can_train":can_train,"can_prompt":can_prompt,"can_delete":can_delete,"model":model}


@app.route('/chatbot/<id>', methods=['GET'])
def chatbot(id):

    questions_file = os.path.join(app.root_path, 'messages/'+id+'/chat-messages.json')
    question_values = {}
    if(os.path.exists(questions_file)):
        with open(questions_file, 'r', encoding="utf-8") as file:
            question_values = json.load(file)

    # questions_array.clear()
    if id not in question_values:
        questions_array = []
    else:    
        # print(question_values[id]['messages'])
        questions_array = question_values[id]['messages']

    # prompt_file = os.path.join(app.root_path, 'prompts/'+id+'/prompts-values.json')
    # prompts_values = {}
    # if(os.path.exists(prompt_file)):
    #     with open(prompt_file, 'r', encoding="utf-8") as file:
    #         prompts_values = json.load(file)

    # print(prompts_values)

    # save_data_to_json({id:{'prompt-role':prompts_values['prompt-role-purpose'],'prompt-dont-know-text':prompts_values['prompt-dont-know-text']}},"messages/"+id+"/chat-messages.json")

    can_upload,can_train,can_prompt,can_delete = get_link_config(id)
    saopaulo_tz = pytz.timezone('America/Sao_Paulo')
    # Get the current datetime in the Sao Paulo timezone
    current_datetime_saopaulo = datetime.now(saopaulo_tz)
    current_datetime_saopaulo_str = current_datetime_saopaulo.strftime('%Y-%m-%d %H:%M:%S')

    return render_template('chatbot2.html', user_id=id, can_delete=can_delete, questions_array=questions_array , current_datetime=current_datetime_saopaulo_str)


def urlize(text):
    url_pattern = re.compile(r'(https?://[^\s]+)')
    return url_pattern.sub(r'<a href="\1" target="_blank">\1</a>', text)

app.jinja_env.filters['urlize'] = urlize

@app.route('/get_links', methods=['GET'])
def get_links():
    # if os.path.exists('links-created.json'):
    #     with open('links-created.json', 'r') as file:
    #         links = json.load(file)
    #         return jsonify(links)
    # else:
    #     return jsonify([])
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as file:
            links = json.load(file)
            return jsonify(links)
    else:
        return jsonify([])

@app.route('/save_link', methods=['POST'])
def save_link():
    new_link = request.json.get('link')
    
    links = []
    if os.path.exists('links-created.json'):
        with open('links-created.json', 'r') as file:
            links = json.load(file)
    
    links.append(new_link)
    
    with open('links-created.json', 'w') as file:
        json.dump(links, file)
    
    return jsonify({"status": "success"})


@app.route('/save_super_admin_settings', methods=['POST'])
def update_or_create_json():
    # Get the posted JSON data
    data = request.json

    save_data_to_json(data, json_file_path)

    return jsonify({'message': 'Data updated or created successfully'})

def save_data_to_json(data, local_json_file_path):
    # Load the existing JSON file if it exists
    if os.path.exists(local_json_file_path):
        with open(local_json_file_path, 'r') as file:
            existing_data = json.load(file)
    else:
        directory = os.path.dirname(local_json_file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        existing_data = {}  # Create an empty dictionary if the file doesn't exist

    # Merge the new data with the existing data
    existing_data.update(data)

    # Save the updated data back to the JSON file
    with open(local_json_file_path, 'w') as file:
        json.dump(existing_data, file, indent=2)
        

@app.route('/process_answer/<id>', methods=['POST'])
def process_answer(id):
    user_question = request.json['question']

    # Call your process_answer function here and get the bot's answer
    # Replace the example response with the actual bot's answer

    # df=pd.read_csv('processed/embeddings.csv', index_col=0)
    df = None
    processed_folder = 'processed/'+id+'/embeddings.csv'
    if(os.path.exists(processed_folder)):
        df=pd.read_csv(processed_folder, index_col=0)
        df['embeddings'] = df['embeddings'].apply(literal_eval).apply(np.array)
    
    bot_answer = answer_question(df, question=user_question,id=id)
    # bot_answer = "This is an example response from the bot."

    return jsonify({'answer': bot_answer})


@app.route('/receivedfiles/<id>', methods=['GET', 'POST'])
# @app.route('/receivedfiles', methods=['GET', 'POST'])
# def received_files(id):
def received_files(id):
    # print(id)
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
            
            user_folder = 'receivedfiles/'+id
            file.save(os.path.join(user_folder, file.filename))

        return redirect(url_for('received_files', id=id))

    files_info = get_received_files_info(id)
    return render_template('received_files.html', files_info=files_info, user_id=id)
    # return render_template('received_files.html', files_info=files_info)

@app.route('/chatbotprompts/<id>', methods=['GET', 'POST'])
def chatbot_prompts(id):
    # print(id)
    # prompt_file = PROMPTS_VALUES_FILE+'-'+id
    prompt_file = os.path.join(app.root_path, 'prompts/'+id+'/prompts-values.json')

    if request.method == 'POST':
        data = {
            'prompt-role-purpose': request.form.get('prompt-role-purpose'),
            'prompt-dont-know-text': request.form.get('prompt-dont-know-text')
        }
        
        if not os.path.exists(os.path.join(app.root_path , 'prompts')):
            os.makedirs(os.path.join(app.root_path , 'prompts'))
    
        if not os.path.exists(os.path.join(app.root_path ,'prompts/'+id)):
            os.makedirs(os.path.join(app.root_path ,'prompts/'+id))
    
        with open(prompt_file, 'w+') as file:
            json.dump(data, file)

    if(os.path.exists(prompt_file)):
        with open(prompt_file, 'r', encoding="utf-8") as file:
            prompts_values = json.load(file)
    else:
        prompts_values = write_prompt_default(id)

    return render_template('chatbot_prompts.html', prompts_values=prompts_values, user_id=id)

def write_prompt_default(id):
    prompt_file = os.path.join(app.root_path, 'prompts/'+id+'/prompts-values.json')
    prompts_values = {"prompt-role-purpose": "Responda \u00e0 pergunta com base no contexto abaixo e, se a pergunta n\u00e3o puder ser respondida com base no contexto, responda ", "prompt-dont-know-text": "Nao tenho conhecimento sobre essa informacao"}
    if not os.path.exists(os.path.join(app.root_path , 'prompts')):
        os.makedirs(os.path.join(app.root_path , 'prompts'))

    if not os.path.exists(os.path.join(app.root_path ,'prompts/'+id)):
        os.makedirs(os.path.join(app.root_path ,'prompts/'+id))

    with open(prompt_file, 'w+') as file:
        json.dump(prompts_values, file)
    return prompts_values

@app.route('/trainchatbot/<id>', methods=['GET', 'POST'])
def train_chatbot(id):
    # print(id)
    # Set Cache-Control and Pragma headers to prevent caching

    if request.method == 'POST':
        received_files = os.listdir('receivedfiles')
        if not received_files:
            return jsonify({'error': 'No files available to train'})
        
        start_chatbot_training(id)
        # return redirect(url_for('train_chatbot'))  # Redirect back to the same page after POST
        return jsonify({'success': 'files available were training'})
    
    recent_files_info = get_received_files_info(id)

    training_file = os.path.join(app.root_path, 'training/'+id+'/training-executions-data.json')
    training_data = []
    if os.path.exists(training_file):
        with open(training_file, 'r') as file:
            training_data = json.load(file)

    response = make_response(render_template('train_chatbot.html', training_data=training_data, files_info=recent_files_info, user_id=id))
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    
    # Set an Expires header with a past date to ensure the content is stale immediately
    response.headers['Expires'] = '0'

    return response

@app.route('/submit_rating', methods=['POST'])
def submit_rating():
    try:
        data = request.get_json()
        rating = int(data['rating'])
        
        # Load existing ratings or create an empty list
        try:
            with open('ratings.json', 'r') as file:
                ratings = json.load(file)
        except FileNotFoundError:
            ratings = []

        # Append the new rating to the ratings list
        ratings.append(rating)

        # Save the updated ratings list to the file
        with open('ratings.json', 'w') as file:
            json.dump(ratings, file)

        return jsonify({'message': 'Rating submitted successfully'})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/submit_delete', methods=['POST'])
def submit_delete():
    try:
        data = request.get_json()
        myid = data['myid']
        
        array_folders = ["receivedfiles/"+myid, "pdf-folder/"+myid, "training/"+myid, "processed"+myid, "messages/"+myid]

        for folder in array_folders:
            # delete files from received files
            folder_path = folder
            if(os.path.exists(folder_path)):
                file_list = os.listdir(folder_path)
                for filename in file_list:
                    # Construct the full file path
                    file_path = os.path.join(folder_path, filename)
                    # print(file_path)
                    
                    # Delete the txt file
                    os.remove(file_path)
        
        return jsonify({'message': 'Chat e arquivos aprendidos deletados ok!'})
    except Exception as e:
        return jsonify({'error': str(e)})



# Helper function to read received files information
def get_received_files_info(id):

    brasilia_tz = pytz.timezone('America/Sao_Paulo')

    user_folder = app.config['UPLOAD_FOLDER']+'/'+id
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)

    files_info = []
    for filename in os.listdir(user_folder):
        if filename.lower().endswith('.pdf') or filename.lower().endswith('.txt'):
            file_path = os.path.join(user_folder, filename)
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
            file_date = datetime.fromtimestamp(os.path.getmtime(file_path))
            file_date_brasilia = file_date.replace(tzinfo=pytz.utc).astimezone(brasilia_tz)
            files_info.append({'filename': filename, 'size': round(file_size, 2), 'datetime': file_date_brasilia})
    return files_info

def start_chatbot_training(id):
    
    train_folder = app.config['UPLOAD_FOLDER'] + '/'+ id
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)

    received_files_path = train_folder
    received_files = [f for f in os.listdir(received_files_path) if f.lower().endswith(('.pdf', '.txt'))]

    # Calculate the elapsed time
    start_time = datetime.now()
    # Your chatbot training function here...
    # Replace the following line with the actual chatbot training code
    # (This is a dummy line to simulate training time)
    import time
    # time.sleep(5)  # Simulate training time of 5 seconds
    
    move_files_to_pdf_folder(id)
    input_folder_path = 'pdf-folder/'+id  # Replace with the path to the folder containing PDF files
    output_folder_path = 'text/pdf/'+id  # Replace with the path to the output folder
    pdf_to_txt(input_folder_path, output_folder_path)
    process_pdf_txt(id)
    do_embeddings(id)
        
    end_time = datetime.now()
    elapsed_time = (end_time - start_time).seconds

    saopaulo_tz = pytz.timezone('America/Sao_Paulo')

    # Get the current datetime in the Sao Paulo timezone
    current_datetime_saopaulo = datetime.now(saopaulo_tz)

    # Format the datetime as a string
    formatted_datetime = current_datetime_saopaulo.strftime("%Y-%m-%d %H:%M:%S")

    # Add a new row to training data
    new_execution = {
        "datetime": formatted_datetime,
        "files": received_files,
        "elapsed_time": elapsed_time
    }

    if not os.path.exists(os.path.join(app.root_path , 'training')):
        os.makedirs(os.path.join(app.root_path , 'training'))
    
    if not os.path.exists(os.path.join(app.root_path ,'training/'+id)):
        os.makedirs(os.path.join(app.root_path ,'training/'+id))
    
    training_file = os.path.join(app.root_path, 'training/'+id+'/training-executions-data.json')
    if not os.path.exists(training_file):
        with open(training_file, 'w+') as file:
            json.dump([], file, indent=4)
            file.truncate()

    with open(training_file, 'r+') as file:
        data = json.load(file)
        data.append(new_execution)
        file.seek(0)
        json.dump(data, file, indent=4)
        file.truncate()





def pdf_to_txt(input_folder, output_folder):


    folder_path = output_folder

    # Get a list of all files in the folder
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    file_list = os.listdir(folder_path)

    # Loop through each file and check if it's a txt file
    for filename in file_list:
        if filename.lower().endswith('.txt'):
            
            # Construct the full file path
            file_path = os.path.join(folder_path, filename)
            # print(file_path)
            
            # Delete the txt file
            os.remove(file_path)
            
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


def move_files_to_pdf_folder(id):
    
    source_folder = "receivedfiles/"+id
    destination_folder = "pdf-folder/"+id

    # Check if the destination folder exists and create it if not
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # remove destination files before getting the new uploaded ones
    file_list = [file for file in os.listdir(destination_folder) if file.lower().endswith(('.pdf', '.txt'))]
    for filename in file_list:
        if filename.lower().endswith('.pdf'):
            
            # Construct the full file path
            file_path = os.path.join(destination_folder, filename)
            # print(file_path)
            
            # Delete the txt file
            os.remove(file_path)
    
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
    
def process_pdf_txt(id):

    # Create a list to store the text files
    texts=[]

    domain = 'pdf/'+id

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
    processed_folder = 'processed/'+id
    if not os.path.exists(processed_folder):
        os.makedirs(processed_folder)

    # Set the text column to be the raw text with the newlines removed
    df['text'] = df.fname + ". " + remove_newlines(df.text)
    df.to_csv(processed_folder+'/scraped.csv')
    df.head()

def do_embeddings(id):

    max_tokens = 8000

    # Load the cl100k_base tokenizer which is designed to work with the ada-002 model
    tokenizer = tiktoken.get_encoding("cl100k_base")

    processed_folder = 'processed/'+id
    df = pd.read_csv(processed_folder+'/scraped.csv', index_col=0)
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

    # df['embeddings'] = df.text.apply(lambda x: openai.Embedding.create(input=x, engine='text-embedding-ada-002')['data'][0]['embedding'])
    # Create empty list to store embeddings
    embeddings = []

    # Iterate through each text value in the dataframe  
    for text in df.text:
        # Call OpenAI Embedding API
        response = openai.Embedding.create(
            input=text, 
            engine='text-embedding-ada-002'
        )

        log_to_file('training: ' + str(response['usage']))

        # Extract embedding vector from response
        embedding = response['data'][0]['embedding'] 

        # Append embedding to list
        embeddings.append(embedding)

    # Add embeddings list as new column in dataframe
    df['embeddings'] = embeddings

    df.to_csv(processed_folder+ '/embeddings.csv')

    # df.head()
    

def create_context(
    question, df, max_len=8000, size="ada"
):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    # Get the embeddings for the question
    response = openai.Embedding.create(input=question, engine='text-embedding-ada-002')
    
    # log_to_file('training: ' + str(response['usage']))

        # Extract embedding vector from response
    q_embeddings = response['data'][0]['embedding'] 
    

    log_to_file('create_context: ' + str(response['usage']))
    # print(q_embeddings)


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
    # model="text-davinci-003",
    model="gpt-3.5-turbo-16k",
    question="Am I allowed to publish model outputs to Twitter, without a human review?",
    max_len=8000,
    size="ada",
    debug=False,
    max_tokens=15000,
    stop_sequence=None,
    id=id
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    if df is not None:
        context = create_context(
            question,
            df,
            max_len=max_len,
            size=size,
        )
    else:
        context = ""

    # If debug, print the raw model response
    if debug:
        print("Context:\n" + context)
        print("\n\n")

    try:
        
        # {"prompt-role-purpose": "Responda à pergunta com base no contexto abaixo e, se a pergunta não puder ser respondida com base no contexto, responda ", "prompt-dont-know-text": "pô brother, aí tu me fudeu...."}
        prompt_file = os.path.join(app.root_path, 'prompts/'+id+'/prompts-values.json')
        if not os.path.exists(prompt_file):
            write_prompt_default(id)

        with open(prompt_file, "r", encoding="utf-8") as file:
            data = json.load(file)

        prompt_dont_know_text = data.get("prompt-dont-know-text", "")
        prompt_role_purpose = data.get("prompt-role-purpose", "")
        # print(prompt_dont_know_text)
        # print(prompt_role_purpose)

        questions_file = os.path.join(app.root_path, 'messages/'+id+'/chat-messages.json')
        question_values = []
        if(os.path.exists(questions_file)):
            with open(questions_file, 'r', encoding="utf-8") as file:
                question_values = json.load(file)

        # questions_array.clear()
        # print(question_values[id]['messages'])
        if id in question_values:
            questions_array = question_values[id]['messages']
        else:
            questions_array = []

        questions_array.append(question)

        # prompt=f"{prompt_role_purpose} \"{prompt_dont_know_text}\"\n\nContexto: {context}\n\n---\n\nPergunta: {question}\nResposta:",
        prompt=f"{prompt_role_purpose} \"{prompt_dont_know_text}\"\n\nContexto: {context}\n\n---\n\n"
        # question = f"Pergunta: {question}\n"

        messages_parameter = [{"role": "system", "content": prompt}]
        # Loop through each question in the questions array and add it to the messages list
        i = 3
        for question in questions_array:
            if i % 2 == 1:
                messages_parameter.append({"role": "user", "content": question})
            else:
                messages_parameter.append({"role": "assistant", "content": question})
            i = i+1
        
        thismodel = get_config_by_id(id)['model']
        response = openai.ChatCompletion.create(
            # model="gpt-3.5-turbo",
            # model="gpt-3.5-turbo-16k",
            model=thismodel,
            temperature=0.0,
            messages=messages_parameter
        )

        # print(messages_parameter)

        # print(response['choices'][0]['message']['content'])
        # print(type(response['choices'][0]['message']['content']))
        log_to_file('completion: '+str(response['usage']))
        # print(response['usage'])
        # print(response['choices'])

        bot_response = response['choices'][0]['message']['content']
        
        questions_array.append(bot_response)
        # print(id)
        # print(prompt_role_purpose)
        # print(prompt_dont_know_text)
        # print(questions_array)

        save_data_to_json({id:{'prompt-role':prompt_role_purpose,
                               'prompt-dont-know-text':prompt_dont_know_text,
                               'messages':questions_array
                               }},"messages/"+id+"/chat-messages.json")


        return bot_response
    
        # # Create a completions using the questin and context
        # response = openai.Completion.create(
        #     prompt=f"{prompt_role_purpose} \"{prompt_dont_know_text}\"\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:",
        #     temperature=0,
        #     max_tokens=max_tokens,
        #     top_p=1,
        #     frequency_penalty=0,
        #     presence_penalty=0,
        #     stop=stop_sequence,
        #     model=model,
        # )
        # return response["choices"][0]["text"].strip()
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
    
        # Get the line number where the exception occurred
        line_no = exc_traceback.tb_lineno
    
        print(f"An exception of type {type(e).__name__} occurred at line {line_no}.")
        # print(e)
        return str(e)

def log_to_file(message):
    
    print(message)
    message = message.replace("\r\n", "")
    brasilia_tz = pytz.timezone('America/Sao_Paulo')
    current_datetime = datetime.now(brasilia_tz)
    
    with open(file_path, 'a') as file:
        formatted_datetime = current_datetime.strftime('%Y-%m-%d %H:%M:%S %Z')
        log_entry = f'{formatted_datetime};{message}\n'
        file.write(log_entry)

if __name__ == '__main__':
    app.run(debug=True)

