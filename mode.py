from flask import Flask, request, jsonify, render_template, session
import threading
import uuid
import sqlite3
import os
import gc
import requests
from bs4 import BeautifulSoup
import json
import schedule
import time
import spacy
from pyngrok import ngrok

gc.collect()

os.environ['KAGGLE_CONFIG_DIR'] = '/content'

app = Flask(__name__)
app.secret_key = "super_secret_key"

conversations = {}

# إعادة إنشاء قاعدة البيانات
db_file = 'palama.db'
if os.path.exists(db_file):
    os.remove(db_file)

# إعداد قاعدة البيانات
def init_db():
    conn = sqlite3.connect('DB_palama.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS conversations
                 (id INTEGER PRIMARY KEY, question TEXT, answer TEXT)''')
    conn.commit()
    conn.close()

init_db()

def save_to_db(question, answer):
    conn = sqlite3.connect('DB_palama.db')
    c = conn.cursor()
    c.execute("INSERT INTO conversations (question, answer) VALUES (?, ?)", (question, answer))
    conn.commit()
    conn.close()

def clean_answer(answer):
    return answer.strip()

def find_answer_in_json(question):
    file_path = 'answers.json'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        if item['question'].lower() == question.lower():
            return item['answer']
    return None

def generate_answer_with_ollama(question):
    url = "http://127.0.0.1:11434/api/generate"
    try:
        response = requests.post(url, json={"prompt": question, "model": "llama3.2"})
        response.raise_for_status()
        print(f"Response from Ollama: {response.text}")
        responses = response.text.split("\n")
        final_response = ""
        for resp in responses:
            if resp.strip():
                data = json.loads(resp)
                final_response += data.get("response", "")
        return clean_answer(final_response)
    except requests.exceptions.RequestException as e:
        print(f"Error generating answer with Ollama: {e}")
        return "An error occurred while generating the answer."

def generate_answer(question):
    answer = find_answer_in_json(question)
    if answer:
        return answer
    else:
        return generate_answer_with_ollama(question)

def load_questions_and_answers(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for item in data:
            question = item['question']
            answer = item['answer']
            save_to_db(question, answer)

def fetch_web_data(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        texts = soup.find_all('p')
        return [text.get_text() for text in texts]
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from {url}: {e}")
        return []

def generate_qa_from_text(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    questions = []
    answers = []
    # استخراج الجمل
    sentences = list(doc.sents)
    for sentence in sentences:
        if "?" in sentence.text:
            continue  # تخطي الأسئلة الموجودة
        question = f"What is mentioned in this sentence: {sentence.text.strip()}"
        questions.append(question)
        answers.append(sentence.text.strip())
    return questions, answers

def job():
    # إضافة الروابط التي تريد جمع البيانات منها
    urls = [
        "https://www.bing.com",
    ]
    all_questions = []
    all_answers = []
    for url in urls:
        web_data = fetch_web_data(url)
        for text in web_data:
            questions, answers = generate_qa_from_text(text)
            all_questions.extend(questions)
            all_answers.extend(answers)
    for question, answer in zip(all_questions, all_answers):
        save_to_db(question, answer)

# جدولة المهمة للتدريب الذاتي
schedule.every().day.at("02:00").do(job)
@app.route('/api/command', methods=['POST'])
def process_command():
    user_id = session.get('user_id', str(uuid.uuid4()))
    if 'user_id' not in session:
        session['user_id'] = user_id
        conversations[user_id] = []
    data = request.get_json()
    command = data.get('command', "").strip()
    if not command:
        return jsonify({"status": "error", "message": "No command provided."})
    print(f"Received command: {command}")
    answer = generate_answer(command.lower())
    return jsonify({"status": "success", "answer": answer})

@app.route('/')
def home():
    return send_from_directory('', 'index.html')

if __name__ == "__main__":
    #ngrok.set_auth_token('2durkD6CSWHQoTrHuV6JpZYREH0_7ZkGwTuhgWWdj6BNZHKUp')
    #public_url = ngrok.connect(5000)
    #print(f"Access your application at: {public_url}")
    threading.Thread(target=lambda: schedule.run_pending()).start()
    app.run(host="0.0.0.0", port=5000)

