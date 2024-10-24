from flask import Flask, request, jsonify, send_from_directory, session, render_template
import threading
from pyngrok import ngrok
import uuid
from googletrans import Translator
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.text import Tokenizer
import kagglehub
import os
import keras_nlp
from google.colab import files
uploaded = files.upload()
import gc
gc.collect()

# إعداد بيئة العمل
os.environ['KAGGLE_CONFIG_DIR'] = '/content'
# أدخل توكين المصادقة هنا

ngrok.set_auth_token('2durkD6CSWHQoTrHuV6JpZYREH0_7ZkGwTuhgWWdj6BNZHKUp')  # استبدل YOUR_AUTHTOKEN_HERE بالتوكين الخاص بك

# إعداد Flask مع secret key للجلسات
app = Flask(__name__)
app.secret_key = "super_secret_key"

# ذاكرة المحادثات لكل مستخدم
conversations = {}
translator = Translator()

# تنزيل النموذج المدرب Gemma باستخدام kagglehub
path = kagglehub.model_download("keras/gemma/keras/gemma_1.1_instruct_2b_en")
print("Path to model files:", path)

# تحميل نموذج Gemma من keras_nlp
gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("gemma_1.1_instruct_2b_en")

# تدريب tokenizer باستخدام بعض البيانات
tokenizer = Tokenizer()
sentences = ["hello", "how are you", "what's your name"]  # يجب استبدالها ببياناتك الحقيقية
tokenizer.fit_on_texts(sentences)

# الردود الجاهزة
predefined_responses = {
    "what's your name": "I'm Palama",
    "are you a human": "No, I'm a robot. My name is Palama.",
    "hello": "Hello! I'm Palama, how can I assist you?",
    "how are you": "I'm just a program, but I'm here to help you!",
    "what can you do": "I can provide information, tell jokes, fetch weather updates, and much more!",
    "tell me a joke": "Why did the scarecrow win an award? Because he was outstanding in his field!",
    "what is your favorite color": "I don't have feelings, but I think blue is nice!",
    "who created you": "I was created by a team Palama.inc.",
    "help me": "Sure! What do you need help with?",
    "tell me a fact": "Did you know that honey never spoils? Archaeologists have found pots of honey in ancient Egyptian tombs that are over 3000 years old and still edible!",
}

# دالة لتوليد الإجابات باستخدام نموذج Gemma
def generate_answer_with_model(input_data):
    if not input_data:  # تحقق مما إذا كانت البيانات فارغة
        return "I'm sorry, I don't understand that."

    # توليد النصوص باستخدام نموذج Gemma من keras_nlp
    generated_text = gemma_lm.generate(input_data, max_length=30)

    # تأكد من أن النص الناتج ليس فارغًا
    if generated_text and len(generated_text) > 0:
        return generated_text[0]
    
    return "I'm sorry, I don't understand that."  # إذا كان النص الناتج فارغًا

# دالة لتوليد الإجابات بناءً على السؤال
def generate_answer(question):
    question_lower = question.lower()

    # تحقق من الردود الجاهزة
    response = predefined_responses.get(question_lower)

    if response is None:
        # إذا لم يكن هناك رد جاهز، استخدم النموذج
        input_data = question  # تأكد من معالجة البيانات بشكل صحيح
        response = generate_answer_with_model(input_data)

    return response if response else "I'm sorry, I don't understand that."

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

    print(f"Received command: {command}")  # طباعة الأمر المستلم

    # تحويل الأمر إلى أحرف صغيرة لمزيد من الأمان
    command_lower = command.lower()

    # تحليل الأوامر بدون الحاجة لكلمة "question"
    answer = generate_answer(command_lower)

    return jsonify({"status": "success", "answer": answer})

@app.route('/')
def home():
    return render_template('index.html')  # تأكد من أن لديك ملف index.html في مجلد templates

if __name__ == "__main__":
    # أدخل توكين المصادقة
    ngrok.set_auth_token('2durkD6CSWHQoTrHuV6JpZYREH0_7ZkGwTuhgWWdj6BNZHKUp')  # استبدل بالتوكين الخاص بك

    # بدء تشغيل النفق
    public_url = ngrok.connect(5000)
    print(f"Access your application at: {public_url}")

    # تشغيل Flask
    app.run(host="0.0.0.0", port=5000)

