# -*- coding: utf-8 -*-

import os
import logging
import re
import requests
import torch
from dotenv import load_dotenv
from flask import Flask, request, jsonify, session
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)

app.secret_key = os.getenv("FLASK_SECRET_KEY", "almafa")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv(dotenv_path='envfile.env')

classifier = pipeline("text-classification", model='bhadresh-savani/distilbert-base-uncased-emotion', top_k=None)

model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def generate_dialogpt_response(user_input):
    if 'chat_history_ids' in session:
        chat_history_ids = torch.tensor(session['chat_history_ids'])
    else:
        chat_history_ids = None

    new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    if chat_history_ids is not None:
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
    else:
        bot_input_ids = new_user_input_ids

    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7
    )

    session['chat_history_ids'] = chat_history_ids.tolist()

    response_ids = chat_history_ids[:, bot_input_ids.shape[-1]:][0]
    bot_response = tokenizer.decode(response_ids, skip_special_tokens=True)

    return bot_response


@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    prediction = classifier(text)
    highest_emotion = max(prediction[0], key=lambda x: x['score'])

    return jsonify({
        "predictions": prediction,
        "dominant_emotion": highest_emotion
    })

@app.route('/test', methods=['GET'])
def test_model():
    test_text = "I love using transformers. The best part is wide range of support and its easy to use"
    prediction = classifier(test_text)

    return jsonify({
        "test_text": test_text,
        "predictions": prediction
    })

@app.route('/webhook', methods=['POST'])
def webhook():
    req = request.get_json()
    text = req["queryResult"]["queryText"]

    if not text:
        return jsonify({"fulfillmentText": "I couldn't understand your message. Please try again."})

    prediction = classifier(text)
    highest_emotion = max(prediction[0], key=lambda x: x['score'])

    dialogpt_response = generate_dialogpt_response(text)

    print("Received query:", text)
    print("DialoGPT response:", dialogpt_response)

    return jsonify({"fulfillmentText": dialogpt_response})

@app.route('/fb_webhook', methods=['GET', 'POST'])
def fb_webhook():
    if request.method == 'GET':
        VERIFY_TOKEN = os.getenv("FB_VERIFY_TOKEN")
        mode = request.args.get("hub.mode")
        token = request.args.get("hub.verify_token")
        challenge = request.args.get("hub.challenge")

        if mode and token:
            if mode == "subscribe" and token == VERIFY_TOKEN:
                print("Webhook verified successfully!")
                return challenge, 200
            else:
                return "Verification token mismatch", 403
        return "Invalid request", 400

    elif request.method == 'POST':
        data = request.get_json()
        messaging_events = data.get('entry', [])[0].get('messaging', [])

        for event in messaging_events:
            if event.get('message'):
                sender_id = event['sender']['id']
                message_text = event['message'].get('text')

                if message_text.lower().strip() in ["reset", "start over", "clear"]:
                    session.pop('chat_history_ids', None)
                    send_fb_message(sender_id, "Conversation reset. Let's start fresh!")
                    continue

                dialogpt_response = generate_dialogpt_response(message_text)
                send_fb_message(sender_id, dialogpt_response)

        return "EVENT_RECEIVED", 200

@app.route('/')
def home():
    return "Emotion Detection API is running!"

def send_fb_message(recipient_id, message_text):
    PAGE_ACCESS_TOKEN = os.getenv("FB_PAGE_ACCESS_TOKEN")
    if not PAGE_ACCESS_TOKEN:
        print("Missing Facebook token")
        return

    url = f"https://graph.facebook.com/v12.0/me/messages?access_token={PAGE_ACCESS_TOKEN}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "recipient": {"id": recipient_id},
        "message": {"text": message_text}
    }

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        print("Failed to send message:", response.text)

@app.route('/reset', methods=['GET'])
def reset_conversation():
    session.pop('chat_history_ids', None)
    return jsonify({"message": "Conversation history cleared."})

@app.route('/prompt_test', methods=['GET'])
def prompt_test():
    user_input = request.args.get("text", "")

    if not user_input:
        return jsonify({"error": "No text provided"}), 400

    response = generate_dialogpt_response(user_input)

    return jsonify({
        "user_input": user_input,
        "chatbot_response": response
    })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)