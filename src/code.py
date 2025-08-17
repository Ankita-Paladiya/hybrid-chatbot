# -*- coding: utf-8 -*-
"""
Chatbot System - Python Script

"""

# Required Libraries
import re
import openai
import spacy
from sentence_transformers import SentenceTransformer
import wikipediaapi
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
import warnings
from transformers import pipeline
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob
from googletrans import Translator
import streamlit as st
import os

# Suppress Warnings
warnings.filterwarnings("ignore")

# OpenAI API Key (replace with env variable for security)
# openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")
# Initialize NLP Tools
nlp = spacy.load("en_core_web_sm")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize Wikipedia API
wiki_wiki = wikipediaapi.Wikipedia(
    language="en",
    user_agent="EnhancedChatbot/2.0 (https://example.com; contact@example.com)"
)

# Memory for Multi-Turn Conversations
user_memory = []
user_preferences = {}
translator = Translator()

# Initialize Zero-Shot Classification Pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Rule-Based Responses
rule_based_responses = {
    "greet": "Hello! How can I assist you today?",
    "farewell": "Goodbye! Have a great day!",
    "name": "I’m your friendly assistant chatbot.",
    "business_hours": "Our business hours are 9 AM to 5 PM, Monday to Friday.",
    "pricing": "Our products range from $10 to $500, depending on your requirements.",
    "returns": "You can return items within 30 days of purchase.",
    "shipping": "We offer free shipping for orders above $50."
}

# Sentiment Analysis
def analyze_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return "positive"
    elif analysis.sentiment.polarity < 0:
        return "negative"
    else:
        return "neutral"

# Query Contextualization
def update_preferences(query):
    global user_preferences
    if "I like" in query:
        preference = query.split("I like")[-1].strip()
        user_preferences["likes"] = preference

def contextualize_response(response):
    global user_preferences
    if "likes" in user_preferences:
        response += f" By the way, I remember you like {user_preferences['likes']}."
    return response
# Multilingual Support
def multilingual_chatbot(query, target_language="en"):
    query_in_english = translator.translate(query, dest="en").text
    response, confidence = hybrid_chatbot(query_in_english)
    response_in_target = translator.translate(response, dest=target_language).text
    return response_in_target, confidence

# Real-Time Visualization of Metrics
def chatbot_dashboard():
    st.title("Chatbot Analytics")
    st.metric("Total Queries", len(user_memory))
    positive_queries = sum(1 for m in user_memory if m['sentiment'] == 'positive')
    negative_queries = sum(1 for m in user_memory if m['sentiment'] == 'negative')
    st.metric("Positive Sentiment Queries", positive_queries)
    st.metric("Negative Sentiment Queries", negative_queries)

# Adjusted detect_intent function
def detect_intent(query):
    """
    Detects intent using zero-shot classification from Hugging Face transformers.
    Returns intent and confidence score.
    """
    candidate_labels = list(rule_based_responses.keys())
    result = classifier(query, candidate_labels)
    intent = result['labels'][0]
    confidence = result['scores'][0]
    if confidence > 0.85:
        return intent, confidence
    else:
        return None, None

# Semantic Search

# Expanded FAQs and Their Embeddings
faq_data = {
    "business_hours": "Our business operates from 9 AM to 5 PM, Monday to Friday.",
    "pricing": "Our products range from $10 to $500, depending on your requirements.",
    "returns": "You can return items within 30 days of purchase.",
    "shipping": "We offer free shipping for orders above $50.",
    "support": "You can reach out to our support team at support@example.com.",
    "payment_methods": "We accept Visa, MasterCard, PayPal, and Apple Pay.",
    "order_changes": "You can modify your order within 24 hours by contacting our support team.",
    "account_reset": "You can reset your password by clicking 'Forgot Password' on the login page.",
    "international_shipping": "Yes, we offer international shipping to select countries.",
    "warranty": "All our products come with a one-year warranty."
    # Add more FAQs as needed
}

# Recompute the embeddings
faq_embeddings = [embedder.encode(faq) for faq in faq_data.keys()]

# Improved Semantic Search
def semantic_search(query):
    """
    Performs semantic similarity search on FAQs using cosine similarity.
    Returns response and similarity score.
    """
    query_embedding = embedder.encode(query).reshape(1, -1)
    faq_embeddings_array = np.array(faq_embeddings)
    similarities = cosine_similarity(query_embedding, faq_embeddings_array)[0]
    best_match_idx = np.argmax(similarities)
    best_similarity = similarities[best_match_idx]
    if best_similarity > 0.7:
        return list(faq_data.values())[best_match_idx], best_similarity
    return None, None

# LLM Integration

# Modified call_llm Function with updated system prompt
def call_llm(query, memory):
    """
    Sends a query to OpenAI GPT and returns its response.
    """
    try:
        context = "\n".join([f"User: {m['user']}\nAssistant: {m['bot']}" for m in memory[-5:]]) if memory else ""
        messages = [
            {"role": "system", "content": "You are a customer service assistant for AcmeCorp. Provide accurate and specific information about our products, services, and policies to help customers with their inquiries."},
            {"role": "user", "content": query}
        ]
        if context:
            messages.insert(1, {"role": "assistant", "content": context})
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=150
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"Error: {str(e)}"

# Wikipedia Fact-Checking
def fact_check(query, response):
    """
    Performs fact-checking using Wikipedia summaries.
    Returns response and similarity score.
    """
    page = wiki_wiki.page(query)
    if page.exists():
        wiki_summary = page.summary[:500]
        response_embedding = embedder.encode(response).reshape(1, -1)
        wiki_embedding = embedder.encode(wiki_summary).reshape(1, -1)
        similarity = cosine_similarity(response_embedding, wiki_embedding)[0][0]
        if similarity < 0.5:
            warning = f"Warning: The response may not be accurate. For reference, Wikipedia says: {wiki_summary}"
            return warning, similarity
        return response, similarity
    else:
        return response, None
# Hybrid Chatbot Logic

# Heuristic Confidence Evaluation
def evaluate_confidence(response):
    """
    Evaluates confidence of GPT responses based on token count and heuristics.
    Returns confidence score.
    """
    token_count = len(response.split())
    if token_count < 10:
        confidence = 0.5  # Low confidence
    elif token_count < 20:
        confidence = 0.7
    else:
        confidence = 0.9  # High confidence
    return confidence

# Enhanced hybrid_chatbot function with context
def hybrid_chatbot(query, target_language="en"):
    """
    Combines rule-based, semantic search, LLM responses, fact-checking, and sentiment analysis.
    Adds support for query contextualization and multilingual responses.
    Returns response and confidence score.
    """
    # Translate the query to English if necessary
    if target_language != "en":
        query = translator.translate(query, dest="en").text

    # Analyze sentiment of the query
    sentiment = analyze_sentiment(query)

    # Update user preferences (if applicable)
    update_preferences(query)

    # Rule-based logic with entity verification
    intent, intent_confidence = detect_intent(query)
    if intent in rule_based_responses and intent_confidence:
        doc = nlp(query)
        person_entities = [ent.text.lower() for ent in doc.ents if ent.label_ == 'PERSON']
        assistant_names = ['you', 'your name', 'assistant', 'chatbot']
        if not person_entities or any(name in query.lower() for name in assistant_names):
            response = rule_based_responses[intent]
            response = contextualize_response(response)
            return response, intent_confidence * 100

    # Semantic search
    faq_response, faq_similarity = semantic_search(query)
    if faq_response:
        response = contextualize_response(faq_response)
        return response, faq_similarity * 100

    # LLM fallback with context
    top_n = 3
    similarities = []
    for faq in faq_data.keys():
        faq_embedding = embedder.encode(faq).reshape(1, -1)
        similarity = cosine_similarity(embedder.encode(query).reshape(1, -1), faq_embedding)[0][0]
        similarities.append(similarity)
    top_indices = np.argsort(similarities)[-top_n:]
    context_faqs = [list(faq_data.values())[i] for i in reversed(top_indices)]

    # Include context in the system prompt
    system_prompt = (
        "You are a customer service assistant for AcmeCorp. Here are some company policies and information:\n"
        f"- {chr(10).join(context_faqs)}\n"
        "Provide accurate and specific information about our products, services, and policies to help customers with their inquiries."
    )

    # Get the LLM response
    llm_response = call_llm_with_context(query, user_memory, system_prompt)
    llm_response = contextualize_response(llm_response)

    # Evaluate confidence of LLM response
    llm_confidence = evaluate_confidence(llm_response) * 100

    # Perform fact-checking using Wikipedia
    fact_checked_response, fact_similarity = fact_check(query, llm_response)
    if fact_similarity is not None:
        llm_confidence = fact_similarity * 100

    # Translate the final response back to the target language if necessary
    if target_language != "en":
        fact_checked_response = translator.translate(fact_checked_response, dest=target_language).text

    # Save the interaction in user memory
    user_memory.append({
        "user": query,
        "bot": fact_checked_response,
        "sentiment": sentiment
    })

    return fact_checked_response, llm_confidence
# Define call_llm_with_context Function
def call_llm_with_context(query, memory, system_prompt):
    """
    Sends a query to OpenAI GPT with provided context and returns its response.
    """
    try:
        context = "\n".join([f"User: {m['user']}\nAssistant: {m['bot']}" for m in memory[-5:]]) if memory else ""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        if context:
            messages.insert(1, {"role": "assistant", "content": context})
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=150
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"Error: {str(e)}"

# LLM-Only Chatbot
def llm_chatbot(query):
    """
    Uses only the LLM to generate responses.
    Returns response and confidence score.
    """
    system_prompt = (
        "You are a customer service assistant for AcmeCorp. "
        "Provide accurate and specific information about our products, services, "
        "and policies to help customers with their inquiries."
    )
    llm_response = call_llm_with_context(query, user_memory, system_prompt)
    llm_confidence = evaluate_confidence(llm_response) * 100
    fact_checked_response, fact_similarity = fact_check(query, llm_response)
    if fact_similarity is not None:
        llm_confidence = fact_similarity * 100
    return fact_checked_response, llm_confidence

# Traditional Rule-Based Chatbot
def traditional_chatbot(query):
    """
    Uses only the rule-based chatbot logic to generate responses.
    Returns response and confidence score.
    """
    intent, intent_confidence = detect_intent(query)
    if intent in rule_based_responses and intent_confidence:
        response = rule_based_responses[intent]
        return response, intent_confidence * 100
    else:
        return "I’m sorry, I didn’t understand that.", 0

# Evaluation Functions
def compute_cosine_similarity_score(response1, response2):
    embedding1 = embedder.encode(response1).reshape(1, -1)
    embedding2 = embedder.encode(response2).reshape(1, -1)
    similarity = cosine_similarity(embedding1, embedding2)[0][0]
    return similarity

def compute_bleu_score(reference, candidate):
    reference_tokens = reference.split()
    candidate_tokens = candidate.split()
    score = sentence_bleu([reference_tokens], candidate_tokens)
    return score

def compute_rouge_score(reference, candidate):
    rouge = Rouge()
    scores = rouge.get_scores(candidate, reference)
    rouge_1_f = scores[0]['rouge-1']['f']
    rouge_2_f = scores[0]['rouge-2']['f']
    rouge_l_f = scores[0]['rouge-l']['f']
    return (rouge_1_f + rouge_2_f + rouge_l_f) / 3  # Average F1 score

def compute_fuzzy_score(reference, candidate):
    score = fuzz.token_set_ratio(reference, candidate) / 100.0
    return score
def check_response(actual_response, expected_response, method='cosine'):
    """
    Checks whether the actual response matches the expected response using different evaluation methods.
    Returns True if the response is acceptable, False otherwise.
    """
    if method == 'cosine':
        similarity = compute_cosine_similarity_score(actual_response, expected_response)
        return similarity > 0.7
    elif method == 'bleu':
        bleu = compute_bleu_score(expected_response, actual_response)
        return bleu > 0.5
    elif method == 'rouge':
        rouge = compute_rouge_score(expected_response, actual_response)
        return rouge > 0.5
    elif method == 'fuzzy':
        fuzzy = compute_fuzzy_score(expected_response, actual_response)
        return fuzzy > 0.8
    else:
        similarity = compute_cosine_similarity_score(actual_response, expected_response)
        return similarity > 0.7

def evaluate_chatbots(test_queries, method='cosine'):
    results = []
    for test in test_queries:
        query = test['query']
        expected_response = test['expected_response']

        # Run through traditional chatbot
        traditional_response, traditional_confidence = traditional_chatbot(query)
        traditional_correct = check_response(traditional_response, expected_response, method)

        # Run through hybrid chatbot
        hybrid_response, hybrid_confidence = hybrid_chatbot(query)
        hybrid_correct = check_response(hybrid_response, expected_response, method)

        # Run through LLM chatbot
        llm_response, llm_confidence = llm_chatbot(query)
        llm_correct = check_response(llm_response, expected_response, method)

        # Collect responses and correctness
        results.append({
            'query': query,
            'expected_response': expected_response,
            'traditional_response': traditional_response,
            'traditional_correct': traditional_correct,
            'hybrid_response': hybrid_response,
            'hybrid_correct': hybrid_correct,
            'llm_response': llm_response,
            'llm_correct': llm_correct
        })
    return results

def compute_accuracy(results):
    traditional_correct = sum(r['traditional_correct'] for r in results)
    hybrid_correct = sum(r['hybrid_correct'] for r in results)
    llm_correct = sum(r['llm_correct'] for r in results)
    total = len(results)
    traditional_accuracy = traditional_correct / total
    hybrid_accuracy = hybrid_correct / total
    llm_accuracy = llm_correct / total
    return {
        'traditional_accuracy': traditional_accuracy,
        'hybrid_accuracy': hybrid_accuracy,
        'llm_accuracy': llm_accuracy
    }

def create_results_dataframe(results):
    df = pd.DataFrame(results)
    # Map correctness to integers for visualization
    df['traditional_correct_int'] = df['traditional_correct'].astype(int)
    df['hybrid_correct_int'] = df['hybrid_correct'].astype(int)
    df['llm_correct_int'] = df['llm_correct'].astype(int)
    return df

def plot_heatmap(df, method):
    correctness_matrix = df[['traditional_correct_int', 'hybrid_correct_int', 'llm_correct_int']].T
    correctness_matrix.columns = df['query']
    plt.figure(figsize=(12, 4))
    sns.heatmap(correctness_matrix, annot=True, cmap='YlGnBu', cbar=False)
    plt.title(f"Chatbot Correctness Heatmap Using {method.upper()} (1 = Correct, 0 = Incorrect)")
    plt.xlabel("Queries")
    plt.ylabel("Chatbots")
    plt.yticks([0.5, 1.5, 2.5], ['Traditional', 'Hybrid', 'LLM'])
    plt.show()
# Main Execution with Evaluation, Heatmap, and Enhanced Features
if __name__ == "__main__":
    # Prepare test queries and expected responses
    test_queries = [
        {"query": "Hello", "expected_intent": "greet", "expected_response": "Hello! How can I assist you today?"},
        {"query": "What are your business hours?", "expected_intent": "business_hours", "expected_response": "Our business hours are 9 AM to 5 PM, Monday to Friday."},
        {"query": "How much does your product cost?", "expected_intent": "pricing", "expected_response": "Our products range from $10 to $500, depending on your requirements."},
        {"query": "I want to return an item", "expected_intent": "returns", "expected_response": "You can return items within 30 days of purchase."},
        {"query": "Do you offer free shipping?", "expected_intent": "shipping", "expected_response": "We offer free shipping for orders above $50."},
        {"query": "Who are you?", "expected_intent": "name", "expected_response": "I’m your friendly assistant chatbot."},
        {"query": "Goodbye", "expected_intent": "farewell", "expected_response": "Goodbye! Have a great day!"},
        {"query": "Tell me about the solar system", "expected_intent": None, "expected_response": "The Solar System consists of the Sun and the objects that orbit it, including planets, moons, asteroids, and comets."},
        {"query": "What payment methods do you accept?", "expected_intent": None, "expected_response": "We accept Visa, MasterCard, PayPal, and Apple Pay."},
        {"query": "Can I change my order after it’s been placed?", "expected_intent": None, "expected_response": "You can modify your order within 24 hours by contacting our support team."},
        {"query": "Blah blubb", "expected_intent": None, "expected_response": "I’m sorry, I didn’t understand that."}
    ]

    # Evaluate chatbots using different methods
    evaluation_methods = ['cosine', 'bleu', 'rouge', 'fuzzy']

    for method in evaluation_methods:
        print(f"\nEvaluating chatbots using {method.upper()} method.")
        results = evaluate_chatbots(test_queries, method=method)
        accuracies = compute_accuracy(results)
        print("Accuracies:")
        print(f"Traditional Chatbot Accuracy: {accuracies['traditional_accuracy']*100:.2f}%")
        print(f"Hybrid Chatbot Accuracy: {accuracies['hybrid_accuracy']*100:.2f}%")
        print(f"LLM Chatbot Accuracy: {accuracies['llm_accuracy']*100:.2f}%")
        df = create_results_dataframe(results)
        plot_heatmap(df, method)

        # Optionally, print detailed results
        print("\nDetailed Results:")
        for r in results:
            print(f"Query: {r['query']}")
            print(f"Expected Response: {r['expected_response']}")
            print(f"Traditional Response: {r['traditional_response']} (Correct: {r['traditional_correct']})")
            print(f"Hybrid Response: {r['hybrid_response']} (Correct: {r['hybrid_correct']})")
            print(f"LLM Response: {r['llm_response']} (Correct: {r['llm_correct']})")
            print("-" * 50)

    # Interactive Chat Loop with Enhanced Features
    print("\nChatbot Ready! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        # Detect user language and allow multilingual responses
        print("\nSelect a language (default: English):")
        print("[1] English\n[2] Spanish\n[3] French\n[4] German\n[5] Italian")
        lang_option = input("Enter the number for your preferred language: ").strip()

        language_map = {
            "1": "en",
            "2": "es",
            "3": "fr",
            "4": "de",
            "5": "it"
        }
        target_language = language_map.get(lang_option, "en")

        # Get chatbot responses
        traditional_response, traditional_confidence = traditional_chatbot(user_input)
        hybrid_response, hybrid_confidence = hybrid_chatbot(user_input, target_language=target_language)
        llm_response, llm_confidence = llm_chatbot(user_input)

        # Store conversation in memory for context-aware responses
        user_memory.append({
            "user": user_input,
            "bot": hybrid_response,
            "sentiment": analyze_sentiment(user_input)
        })

        # Display the responses with confidence/probability
        print("\nResponses:")
        print(f"Traditional Chatbot: {traditional_response} (Confidence: {traditional_confidence:.2f}%)")
        print(f"Hybrid Chatbot: {hybrid_response} (Confidence: {hybrid_confidence:.2f}%)")
        print(f"LLM Chatbot: {llm_response} (Confidence: {llm_confidence:.2f}%)")

        # Real-Time Metrics Visualization
        print("\nMetrics:")
        positive_queries = sum(1 for m in user_memory if m["sentiment"] == "positive")
        negative_queries = sum(1 for m in user_memory if m["sentiment"] == "negative")
        print(f"Total Queries: {len(user_memory)}")
        print(f"Positive Sentiment Queries: {positive_queries}")
        print(f"Negative Sentiment Queries: {negative_queries}")
        print("-" * 50)
