import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import random

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Define patterns and responses
patterns_responses = [
    (r'hello|hi|hey|good\s(morning|afternoon|evening)', ['Hello!', 'Hi there!', 'Hey!']),
    (r'how are you', ['I\'m doing well, thank you!', 'I\'m here and ready to assist.', 'Feeling good, thanks for asking!']),
    (r'bye|goodbye', ['Goodbye!', 'See you later!', 'Have a great day!']),
    (r'thank you|thanks', ['You\'re welcome!', 'No problem!', 'Happy to help!']),
    (r'(\d+)\s?(?:\+|\-|\*|\/)\s?(\d+)', lambda x, y: str(eval(x + y))),  # Basic arithmetic
    # Add more patterns and responses as needed
]

def preprocess_input(user_input):
    # Tokenize the user input
    tokens = word_tokenize(user_input.lower())
    
    # Remove stopwords and lemmatize the tokens
    filtered_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    
    return filtered_tokens


def respond_to_user_input(user_input):
    # Preprocess user input
    tokens = preprocess_input(user_input)
    
    # Check each pattern against the preprocessed tokens
    for pattern, response in patterns_responses:
        match = re.match(pattern, user_input)
        if match:
            groups = match.groups()
            if callable(response):
                return response(*groups)  # Pass matched groups to the response function
            else:
                return random.choice(response)
    
    # If no match found, generate a generic response
    return generate_response(tokens)

def generate_response(tokens):
    # Example: Generate a response based on the input tokens
    # Implement more sophisticated logic here based on token analysis
    return "Hmm, that's interesting. Tell me more!"

# Main loop to interact with the chatbot
print("Welcome! Ask me anything or say goodbye to exit.")

while True:
    user_input = input("You: ")
    if user_input.lower() == 'bye':
        print("Bot:", respond_to_user_input(user_input))
        break
    else:
        print("Bot:", respond_to_user_input(user_input))