from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
from dotenv import load_dotenv
import re
import nltk
from nltk.corpus import stopwords
import json

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

load_dotenv()

app = Flask(__name__)
CORS(app)

HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

MODEL_NAME = "nlpaueb/legal-bert-small-uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


chat_histories = {}

# Define IPC specific categories with detailed subcategories
IPC_CATEGORIES = {
    "offenses_against_state": [
        "sedition", "waging war", "conspiracy", "anti-national", "sovereignty", "section 121",
        "section 124", "section 131", "treason", "state security"
    ],
    "offenses_against_public": [
        "public tranquility", "unlawful assembly", "rioting", "affray", "section 141",
        "section 146", "section 159", "public peace", "public order"
    ],
    "offenses_against_person": [
        "murder", "culpable homicide", "hurt", "grievous hurt", "assault", "criminal force",
        "kidnapping", "abduction", "rape", "sexual assault", "section 299", "section 300",
        "section 319", "section 320", "section 351", "section 359", "section 375"
    ],
    "property_offenses": [
        "theft", "extortion", "robbery", "dacoity", "criminal breach of trust", "cheating",
        "mischief", "trespass", "section 378", "section 383", "section 390", "section 391",
        "section 405", "section 415", "section 425", "section 441"
    ],
    "document_offenses": [
        "forgery", "counterfeiting", "false document", "section 463", "section 489"
    ],
    "criminal_conspiracy": [
        "conspiracy", "abetment", "section 120", "section 107", "section 108"
    ],
    "defamation": [
        "defamation", "slander", "libel", "section 499", "section 500"
    ],
    "attempt_offenses": [
        "attempt to murder", "attempt to commit", "section 511", "section 307"
    ],
    "ipc_procedures": [
        "fir", "chargesheet", "bail", "arrest", "investigation", "police report",
        "judicial custody", "police custody", "cognizable", "non-cognizable"
    ]
}

# Add IPC sections
IPC_SECTIONS = {}
for i in range(1, 512):  # IPC has sections from 1 to 511
    section_terms = [f"section {i}", f"ipc {i}", f"ipc section {i}", f"s.{i}", f"s. {i}"]
    IPC_SECTIONS[f"section_{i}"] = section_terms

# Combine with IPC_CATEGORIES
IPC_CATEGORIES.update(IPC_SECTIONS)

# Flatten the categories for easy checking
IPC_TERMS = set()
for category, terms in IPC_CATEGORIES.items():
    IPC_TERMS.update(terms)

# Additional general IPC terms
GENERAL_IPC_TERMS = {
    "ipc", "indian penal code", "penal code", "criminal code", "crpc", "criminal procedure code",
    "bailable", "non-bailable", "punishment", "imprisonment", "fine", "criminal law", "offence",
    "criminal", "accused", "defendant", "complainant", "prosecution", "defense"
}
IPC_TERMS.update(GENERAL_IPC_TERMS)

# List of obviously non-legal topics
NON_LEGAL_TOPICS = {
    # Food and cooking
    "recipe", "cook", "bake", "ingredient", "dish", "meal", "cuisine", "restaurant", "food",
    "breakfast", "lunch", "dinner", "dessert", "appetizer", "snack", "taste", "flavor",
    # Entertainment
    "movie", "film", "show", "series", "actor", "actress", "director", "plot", "episode",
    "music", "song", "album", "artist", "band", "concert", "lyric", "genre", "playlist",
    "game", "gaming", "player", "level", "character", "console", "score", "highscore",
    # Clearly unrelated
    "birthday", "party", "celebration", "holiday", "festival", "decoration", "learn", "become", "gift",
    "garden", "plant", "flower", "vegetable", "gardening", "landscaping", "seed", "soil", "mulch", "clothes", "brand"
}

def preprocess_text(text):
    """Clean and preprocess text for classification"""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters except spaces and alphanumerics
    text = re.sub(r'[^\w\s]', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text   
    
def check_query_type(text):
    """Determine if a query is IPC-related based on the context and intent"""
    text = preprocess_text(text)
    
    # Check for IPC specific terms first
    if any(term in text for term in GENERAL_IPC_TERMS):
        return "ipc"
    
    # Check for IPC section mentions
    section_pattern = r'(section|s\.)\s*\d{1,3}(\s+of\s+ipc|\s+ipc)?'
    if re.search(section_pattern, text, re.IGNORECASE):
        return "ipc"
    
    # Check for common legal question patterns
    legal_patterns = [
        r'(what|how).*(law|legal|right|sue|complaint|file|court)',
        r'(can|may|should) (i|we|one|they).*(sue|legal|law|rights|file|claim|case)',
        r'(is|are).*(legal|illegal|lawsuit|crime|breach|violation|beat|hit)',
        r'(my|their|his|her|our|someone).*(rights|obligations|liabilities|responsibilities)',
        r'(file|filing).*(complaint|lawsuit|suit|case|claim|charge)',
        r'(legal|law|attorney|lawyer|court|judge).*(advice|question|problem|issue|help)',
        r'(compensation|damages|settlement|penalty|fine).*(for|due to|because)',
    ]
    
    for pattern in legal_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return "legal"
    
    # Check for question intent patterns that indicate legal questions
    intent_indicators = [
        "am i liable", "is it against the law", "legal consequences", 
        "what are my rights", "seek damages", "press charges",
        "file a complaint", "take legal action", "legal remedy",
        "legal options", "criminal offense", "criminal offence", "punishment"
    ]
    
    for indicator in intent_indicators:
        if indicator in text.lower():
            return "legal"
    
    # If no legal pattern is found, check for common non-legal intents
    non_legal_patterns = [
        r'how (to|do i|can i|are) (make|cook|bake|prepare|you)',
        r'what is (the|a good|the best) (recipe|way to cook|brand)',
        r'(recommend|suggest).*(recipe|restaurant|movie|book|game)',
        r'(where|how|who).*(find|buy|purchase|get|is).*(product|item)'
    ]
    
    for pattern in non_legal_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return "non_legal"
    
    # For anything unclear, we'll use the model
    return "uncertain"
    
def is_legal_context(text, non_legal_term):
    """Check if a non-legal term appears in a legal context"""
    legal_context_patterns = [
        fr"(complaint|sue|lawsuit|legal action|dispute).*(against|regarding|about).*{non_legal_term}",
        fr"(liability|damages|compensation).*(related to|for).*{non_legal_term}",
        fr"{non_legal_term}.*(violate|breach|infringe|comply).*(law|regulation|code|rule)",
        fr"{non_legal_term}.*(business|shop|restaurant|company).*(license|permit|regulation)"
    ]
    
    for pattern in legal_context_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    
    return False

def extract_ipc_sections(text):
    """Extract specific IPC sections mentioned in the query"""
    text = text.lower()
    matches = []
    
    # Pattern for "section X" or "s. X" where X is a number
    section_pattern = r'(section|s\.)\s*(\d{1,3})'
    for match in re.finditer(section_pattern, text):
        section_num = match.group(2)
        matches.append(f"section_{section_num}")
    
    # Pattern for "IPC X" or "IPC section X"
    ipc_pattern = r'ipc\s*(section)?\s*(\d{1,3})'
    for match in re.finditer(ipc_pattern, text):
        section_num = match.group(2)
        matches.append(f"section_{section_num}")
    
    return matches

def classify_query(text):
    """Classify if the query is IPC-related using context-aware approach"""
    
    intent_result = check_query_type(text)
    
    if intent_result == "ipc":
        return 1, 0.98, "ipc_specific"
    elif intent_result == "legal":
        preprocessed = preprocess_text(text)
        for term in IPC_TERMS:
            if term in preprocessed:
                return 1, 0.95, "ipc_term_found"
        return 1, 0.85, "general_legal"
    elif intent_result == "non_legal":
        words = set(preprocess_text(text).split())
        for word in words:
            if word in NON_LEGAL_TOPICS and is_legal_context(text, word):
                return 1, 0.85, "context_legal"
        return 0, 0.95, "intent_non_legal"
    
    text = preprocess_text(text)
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    
    probabilities = torch.softmax(outputs.logits, dim=1)
    prediction = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0][prediction].item()
    
    return prediction, confidence, "model_prediction"

def get_ipc_category(text):
    """Determine which IPC category/categories a query belongs to"""
    text = preprocess_text(text)
    categories = []
    
    # First check for specific IPC sections
    section_categories = extract_ipc_sections(text)
    categories.extend(section_categories)
    
    # Then check for category keywords
    for category, terms in IPC_CATEGORIES.items():
        if category.startswith("section_"):  # Skip section categories we already checked
            continue
        for term in terms:
            if term in text:
                categories.append(category)
                break
    
    # Remove duplicates
    categories = list(set(categories))
    return categories

def format_structured_response(raw_response):
    """Format AI-generated responses into a structured, user-friendly format."""
    
    # Check if the response is already formatted properly
    if re.search(r'^\d+\.', raw_response, re.MULTILINE):
        return raw_response

    # Split the response into lines
    lines = raw_response.strip().split('\n')
    formatted_lines = []

    
    # Extract key points - looking for sentences that may contain legal information
    key_sections = []
    current_point = ""

    for line in lines:
        line = line.strip()
        if not line:
            continue  # Skip empty lines

        # Identify new key points based on numbering (1., 2., etc.) or legal keywords
        if re.match(r'^\d+\.', line) or any(keyword in line.lower() for keyword in ["section", "ipc", "punishment", "offense", "crime", "penalty"]):
            if current_point:
                key_sections.append(current_point.strip())  # Store previous section before starting a new one
            current_point = line
        else:
            current_point += " " + line  # Append to current section

    # Add the last extracted section
    if current_point:
        key_sections.append(current_point.strip())

    # Format the extracted key sections with numbering
    for i, section in enumerate(key_sections, 1):
        formatted_lines.append(f" {section}")
    
   

    return "\n".join(formatted_lines)


# IPC section information database - contains key sections and their info
IPC_SECTION_INFO = {
    "section_299": {
        "title": "Culpable homicide",
        "description": "Whoever causes death by doing an act with the intention of causing death, or with the intention of causing such bodily injury as is likely to cause death, or with the knowledge that they are likely by such act to cause death, commits the offence of culpable homicide.",
        "punishment": "Punishment varies depending on specifics and is detailed in Section 304."
    },
    "section_300": {
        "title": "Murder",
        "description": "Culpable homicide is murder if the act by which the death is caused is done with the intention of causing death, or if it is done with the intention of causing such bodily injury as the offender knows to be likely to cause the death of the person to whom the harm is caused.",
        "punishment": "Death or imprisonment for life, and shall also be liable to fine. Specified in Section 302."
    },
    "section_302": {
        "title": "Punishment for murder",
        "description": "Whoever commits murder shall be punished with death, or imprisonment for life, and shall also be liable to fine.",
        "punishment": "Death or imprisonment for life, and fine."
    },
    "section_304": {
        "title": "Punishment for culpable homicide not amounting to murder",
        "description": "Whoever commits culpable homicide not amounting to murder shall be punished with imprisonment for life, or imprisonment up to 10 years, and fine.",
        "punishment": "Imprisonment for life, or imprisonment up to 10 years, and fine."
    },
    "section_304A": {
        "title": "Causing death by negligence",
        "description": "Whoever causes the death of any person by doing any rash or negligent act not amounting to culpable homicide.",
        "punishment": "Imprisonment up to 2 years, or fine, or both."
    },
    "section_375": {
        "title": "Rape",
        "description": "Sexual assault defined as non-consensual penetration or specific sexual acts under certain circumstances including against will, without consent, with consent obtained through fear or intimidation, etc.",
        "punishment": "Rigorous imprisonment not less than 10 years, may extend to imprisonment for life, and fine. Specified in Section 376."
    },
    "section_376": {
        "title": "Punishment for rape",
        "description": "Whoever commits rape shall be punished with rigorous imprisonment for a term not less than 10 years, but which may extend to imprisonment for life, and shall also be liable to fine.",
        "punishment": "Rigorous imprisonment not less than 10 years, up to life imprisonment, and fine."
    },
    "section_378": {
        "title": "Theft",
        "description": "Whoever, intending to take dishonestly any movable property out of the possession of any person without that person's consent, moves that property is said to commit theft.",
        "punishment": "Imprisonment up to 3 years, or fine, or both. Specified in Section 379."
    },
    "section_379": {
        "title": "Punishment for theft",
        "description": "Whoever commits theft shall be punished with imprisonment up to 3 years, or fine, or both.",
        "punishment": "Imprisonment up to 3 years, or fine, or both."
    },
    "section_499": {
        "title": "Defamation",
        "description": "Whoever, by words either spoken or intended to be read, or by signs or by visible representations, makes or publishes any imputation concerning any person intending to harm, or knowing or having reason to believe that such imputation will harm, the reputation of such person.",
        "punishment": "Simple imprisonment up to 2 years, or fine, or both. Specified in Section 500."
    },
    "section_500": {
        "title": "Punishment for defamation", 
        "description": "Whoever defames another shall be punished with simple imprisonment for a term which may extend to two years, or with fine, or with both.",
        "punishment": "Simple imprisonment up to 2 years, or fine, or both."
    },
    "section_319": {
        "title": "Hurt",
        "description": "Whoever causes bodily pain, disease or infirmity to any person is said to cause hurt.",
        "punishment": "Imprisonment up to 1 year, or fine up to Rs. 1,000, or both. Specified in Section 323."
    },
    "section_320": {
        "title": "Grievous hurt",
        "description": "Includes emasculation, permanent privation of eyesight or hearing, destruction of any member or joint, permanent disfiguration of head or face, fracture or dislocation of bone or tooth, or any hurt which endangers life or causes severe bodily pain or inability to follow ordinary pursuits for 20 days.",
        "punishment": "Imprisonment up to 7 years, and fine. Specified in Section 325."
    },
    "section_415": {
        "title": "Cheating",
        "description": "Whoever, by deceiving any person, fraudulently or dishonestly induces the person so deceived to deliver any property to any person, or to consent that any person shall retain any property, or intentionally induces the person so deceived to do or omit to do anything which he would not do or omit if he were not so deceived.",
        "punishment": "Imprisonment up to 1 year, or fine, or both. Specified in Section 417."
    }
}

def get_ipc_section_info(section_id):
    """Get information about a specific IPC section"""
    # Convert section ID to standard format (section_XXX)
    if not section_id.startswith("section_"):
        # Extract numbers from the section_id
        numbers = re.findall(r'\d+', section_id)
        if numbers:
            section_id = f"section_{numbers[0]}"
        else:
            return None
    
    return IPC_SECTION_INFO.get(section_id)

def generate_ipc_response(query):
    """Generate an IPC-focused response using structured information"""
    # Identify IPC categories and sections in the query
    categories = get_ipc_category(query)
    sections = [cat for cat in categories if cat.startswith("section_")]
    
    # Check for specific IPC sections first
    section_info = []
    for section in sections:
        info = get_ipc_section_info(section)
        if info:
            section_info.append(info)
    
    if section_info:
        # Format the response with the section information
        response = "Here is information about the Indian Penal Code sections relevant to your query:\n\n"
        
        for i, info in enumerate(section_info, 1):
            response += f"{i}. Section {section[8:]} - {info['title']}\n"
            response += f"   • Description: {info['description']}\n"
            response += f"   • Punishment: {info['punishment']}\n\n"
        
        response += "Please note: This information is provided for educational purposes only and does not constitute legal advice. Consult with a qualified legal professional for advice specific to your situation."
        
        return response
    
    # If no specific sections are found or we don't have info for them,
    # use Hugging Face API for general response
    return generate_huggingface_response(query)

def generate_huggingface_response(query, model_name="mistralai/Mistral-7B-Instruct-v0.2"):
    """Generate a legal response using Hugging Face's API"""
    API_URL = f"https://api-inference.huggingface.co/models/{model_name}"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}

    # Get the legal categories for this query to refine the prompt
    categories = get_ipc_category(query)
    category_info = ""
    
    if categories:
        category_names = []
        for cat in categories:
            if cat.startswith("section_"):
                category_names.append(f"IPC Section {cat[8:]}")
            else:
                category_names.append(cat.replace('_', ' ').title())
        
        category_info = f"This appears to be a question about {', '.join(category_names)}. "
    
    # Create a better prompt specifically for IPC questions
    system_prompt = """You are a helpful legal assistant specializing in Indian Penal Code (IPC). You provide information about Indian criminal laws and legal concepts in simple terms.
You are not a lawyer, and your responses are not legal advice. The user should consult with a licensed attorney for specific legal advice. Donot provide informations about non legal matters.if the user asked a non legal question tell the user to ask a valid legal question.

IMPORTANT: Format your response in a structured format with:
 Bullet points (•) for supporting details under each main point where appropriate Clear, concise language explaining legal terms.
  
 

When discussing IPC sections, always include:
- The section number
- What the offense/provision is called
- Brief explanation of the elements of the offense/provision
- The punishment prescribed

For example:
"Regarding your question about assault under Indian Penal Code:

1. IPC Section 351 - Assault
   • Assault is defined as any gesture or preparation that causes any person to apprehend that criminal force is about to be applied
   • Unlike hurt (Section 319), assault doesn't require actual physical contact

2. IPC Section 352 - Punishment for Assault
   • Punishment includes imprisonment up to 3 months, or fine up to Rs. 500, or both
"
"""
    
    prompt = f"{system_prompt}\n\n{category_info}Question about Indian Penal Code: {query}\n\nAnswer:"

    data = {"inputs": prompt, "parameters": {"max_length": 1000, "temperature": 0.7}}

    try:
        response = requests.post(API_URL, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        if isinstance(result, list) and len(result) > 0:
            raw_response = result[0]['generated_text'].split("Answer:")[-1].strip()
        else:
            raw_response = result.get('generated_text', '').split("Answer:")[-1].strip()
        
        # Format the response in a structured way if it's not already
        formatted_response = format_structured_response(raw_response)
        return formatted_response
            
    except requests.exceptions.RequestException as e:
        print(f"Error calling Hugging Face API: {e}")
        return "I'm experiencing technical difficulties. Please try again later."
def generate_simple_legal_response(query):
    """Generate a simplified legal response without mentioning IPC sections unless explicitly asked."""
    response = generate_huggingface_response(f"Provide a simple legal explanation without mentioning IPC sections: {query}")
    return response

def generate_expanded_response(previous_query):
    """Generate a more detailed response based on the last user query."""
    response = generate_huggingface_response(f"Explain in more detail about: {previous_query}")
    return response
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_query = data.get("message", "").strip()
    session_id = data.get("session_id", "default")

    print(f"Received query: {user_query}")

    # Initialize chat history if not present
    if session_id not in chat_histories:
        chat_histories[session_id] = []

    chat_history = chat_histories[session_id]

    if not user_query:
        return jsonify({"response": "Please enter a valid legal question."})

    # Handle follow-up queries like "explain more", "clarify", etc.
    follow_up_keywords = {"explain more", "tell me more", "clarify", "continue", "elaborate"}
    if user_query.lower() in follow_up_keywords:
        last_query = None
        for i in range(len(chat_history)-1, -1, -1):
            if chat_history[i]["role"] == "user":
                last_query = chat_history[i]["content"]
                break

        if last_query:
            ai_response = generate_expanded_response(last_query)
        else:
            ai_response = "There is no previous question to elaborate on."
        
        return jsonify({"response": ai_response})

    # Classify query type
    is_legal, confidence, reason = classify_query(user_query)
    print(f"Classification: {is_legal}, Confidence: {confidence}, Reason: {reason}")

    if is_legal == 0:
        return jsonify({"response": "I'm designed to answer questions related to Indian Penal Code and legal matters only.Ask a valid legal question"})

    # Check if user explicitly asks for IPC sections
    contains_ipc = any(term in user_query.lower() for term in IPC_TERMS)

    if contains_ipc:
        ai_response = generate_ipc_response(user_query)  # Provide IPC-specific response
    else:
        ai_response = generate_simple_legal_response(user_query)  # Provide a simplified legal response

    # Store conversation history
    chat_history.append({"role": "user", "content": user_query})
    chat_history.append({"role": "assistant", "content": ai_response})

    # Trim chat history to avoid excessive memory use
    if len(chat_history) > 20:
        chat_history = chat_history[-20:]

    chat_histories[session_id] = chat_history

    return jsonify({"response": ai_response})


@app.route('/reset', methods=['POST'])
def reset_chat():
    data = request.json
    session_id = data.get("session_id", "default")
    
    if session_id in chat_histories:
        chat_histories[session_id] = []
    
    return jsonify({"status": "Chat history reset successfully"})

# Debug endpoint to test classification and IPC section identification
@app.route('/test_ipc', methods=['POST'])
def test_ipc():
    data = request.json
    query = data.get("message", "").strip()
    
    keyword_result = check_query_type(query)
    is_legal, confidence, reason = classify_query(query)
    categories = get_ipc_category(query)
    
    # Extract IPC sections
    sections = extract_ipc_sections(query)
    section_info = {}
    
    for section in sections:
        info = get_ipc_section_info(section)
        if info:
            section_info[section] = info
    
    result = {
        "query": query,
        "keyword_check": keyword_result,
        "model_classification": {
            "is_legal": bool(is_legal),
            "confidence": confidence,
            "reason": reason
        },
        "categories": categories,
        "detected_sections": sections,
        "section_info": section_info
    }
    
    return jsonify(result)

# Endpoint to fetch information about specific IPC sections
@app.route('/ipc_section/<section_number>', methods=['GET'])
def get_section(section_number):
    section_id = f"section_{section_number}"
    section_info = get_ipc_section_info(section_id)
    
    if section_info:
        return jsonify({
            "section": section_number,
            "info": section_info
        })
    else:
        return jsonify({
            "error": f"Information for IPC Section {section_number} not found"
        }), 404

# Expand IPC database endpoint - allows adding new sections or updating existing ones
@app.route('/admin/update_ipc_database', methods=['POST'])
def update_ipc_database():
    # In a production environment, this should be secured with authentication
    data = request.json
    
    if not data or "section_number" not in data or "info" not in data:
        return jsonify({"error": "Invalid data format"}), 400
    
    section_number = data["section_number"]
    section_info = data["info"]
    
    # Validate section info
    required_fields = ["title", "description", "punishment"]
    for field in required_fields:
        if field not in section_info:
            return jsonify({"error": f"Missing required field: {field}"}), 400
    
    # Update the database
    section_id = f"section_{section_number}"
    IPC_SECTION_INFO[section_id] = section_info
    
    return jsonify({
        "status": "success",
        "message": f"IPC Section {section_number} updated successfully"
    })

if __name__ == '__main__':
    # Add more IPC sections to the database
    # This could be moved to a separate JSON file for better management
    additional_sections = {
        "section_124A": {
            "title": "Sedition",
            "description": "Whoever by words, either spoken or written, or by signs, or by visible representation, or otherwise, brings or attempts to bring into hatred or contempt, or excites or attempts to excite disaffection towards the Government established by law in India.",
            "punishment": "Imprisonment for life and fine, or imprisonment up to 3 years and fine, or fine."
        },
        "section_153A": {
            "title": "Promoting enmity between different groups",
            "description": "Promoting enmity between different groups on grounds of religion, race, place of birth, residence, language, etc., and doing acts prejudicial to maintenance of harmony.",
            "punishment": "Imprisonment up to 3 years, or fine, or both."
        },
        "section_295A": {
            "title": "Deliberate and malicious acts intended to outrage religious feelings",
            "description": "Deliberate and malicious acts intended to outrage religious feelings of any class by insulting its religion or religious beliefs.",
            "punishment": "Imprisonment up to 3 years, or fine, or both."
        },
        "section_306": {
            "title": "Abetment of suicide",
            "description": "If any person commits suicide, whoever abets the commission of such suicide.",
            "punishment": "Imprisonment up to 10 years and fine."
        },
        "section_354": {
            "title": "Assault or criminal force to woman with intent to outrage her modesty",
            "description": "Whoever assaults or uses criminal force to any woman, intending to outrage or knowing it to be likely that he will thereby outrage her modesty.",
            "punishment": "Imprisonment not less than 1 year but which may extend to 5 years, and fine."
        },
        "section_420": {
            "title": "Cheating and dishonestly inducing delivery of property",
            "description": "Whoever cheats and thereby dishonestly induces the person deceived to deliver any property to any person, or to make, alter or destroy the whole or any part of a valuable security.",
            "punishment": "Imprisonment up to 7 years and fine."
        },
        "section_498A": {
            "title": "Husband or relative of husband subjecting a woman to cruelty",
            "description": "Whoever, being the husband or the relative of the husband of a woman, subjects such woman to cruelty.",
            "punishment": "Imprisonment up to 3 years and fine."
        },
        "section_509": {
            "title": "Word, gesture or act intended to insult the modesty of a woman",
            "description": "Whoever, intending to insult the modesty of any woman, utters any word, makes any sound or gesture, or exhibits any object, intending that such word or sound shall be heard, or that such gesture or object shall be seen, by such woman.",
            "punishment": "Imprisonment not less than 1 year but which may extend to 3 years, and fine."
        },
        "section_304B": {
            "title": "Dowry death",
            "description": "Where the death of a woman is caused by any burns or bodily injury or occurs otherwise than under normal circumstances within seven years of her marriage and it is shown that soon before her death she was subjected to cruelty or harassment by her husband or any relative of her husband for, or in connection with, any demand for dowry.",
            "punishment": "Imprisonment not less than 7 years but which may extend to imprisonment for life."
        },
        "section_377": {
            "title": "Unnatural offences",
            "description": "Whoever voluntarily has carnal intercourse against the order of nature with any man, woman or animal.",
            "punishment": "Imprisonment for life, or imprisonment up to 10 years, and fine."
        }
    }
    
    # Update the database with additional sections
    IPC_SECTION_INFO.update(additional_sections)
    
    # Run the Flask app
    app.run(debug=False, host='0.0.0.0', port=int(os.getenv("PORT", 5000)))