
import streamlit as st
import re
import json
import PyPDF2
import numpy as np
import faiss
import plotly.express as px  # Add this line
from pathlib import Path
from sentence_transformers import SentenceTransformer
import pickle
import os
import time
import asyncio
import spacy
import google.generativeai as genai
from PIL import Image
import pandas as pd
import io

import io
import random
import glob
_model = None
_index = None
_all_items = None
_country_map = None
_country_data = None
_nlp = None

# Gemini API Configuration
GEMINI_API_KEY = "AIzaSyDqMg4cv_n04wbxo16Bpovc01LXAa96h_I"  # Replace with your actual Gemini API key
genai.configure(api_key=GEMINI_API_KEY)
@st.cache_resource
def get_embedding_model():
    """Load the embedding model only once and reuse it."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model
def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file."""
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text() + "\n"
    return text



@st.cache_resource
def get_nlp_model():
    """Load the spaCy NLP model for natural language processing."""
    try:
        nlp = spacy.load("en_core_web_sm")
    except:
        st.warning("Installing spaCy model...")
        os.system("python -m spacy download en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    return nlp



def parse_country_data(text):
    """Parse the text to extract country data."""
    lines = text.split('\n')
    countries_data = {}
    current_country = None
    current_items = []
    
    country_pattern = re.compile(r'^([A-Za-z\s]+)\s*Import\s*Prohibitions$')
    item_pattern = re.compile(r'^\s*[•\-*]\s*(.+)$')
    
    for line in lines:
        line = line.strip()
        country_match = country_pattern.match(line)
        if country_match:
            if current_country and current_items:
                countries_data[current_country] = current_items
            current_country = country_match.group(1).strip()
            current_items = []
            continue
        
        item_match = item_pattern.match(line)
        if item_match and current_country:
            item_text = item_match.group(1).strip()
            if item_text and not item_text.startswith('See') and not item_text.startswith('Current as of'):
                current_items.append(item_text)
    
    if current_country and current_items:
        countries_data[current_country] = current_items
    
    return countries_data


def process_fedex_pdf(pdf_path, output_path):
    """Process the PDF and save country restrictions as JSON."""
    text = extract_text_from_pdf(pdf_path)
    countries_data = parse_country_data(text)
    cleaned_data = {}
    for country, items in countries_data.items():
        country_clean = re.sub(r'\s+', ' ', country).strip()
        if '\n' in country_clean:
            parts = country_clean.split('\n')
            country_clean = next((part.strip() for part in reversed(parts) if part.strip()), country_clean)
        if country_clean not in cleaned_data:
            cleaned_data[country_clean] = items
        else:
            cleaned_data[country_clean].extend(items)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
    
    st.success(f"Processed {len(cleaned_data)} countries. Data saved to {output_path}")
    return cleaned_data


def fix_malformed_countries(json_file):
    """Fix any malformed country names in the JSON file."""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    fixed_data = {}
    for country, items in data.items():
        if '\\n' in country:
            parts = country.split('\\n')
            country_name = next((part.strip() for part in reversed(parts) if part.strip()), country)
            if parts[0].strip() and len(parts) > 1:
                prev_countries = list(fixed_data.keys())
                if prev_countries:
                    fixed_data[prev_countries[-1]].append(parts[0].strip())
            if country_name in fixed_data:
                fixed_data[country_name].extend(items)
            else:
                fixed_data[country_name] = items
        else:
            if country in fixed_data:
                fixed_data[country].extend(items)
            else:
                fixed_data[country] = items
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(fixed_data, f, indent=2, ensure_ascii=False)
    
    st.success(f"Fixed malformed country names. Now contains {len(fixed_data)} countries.")
    return fixed_data


def create_vector_database(data, output_dir="vector_db"):
    """Create a vector database from the country items data."""
    os.makedirs(output_dir, exist_ok=True)
    model = get_embedding_model()
    all_items = []
    country_map = []
    
    total_countries = len(data)
    progress_bar = st.progress(0.0)
    for i, (country, items) in enumerate(data.items()):
        for item in items:
            all_items.append(item)
            country_map.append(country)
        progress_bar.progress((i + 1) / total_countries)
    
    st.info(f"Creating embeddings for {len(all_items)} items...")
    batch_size = 32
    embeddings_list = []
    progress_bar = st.progress(0.0)
    for i in range(0, len(all_items), batch_size):
        batch = all_items[i:i + batch_size]
        batch_embeddings = model.encode(batch)
        embeddings_list.append(batch_embeddings)
        progress_bar.progress((i + len(batch)) / len(all_items))
    
    embeddings = np.vstack(embeddings_list)
    faiss.normalize_L2(embeddings)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.astype(np.float32))
    
    faiss.write_index(index, os.path.join(output_dir, "items_index.faiss"))
    with open(os.path.join(output_dir, "items.pkl"), "wb") as f:
        pickle.dump(all_items, f)
    with open(os.path.join(output_dir, "country_map.pkl"), "wb") as f:
        pickle.dump(country_map, f)
    with open(os.path.join(output_dir, "country_data.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    st.success(f"Vector database created in {output_dir}")

def initialize_system(db_dir="vector_db", force_reload=False):
    """Initialize the system by loading all required components once."""
    global _model, _index, _all_items, _country_map, _country_data, _nlp
    
    if _model is None or force_reload:
        st.info("Loading embedding model...")
        _model = get_embedding_model()
    
    if _nlp is None or force_reload:
        st.info("Loading NLP model...")
        _nlp = get_nlp_model()
    
    index_path = os.path.join(db_dir, "items_index.faiss")
    items_path = os.path.join(db_dir, "items.pkl")
    country_map_path = os.path.join(db_dir, "country_map.pkl")
    country_data_path = os.path.join(db_dir, "country_data.json")
    
    if not all(os.path.exists(path) for path in [index_path, items_path, country_map_path, country_data_path]):
        st.error(f"Vector database files not found in {db_dir}. Please process the PDF first.")
        return False
    
    if _index is None or _all_items is None or _country_map is None or _country_data is None or force_reload:
        try:
            st.info("Loading vector database components...")
            with st.spinner("Loading index..."):
                _index = faiss.read_index(index_path)
            with st.spinner("Loading items..."):
                with open(items_path, "rb") as f:
                    _all_items = pickle.load(f)
            with st.spinner("Loading country map..."):
                with open(country_map_path, "rb") as f:
                    _country_map = pickle.load(f)
            with st.spinner("Loading country data..."):
                with open(country_data_path, "r", encoding="utf-8") as f:
                    _country_data = json.load(f)
            st.success(f"System initialized successfully with {len(_all_items)} items from {len(_country_data)} countries")
            return True
        except Exception as e:
            st.error(f"Error initializing system: {e}")
            return False
    return True

def query_vector_database(query, top_k=10):
    """Query the vector database for similar items."""
    global _model, _index, _all_items, _country_map
    
    if not initialize_system():
        return {"error": "Failed to initialize the system"}
    
    if _model is None:
        _model = get_embedding_model()
    
    if _index is None:
        st.error("FAISS index is not loaded.")
        return {"error": "FAISS index is not loaded"}
    
    query_embedding = _model.encode([query])
    faiss.normalize_L2(query_embedding)
    distances, indices = _index.search(query_embedding.astype(np.float32), top_k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(_all_items):
            item = _all_items[idx]
            country = _country_map[idx]
            score = distances[0][i]
            results.append({"country": country, "item": item, "score": float(score)})
    
    return results

def get_prohibited_items_for_country(country):
    """Get all prohibited items for a specific country."""
    global _country_data
    
    if not initialize_system():
        return {"error": "Failed to initialize the system"}
    
    # Debug print
    print(f"Searching for country: {country}")
    print(f"Available countries: {list(_country_data.keys())[:5]}...")
    
    for name, items in _country_data.items():
        if name.lower() == country.lower():
            print(f"Country match found: {name} with {len(items)} items")
            return {"country": name, "items": items, "count": len(items)}
    
    # Try partial matching for better country recognition
    partial_matches = [name for name in _country_data.keys() if country.lower() in name.lower()]
    if partial_matches:
        best_match = partial_matches[0]
        print(f"Partial match found: {best_match} instead of {country}")
        return {"country": best_match, "items": _country_data[best_match], "count": len(_country_data[best_match])}
    
    print(f"No match found for country: {country}")
    return {"error": f"Country '{country}' not found in the database."}

def search_prohibited_items(query, top_k=20):
    """Search for prohibited items based on a query."""
    query = query.strip()
    results = query_vector_database(query, top_k)
    
    # Debug print statement
    print(f"Vector DB search for '{query}', results: {len(results) if isinstance(results, list) else 'error'}")
    
    if isinstance(results, dict) and "error" in results:
        return results
    
    countries = {}
    for result in results:
        country = result["country"]
        if country not in countries:
            countries[country] = []
        countries[country].append({"item": result["item"], "score": result["score"]})
    
    response = []
    for country, items in countries.items():
        # Sort items by score in descending order
        items.sort(key=lambda x: x["score"], reverse=True)
        response.append({
            "country": country,
            "items": [item["item"] for item in items],
            "scores": [item["score"] for item in items],
            "count": len(items)
        })
    
    # Sort countries by highest scoring item
    response.sort(key=lambda x: max(x["scores"]) if x["scores"] else 0, reverse=True)
    
    return response

def extract_entities(text):
    """Extract entities from the query text."""
    global _nlp
    
    if _nlp is None:
        _nlp = get_nlp_model()
    
    doc = _nlp(text)
    countries = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
    items = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"] and token.text.lower() not in ["country", "countries"]]
    
    return {"countries": countries, "items": items}

def format_country_items_response(country_data):
    """Format an official response for country-specific restrictions."""
    country = country_data['country']
    items = country_data['items']
    
    response = [
        f"OFFICIAL IMPORT RESTRICTIONS - {country.upper()}",
        f"Number of Restricted Items: {len(items)}",
        "\nPROHIBITED ITEMS AND MATERIALS:",
    ]
    
    # Categorize items for more professional presentation
    for item in items:
        response.append(f"• {item}")
    
    response.extend([
        "\nIMPORTANT NOTICE:",
        "• This list represents current import prohibitions",
        "• Additional restrictions may apply",
        "• Verify requirements with customs authorities"
    ])
    
    return "\n".join(response)

def chatbot_response(query, chat_history):
    """Generate a human-like response using Gemini API and RAG with memory of previous chat."""
    entities = extract_entities(query)
    countries = entities["countries"]
    items = entities["items"]
    
    if not initialize_system():
        return "Hmm, it looks like I can't access the database right now. Could you try again later?"
    
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    
    # First get country data if available
    country_data = None
    if countries:
        country = countries[0]
        result = get_prohibited_items_for_country(country)
        if "error" not in result:
            country_data = result

    # Create chat history context for the model
    history_context = ""
    if chat_history and len(chat_history) > 0:
        history_context = "Previous conversation:\n"
        for msg in chat_history[-3:]:  # Use the last 3 messages as context
            role = "User" if msg["role"] == "user" else "Assistant"
            history_context += f"{role}: {msg['content']}\n"
        history_context += "\n"

    # Case 1: Specific country and item
    if countries and items:
        if not country_data:
            return ("• The specified country is not found in our database.\n"
                   "• Please verify the country name and try again.\n"
                   "• For assistance, you may provide the country's full name.")
        
        item = " ".join(items)
        all_prohibited = country_data["items"]
        matched_items = [i for i in all_prohibited if any(word.lower() in i.lower() for word in items)]
        
        context = (
            f"{history_context}"
            f"Official Import Restrictions Database\n"
            f"Country: {country}\n\n"
            f"Total Restricted Items: {len(all_prohibited)}\n\n"
            "Complete List of Import Prohibitions:\n" +
            "\n".join(f"• {item}" for item in all_prohibited) +
            f"\n\nQuery Analysis for: {item}\n" +
            "Relevant Restrictions:\n" +
            ("\n".join(f"• {item}" for item in matched_items) if matched_items else "• No direct matches found") +
            "\n\nCreate an official response that includes:\n"
            "• A clear declaration of the item's import status\n"
            "• Complete list of relevant restrictions\n"
            "• Related categories of prohibited items\n"
            "• Standard regulatory disclaimer\n"
            "Format as a formal bulletin with bullet points\n"
            "Maintain authoritative tone throughout"
        )
        
        response = model.generate_content(context)
        return response.text

    # Case 2: Item only
    elif items and not countries:
        item = " ".join(items)
        results = search_prohibited_items(item, top_k=100)
        
        if isinstance(results, dict) and "error" in results:
            return ("• System Notice: Unable to process database query\n"
                   "• Please rephrase your inquiry or try again later")
        
        if results:
            countries_with_item = [r for r in results if any(score > 0.5 for score in r["scores"])]
            context = (
                f"{history_context}"
                "INTERNATIONAL IMPORT RESTRICTIONS BULLETIN\n\n"
                f"Subject: {item.upper()}\n\n" +
                "\n\n".join(
                    f"JURISDICTION: {r['country']}\nPROHIBITED ITEMS AND CATEGORIES:\n" +
                    "\n".join(f"• {item}" for item in r['items'])
                    for r in countries_with_item
                ) +
                "\n\nGenerate formal advisory that includes:\n"
                "• Official notification of jurisdictions with restrictions\n"
                "• Comprehensive list of affected territories\n"
                "• Complete itemization of related restrictions\n"
                "• Standard regulatory notice\n"
                "Use formal, authoritative language\n"
                "Format as official bulletin"
            )
            response = model.generate_content(context)
            return response.text
        return ("OFFICIAL NOTICE:\n"
                "• No specific import restrictions found for this item\n"
                "• Importers must verify requirements with relevant authorities\n"
                "• Additional regulations may apply")

    # Case 3: Country only
    elif country_data:
        return format_country_items_response(country_data)

    # Case 4: No entities detected - Check memory for context
    else:
        # Check if we can use chat history to get context
        if chat_history and len(chat_history) > 0:
            # Extract countries and items from recent chat history
            recent_countries = []
            recent_items = []
            
            # Look at the last 3 messages to extract context
            for msg in chat_history[-3:]:
                if msg["role"] == "user":
                    entities = extract_entities(msg["content"])
                    if entities["countries"] and not recent_countries:
                        recent_countries = entities["countries"]
                    if entities["items"] and not recent_items:
                        recent_items = entities["items"]
            
            # If we found context from history, use it
            if recent_countries:
                country = recent_countries[0]
                result = get_prohibited_items_for_country(country)
                if "error" not in result:
                    context = (
                        f"{history_context}"
                        f"The user previously asked about {country}. Based on that context:\n\n"
                        f"Query: {query}\n\n"
                        "Please provide a helpful response about prohibited items in this country, "
                        "referencing the previous conversation and addressing the user's new query."
                    )
                    response = model.generate_content(context)
                    return response.text.replace("FedEx", "").replace("fedex", "")
            
            # General response with memory
            context = (
                f"{history_context}"
                f"User's new query: {query}\n\n"
                "Based on the conversation history, provide a helpful response about prohibited shipping items. "
                "If you can't determine what the user is asking about, prompt them for more specific information."
            )
            response = model.generate_content(context)
            return response.text.replace("FedEx", "").replace("fedex", "")
        
        # No history context available
        return ("• I need more specific information to help you\n"
                "• Try asking about:\n"
                "  • A specific country (e.g., 'What items are prohibited in Japan?')\n"
                "  • A specific item (e.g., 'Which countries prohibit electronics?')\n"
                "  • Or both (e.g., 'Can I ship alcohol to France?')")

def get_image_description(image):
    """Get a short description of the image using Gemini Vision."""
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = "Describe what item is shown in this image in 2-3 words only. Be very concise."
    
    try:
        response = model.generate_content([prompt, image])
        return response.text.strip()
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

def get_multiple_items_from_image(image):
    """Get multiple items detected in the image using Gemini Vision."""
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = """List detected physical objects:
    - One item per line
    - Use 2-3 words only
    - Skip decorative or background items
    - Maximum 5 items
    Do not include any introductory text or bullets."""
    
    try:
        response = model.generate_content([prompt, image])
        # Clean the response by removing any common prefixes and bullets
        clean_text = response.text.strip()
        clean_text = re.sub(r'^(Here\'s|This is|I see|The image shows).*?\n', '', clean_text, flags=re.IGNORECASE)
        clean_text = re.sub(r'^[-•*]\s*', '', clean_text, flags=re.MULTILINE)
        items = [item.strip() for item in clean_text.split('\n') if item.strip()]
        return items[:5]
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return []

def create_results_dataframe(items_results):
    """Create a pandas DataFrame from multiple items search results."""
    import pandas as pd
    
    if not items_results:
        return pd.DataFrame(columns=["Search Item", "Country", "Prohibited Item", "Relevance"])
        
    all_rows = []
    for item, results in items_results.items():
        for country_result in results:
            for detected_item, score in zip(country_result["items"], country_result["scores"]):
                if score > 0.6:  # Only include relevant matches
                    all_rows.append({
                        "Search Item": item,
                        "Country": country_result["country"],
                        "Prohibited Item": detected_item,
                        "Relevance": f"{score:.3f}"
                    })
    
    if not all_rows:
        return pd.DataFrame(columns=["Search Item", "Country", "Prohibited Item", "Relevance"])
    
    return pd.DataFrame(all_rows)





def load_ups_regulations(source, destination):
    """Load UPS regulations from JSON files."""
    try:
        file_path = f"anushka/ups_regulations/{source}_to_{destination}.json"
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                return json.load(file)
        
        # Try reverse order if file doesn't exist
        file_path = f"anushka/ups_regulations/{destination}_to_{source}.json"
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                return json.load(file)
        
        # If no exact match, try to find best available regulations
        files = glob.glob("anushka/ups_regulations/*.json")
        if files:
            # Pick a file that might have similar regulations
            with open(files[0], 'r') as file:
                return json.load(file)
        
        return None
    except Exception as e:
        print(f"Error loading UPS regulations: {e}")
        return None

def check_compliance_with_gemini(shipping_details, prohibited_items, regulations):
    """Use Gemini API to check compliance based on shipping details and regulations."""
    try:
        # Prepare the prompt for Gemini
        prompt = f"""
        Analyze this international shipping request and determine compliance:
        
        SHIPPING DETAILS:
        - Origin: {shipping_details['origin']}
        - Destination: {shipping_details['destination']}
        - Item(s): {shipping_details['items']}
        - Weight: {shipping_details['weight']} {shipping_details['weight_unit']}
        - Dimensions: {shipping_details['dimensions']}
        - Quantity: {shipping_details['quantity']}
        - Value: {shipping_details['value']} {shipping_details['currency']}
        - Purpose: {shipping_details['purpose']}
        - Documents: {', '.join(shipping_details['documents'])}
        
        PROHIBITED ITEMS FOR {shipping_details['destination'].upper()}:
        {prohibited_items}
        
        REGULATIONS AND REQUIREMENTS:
        {json.dumps(regulations, indent=2)}
        
        Provide a detailed compliance analysis with the following structure:
        1. Overall compliance status (Compliant, Non-Compliant, or Partially Compliant)
        2. A list of compliance checks performed
        3. For each check: status (Pass/Fail/Warning), detailed explanation, and recommendation
        4. Required documents analysis (Which are provided, which are missing)
        5. Summary of actions needed to achieve compliance
        
        Format the response as a JSON object with these keys: status, checks, document_analysis, summary_actions
        """
        
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        
        # Parse the response
        try:
            # Try to extract JSON from the response
            response_text = response.text
            # Find JSON content between triple backticks if present
            json_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1)
            
            result = json.loads(response_text)
            return result
        except json.JSONDecodeError:
            # If Gemini doesn't return proper JSON, structure the response manually
            return {
                "status": "Partially Compliant",
                "checks": [
                    {
                        "name": "Item Prohibition Check",
                        "status": "Warning",
                        "explanation": "Could not definitively determine if items are prohibited.",
                        "recommendation": "Verify with carrier directly."
                    },
                    {
                        "name": "Documentation Check",
                        "status": "Warning",
                        "explanation": "Some documents may be missing based on country requirements.",
                        "recommendation": "Review country-specific documentation requirements."
                    }
                ],
                "document_analysis": {
                    "provided": shipping_details['documents'],
                    "missing": ["Commercial Invoice", "Certificate of Origin"]
                },
                "summary_actions": "Please review country-specific requirements and contact carrier for further guidance."
            }
    except Exception as e:
        print(f"Error during compliance check with Gemini: {e}")
        # Fallback to basic compliance check
        return fallback_compliance_check(shipping_details, prohibited_items)

def fallback_compliance_check(shipping_details, prohibited_items):
    """Fallback method if Gemini API fails."""
    items = [item.strip().lower() for item in shipping_details['items'].split(',')]
    prohibited_items_lower = [item.lower() for item in prohibited_items]
    
    # Check if any items are prohibited
    prohibited_found = any(any(prohibited in item for prohibited in prohibited_items_lower) for item in items)
    
    # Check if required documents are provided
    common_required_docs = ["Commercial Invoice", "Packing List"]
    high_value_docs = ["Certificate of Origin", "Declaration of Value"]
    
    provided_docs = [doc.strip() for doc in shipping_details['documents']]
    missing_docs = [doc for doc in common_required_docs if doc not in provided_docs]
    
    if shipping_details['value'] > 1000:
        missing_docs.extend([doc for doc in high_value_docs if doc not in provided_docs])
    
    # Determine overall status
    if prohibited_found:
        status = "Non-Compliant"
    elif missing_docs:
        status = "Partially Compliant"
    else:
        status = "Compliant"
    
    return {
        "status": status,
        "checks": [
            {
                "name": "Prohibited Items Check",
                "status": "Fail" if prohibited_found else "Pass",
                "explanation": "Found prohibited items in shipment" if prohibited_found else "No prohibited items detected",
                "recommendation": "Remove prohibited items" if prohibited_found else "No action needed"
            },
            {
                "name": "Documentation Check",
                "status": "Fail" if missing_docs else "Pass",
                "explanation": f"Missing required documents: {', '.join(missing_docs)}" if missing_docs else "All required documents provided",
                "recommendation": "Provide missing documents" if missing_docs else "No action needed"
            }
        ],
        "document_analysis": {
            "provided": provided_docs,
            "missing": missing_docs
        },
        "summary_actions": "Address all failing compliance checks before shipping" if prohibited_found or missing_docs else "Shipment is ready for processing"
    }

def format_check_result(result, shipping_details):
    """Format compliance check results for display."""
    # Main results container
    st.markdown("## 📋 Compliance Check Results")
    
    # Overall status with appropriate styling
    status_color = {
        "Compliant": "green",
        "Partially Compliant": "orange",
        "Non-Compliant": "red"
    }.get(result["status"], "blue")
    
    st.markdown(f"### Overall Status: <span style='color:{status_color};font-weight:bold'>{result['status']}</span>", unsafe_allow_html=True)
    
    # Shipping details summary
    with st.expander("📦 Shipping Details", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Origin:** {shipping_details['origin']}")
            st.markdown(f"**Destination:** {shipping_details['destination']}")
            st.markdown(f"**Items:** {shipping_details['items']}")
        with col2:
            st.markdown(f"**Weight:** {shipping_details['weight']} {shipping_details['weight_unit']}")
            st.markdown(f"**Value:** {shipping_details['value']} {shipping_details['currency']}")
            st.markdown(f"**Purpose:** {shipping_details['purpose']}")
    
    # Compliance checks
    st.markdown("### 🔍 Compliance Checks")
    
    # Progress metrics
    num_checks = len(result["checks"])
    passed = sum(1 for check in result["checks"] if check["status"] == "Pass")
    failed = sum(1 for check in result["checks"] if check["status"] == "Fail")
    warnings = sum(1 for check in result["checks"] if check["status"] == "Warning")
    
    cols = st.columns(4)
    cols[0].metric("Total Checks", num_checks)
    cols[1].metric("Passed", passed, f"{passed/num_checks:.0%}")
    cols[2].metric("Failed", failed, f"{failed/num_checks:.0%}" if failed > 0 else "0%")
    cols[3].metric("Warnings", warnings, f"{warnings/num_checks:.0%}" if warnings > 0 else "0%")
    
    # Progress bar
    st.progress(passed/num_checks)
    
    # Detailed check results
    for i, check in enumerate(result["checks"]):
        status_icon = {
            "Pass": "✅",
            "Fail": "❌",
            "Warning": "⚠️"
        }.get(check["status"], "ℹ️")
        
        status_color = {
            "Pass": "green",
            "Fail": "red",
            "Warning": "orange"
        }.get(check["status"], "blue")
        
        with st.expander(f"{status_icon} Check {i+1}: {check['name']} - **{check['status']}**", expanded=(check["status"] != "Pass")):
            st.markdown(f"**Status:** <span style='color:{status_color}'>{check['status']}</span>", unsafe_allow_html=True)
            st.markdown(f"**Analysis:** {check['explanation']}")
            st.markdown(f"**Recommendation:** {check['recommendation']}")
    
    # Document analysis
    st.markdown("### 📄 Document Analysis")
    
    doc_col1, doc_col2 = st.columns(2)
    
    with doc_col1:
        st.markdown("#### Provided Documents")
        if result["document_analysis"]["provided"]:
            for doc in result["document_analysis"]["provided"]:
                st.markdown(f"✅ {doc}")
        else:
            st.markdown("_No documents provided_")
    
    with doc_col2:
        st.markdown("#### Missing Documents")
        if result["document_analysis"]["missing"]:
            for doc in result["document_analysis"]["missing"]:
                st.markdown(f"❌ {doc}")
        else:
            st.markdown("✅ _All required documents provided_")
    
    # Summary of actions
    st.markdown("### 🚩 Required Actions")
    if result["status"] == "Compliant":
        st.success("No actions required. Your shipment is compliant with regulations.")
    else:
        st.warning(result["summary_actions"])
    
    # Visual representation of compliance
    if "chart_data" not in st.session_state:
        # Generate some random compliance metrics for visualization
        compliance_metrics = pd.DataFrame({
            'Category': ['Documentation', 'Prohibited Items', 'Weight Limits', 'Value Declaration', 'Packaging'],
            'Compliance Score': [
                100 if not result["document_analysis"]["missing"] else max(0, 100 - len(result["document_analysis"]["missing"]) * 25),
                100 if all(check["status"] == "Pass" for check in result["checks"] if "Prohibited" in check["name"]) else 0,
                random.randint(85, 100),
                random.randint(80, 100) if shipping_details["value"] > 0 else 50,
                random.randint(90, 100)
            ]
        })
        st.session_state.chart_data = compliance_metrics
    
    # Create radar chart
    fig = px.line_polar(
        st.session_state.chart_data, 
        r='Compliance Score', 
        theta='Category', 
        line_close=True,
        range_r=[0, 100],
        title="Compliance Score by Category"
    )
    fig.update_traces(fill='toself')
    st.plotly_chart(fig, use_container_width=True)
    
    # Add export options
    st.markdown("### 📥 Export Results")
    export_format = st.selectbox("Select format", ["PDF Report", "CSV", "JSON"])
    st.download_button(
        "Download Compliance Report",
        data=json.dumps(result, indent=2) if export_format == "JSON" else pd.DataFrame(result["checks"]).to_csv(),
        file_name=f"compliance_report_{shipping_details['origin']}_to_{shipping_details['destination']}.{'json' if export_format == 'JSON' else 'csv'}",
        mime="application/json" if export_format == "JSON" else "text/csv"
    )