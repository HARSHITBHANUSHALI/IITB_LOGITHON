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
from myfiles.first import extract_text_from_pdf,get_nlp_model,parse_country_data,process_fedex_pdf,get_embedding_model,fix_malformed_countries
from myfiles.first import create_vector_database,initialize_system,query_vector_database,get_prohibited_items_for_country,search_prohibited_items,extract_entities,format_country_items_response,chatbot_response,get_image_description,get_multiple_items_from_image,create_results_dataframe
from myfiles.first import load_ups_regulations,check_compliance_with_gemini,fallback_compliance_check,format_check_result
# Set page configuration
st.set_page_config(
    page_title="International Prohibited Items Search",
    page_icon="🌎",
    layout="wide"
)

# Global variables to keep resources loaded
_model = None
_index = None
_all_items = None
_country_map = None
_country_data = None
_nlp = None


# Gemini API Configuration
GEMINI_API_KEY = "AIzaSyDqMg4cv_n04wbxo16Bpovc01LXAa96h_I"  # Replace with your actual Gemini API key
genai.configure(api_key=GEMINI_API_KEY)


def load_country_data_from_file(file_path):
    """Load country data from a text file."""
    with open(file_path, 'r') as file:
        countries_list = eval(file.read())
    countries_data = {country: [] for country in countries_list}
    return countries_data

def main():
    st.title("🌎 International Prohibited Items Search")
    st.write("Search for prohibited items by country, find countries that prohibit specific items, or chat with me!")
    db_dir = "vector_db"  # Use your default or get from sidebar
    initialize_system(db_dir)
    
    # Get country options globally
    global _country_data  # Add this line to ensure _country_data is global
    _country_data = load_country_data_from_file("countries.txt")
    country_options = []
    if _country_data:
        country_options = sorted(list(_country_data.keys()))
        print("country options are",country_options)
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    with st.sidebar:
        st.header("Database Setup")
        pdf_file = st.text_input("PDF File Path", "fedex-international-connect-country-region-specific-prohibited-and-restricted-items[1].pdf")
        output_file = st.text_input("Output JSON Path", "fedex_country_restrictions2.json")
        db_dir = st.text_input("Vector Database Directory", "vector_db")
        
        db_exists = all(os.path.exists(os.path.join(db_dir, fname)) for fname in ["items_index.faiss", "items.pkl", "country_map.pkl", "country_data.json"])
        
        if db_exists:
            st.success("Vector database found!")
            if st.button("Reload Database"):
                initialize_system(db_dir, force_reload=True)
        else:
            st.warning("Vector database not found. You'll need to process the PDF first.")
        
        process_pdf = st.checkbox("Process PDF", value=not db_exists)
        if process_pdf:
            if st.button("Process FedEx PDF"):
                if not Path(pdf_file).exists():
                    st.error(f"Error: PDF file '{pdf_file}' not found.")
                else:
                    with st.spinner("Processing PDF..."):
                        data = process_fedex_pdf(pdf_file, output_file)
                        data = fix_malformed_countries(output_file)
                        create_vector_database(data, db_dir)
                        initialize_system(db_dir, force_reload=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["Search by Country", "Search by Item", "Search by Item 2", "Chat with DB"])
    
    with tab1:
        st.header("Search Prohibited Items by Country")
        initialize_system(db_dir)
        country_options = sorted(list(_country_data.keys())) if _country_data else []
        selected_country = st.selectbox("Select a country", [""] + country_options)
        
        if selected_country:
            start_time = time.time()
            result = get_prohibited_items_for_country(selected_country)
            query_time = time.time() - start_time
            
            if "error" in result:
                st.error(result["error"])
            else:
                st.success(f"Found {result['count']} prohibited items for {result['country']} (query time: {query_time:.3f}s)")
                for item in result["items"]:
                    st.markdown(f"- {item}")
    
    with tab2:
        st.header("Search Countries by Prohibited Item")
        initialize_system(db_dir)
        
        # Add image upload option
        uploaded_file = st.file_uploader("Upload an image of the item", type=['png', 'jpg', 'jpeg'])
        
        if uploaded_file:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Get image description
            with st.spinner("Analyzing image..."):
                description = get_image_description(image)
                if description:
                    st.success(f"Detected item: {description}")
                    # Auto-fill the search query with the description
                    query = description
                else:
                    st.error("Could not analyze the image. Please try again or enter your query manually.")
                    query = st.text_input("Enter item description (e.g., 'weapons', 'alcohol', 'tobacco')")
        else:
            query = st.text_input("Enter item description (e.g., 'weapons', 'alcohol', 'tobacco')")
        
        top_k = st.slider("Maximum number of results", 5, 50, 20)
        
        if query:
            start_time = time.time()
            results = search_prohibited_items(query, top_k)
            query_time = time.time() - start_time
            
            if isinstance(results, dict) and "error" in results:
                st.error(results["error"])
            else:
                st.success(f"Found prohibitions in {len(results)} countries (query time: {query_time:.3f}s)")
                for country_result in results:
                    with st.expander(f"{country_result['country']} ({country_result['count']} items)"):
                        for i, (item, score) in enumerate(zip(country_result["items"], country_result["scores"])):
                            if score > 0.6:
                                st.markdown(f"- {item} (relevance score: {score:.3f})")

    with tab3:
        st.header("Multi-Item Search")
        initialize_system(db_dir)
        
        uploaded_file = st.file_uploader("Upload an image with multiple items", type=['png', 'jpg', 'jpeg'], key="multi_upload")
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            with st.spinner("Analyzing image for multiple items..."):
                detected_items = get_multiple_items_from_image(image)
                if detected_items:
                    st.success(f"Detected {len(detected_items)} items:")
                    for item in detected_items:
                        st.markdown(f"- {item}")
                else:
                    detected_items = []
                    st.warning("No items detected. Please enter items manually.")
        else:
            detected_items = []
        
        items_text = st.text_area(
            "Enter items (one per line) or edit detected items:",
            value="\n".join(detected_items) if detected_items else "",
            height=100
        )
        
        search_items = [item.strip() for item in items_text.split("\n") if item.strip()]
        
        if search_items:
            top_k = st.slider("Maximum results per item", 5, 50, 20, key="multi_slider")
            
            with st.spinner("Searching for multiple items..."):
                all_results = {}
                for item in search_items:
                    results = search_prohibited_items(item, top_k)
                    if not isinstance(results, dict):
                        all_results[item] = results
                
                if all_results:
                    df = create_results_dataframe(all_results)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        selected_items = st.multiselect(
                            "Filter by Search Item",
                            options=sorted(df["Search Item"].unique()),
                            default=sorted(df["Search Item"].unique())
                        )
                    with col2:
                        selected_countries = st.multiselect(
                            "Filter by Country",
                            options=sorted(df["Country"].unique()),
                            default=sorted(df["Country"].unique()))
                    
                    filtered_df = df[
                        df["Search Item"].isin(selected_items) &
                        df["Country"].isin(selected_countries)
                    ]
                    
                    st.dataframe(
                        filtered_df,
                        column_config={
                            "Relevance": st.column_config.ProgressColumn(
                                "Relevance",
                                min_value=0,
                                max_value=1,
                                format="%.3f"
                            )
                        },
                        hide_index=True
                    )
                    
                    csv = filtered_df.to_csv(index=False)
                    st.download_button(
                        "Download Results",
                        csv,
                        "prohibited_items_search.csv",
                        "text/csv",
                        key='download-csv'
                    )
                    
                    # Add visualizations
                    st.subheader("Visualizations")
                    viz_col1, viz_col2 = st.columns(2)
                    
                    with viz_col1:
                        st.subheader("🌍 Country Prohibition Intensity")
                        filtered_df['Relevance'] = pd.to_numeric(filtered_df['Relevance'])
                        pivot_df = filtered_df.pivot_table(
                            index='Search Item',
                            columns='Country',
                            values='Relevance',
                            aggfunc='mean'
                        ).fillna(0)
                        fig_heatmap = px.imshow(
                            pivot_df,
                            color_continuous_scale='RdYlBu_r',
                            title='Country Prohibition Intensity',
                            labels={'x': 'Country', 'y': 'Search Item', 'color': 'Relevance Score'}
                        )
                        fig_heatmap.update_layout(
                            height=400,
                            margin=dict(l=0, r=0, t=30, b=0)
                        )
                        st.plotly_chart(fig_heatmap, use_container_width=True)
                    
                    with viz_col2:
                        st.subheader("🌟 Restriction Hierarchy")
                        fig_sunburst = px.sunburst(
                            filtered_df,
                            path=['Search Item', 'Country'],
                            values='Relevance',
                            title='Hierarchical View of Restrictions',
                            color='Relevance',
                            color_continuous_scale='RdYlBu_r'
                        )
                        fig_sunburst.update_layout(
                            height=400,
                            margin=dict(l=0, r=0, t=30, b=0)
                        )
                        st.plotly_chart(fig_sunburst, use_container_width=True)
                    
                    st.subheader("📊 Prohibition Distribution")
                    fig_bar = px.bar(
                        filtered_df,
                        x='Country',
                        y='Relevance',
                        color='Search Item',
                        title='Prohibition Distribution by Country and Item',
                        labels={'Relevance': 'Relevance Score'},
                        barmode='group'
                    )
                    fig_bar.update_layout(
                        height=400,
                        margin=dict(l=0, r=0, t=30, b=0)
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)


        
    with tab4:
        st.header("Chat with Assistant about Prohibited Items")
        st.write("Ask me anything about international shipping restrictions!")
        
        # Initialize chat history in session state if it doesn't exist
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        # Add a button to clear chat history
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
        
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        user_input = st.chat_input("Type your question here (e.g., 'Is alcohol prohibited in Japan?')")
        
        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(user_input)
            
            # Generate and display assistant response with memory
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Pass the entire chat history to the response function
                    response = chatbot_response(user_input, st.session_state.chat_history)
                    st.markdown(response)
                
                # Add assistant response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()


