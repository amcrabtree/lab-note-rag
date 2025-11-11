import streamlit as st
from datetime import datetime
import os
from pdf_to_text_gpt4o import extract_text_from_image_openai
from dotenv import load_dotenv
import pandas as pd
from io import BytesIO
import fitz  # PyMuPDF
from PIL import Image
import openai

DATABASE_DIR = "./database"
IMAGE_DIR = "./database/img"
DATABASE_TSV = os.path.join(DATABASE_DIR, "lab_book_database.tsv")
IMAGE_DPI = 100
API_PASSWORD = "yeast"

def split_double_page(image):
    width, height = image.size
    if width > height * 1.2:  # heuristic for double pages
        mid_x = width // 2
        return [image.crop((0, 0, mid_x+50, height)), image.crop((mid_x-50, 0, width, height))]
    return [image]

extraction_instructions = """
    You are a microbiology researcher who is tasked with transcribing all the handwritten notes 
    from images of lab notebook pages. Extract all handwritten text from this lab notebook page 
    and return it as markdown text. 
    Bold any text that looks like a date. 
    Italicise any text in Latin. 
    If a date looks like the start of a new notebook entry, then make it a header. 
    If you detect there is an agarose gel image, write a hash tag "#gel_image" for each gel image detected. 
    if you detect a hand-drawn agar plate diagram, write a hash tag "#plate_diagram" for each plate diagram detected. 
    There is a common strain named BJH001, sometimes just referred to as 001, which is referenced often.
    Other common terms include "Okada" as a type of dsRNA extraction protocol. 
    If you see a page number at the very bottom of the page, append it to each date header on that page.
    """
# --------- Session State ---------
if "timestamp" not in st.session_state:
    st.session_state.timestamp = None
if "reset_counter" not in st.session_state:
    st.session_state.reset_counter = 0
if 'current_page_img' not in st.session_state:
    st.session_state.current_page = None
if 'current_page_text' not in st.session_state:
    st.session_state.current_page_text = None
if 'current_page_number' not in st.session_state:
    st.session_state.current_page_number = None
if 'database_df' not in st.session_state:
    if os.path.exists(DATABASE_TSV):
        st.session_state.database_df = pd.read_csv(DATABASE_TSV, sep="\t")
    else:
        st.session_state.database_df = None
if 'uploaded_df' not in st.session_state:
    st.session_state.uploaded_df = None
if 'starting_page_number' not in st.session_state:
    st.session_state.starting_page_number = None
if 'current_image_dict' not in st.session_state:
    st.session_state.current_image_dict = None
if 'password' not in st.session_state:
    st.session_state.password = None
if 'client' not in st.session_state:
    st.session_state.client = None

# ---------- Page Config ----------
st.set_page_config(
    page_title="Lab Note Finder",
    page_icon="🧫",
    layout="wide"
)

# ----------------------------
# Header
# ----------------------------
st.title("🧫 Lab Note Finder")
st.caption("AI-powered lab notebook search.")

# ----------------------------
# Tabs
# ----------------------------
tab1, tab2 = st.tabs(["📚 Upload Notebook", "🔍 Query Notebook"])

# ----------------------------
# TAB 1: Upload / Update
# ----------------------------
with tab1:
    st.subheader("📚 Upload Lab Notebook")

    if st.session_state.client is None:
        st.session_state.password = st.text_input("Enter your password", type="password")

    if st.session_state.password != "":
        if st.session_state.password != API_PASSWORD:
            st.markdown("Password is incorrect. Try again.")
        else:
            st.session_state.client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    if st.session_state.client is not None:
        # --- UI ---
        uploaded_lab_book = st.file_uploader("Upload a combined lab notebook PDF", type=["pdf"],
                                            key=f"upload_{st.session_state.reset_counter}")

        col1, col2 = st.columns(2)
        with col1:
            lab_book_name = st.text_input("Enter lab book name (ex: 'AMC Book 1')",
                                        key=f"name_{st.session_state.reset_counter}")
        with col2:
            starting_page_number = st.number_input("Enter notebook starting page number", 
                                                min_value=1, step=1, value=1,
                                                key=f"page_{st.session_state.reset_counter}")

        # if st.session_state.uploaded_df is not None:
        #     if st.button("Clear Data"):
        #         st.session_state.reset_counter += 1  # change keys → widgets reset visually
        #         st.session_state.uploaded_df = None
        #         st.rerun()

        # --- LOAD ---

        if st.button("Load", type="primary", key=f"load_{st.session_state.reset_counter}"):
            st.success(f"📄 {uploaded_lab_book.name} uploaded successfully!")
            st.session_state.uploaded_df = None
            st.session_state.current_image_dict = None
            st.session_state.current_page_number = starting_page_number
            st.session_state.starting_page_number = starting_page_number

            current_image_dict = {}
            st.session_state.timestamp = datetime.now().strftime("%Y-%m-%d")

            # Read uploaded PDF into memory
            pdf_bytes = uploaded_lab_book.read()

            # Open PDF with PyMuPDF
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            text_df = pd.DataFrame()

            with st.spinner("Extracting text from scanned pages- this may take about 1 minute/page", show_time=True):
                page_counter = starting_page_number
                for i, page in enumerate(doc):

                    # Render page to a pixmap
                    pix = page.get_pixmap(dpi=IMAGE_DPI)
                    raw_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    img_list = split_double_page(raw_img)

                    for img in img_list: 
                        current_image_dict[page_counter] = img

                        # Extract text and add to pandas dataframe
                        text_output = extract_text_from_image_openai(img, st.session_state.client, extraction_instructions)
                        row_df = pd.DataFrame({
                            'lab_book_name': [lab_book_name],
                            'scanned_page_number': [i+1],
                            'physical_page_number': [page_counter],
                            'timestamp': [st.session_state.timestamp],
                            'image_filename': [f'{lab_book_name}_Page{page_counter}.png'],
                            'extracted_text': [text_output],
                        })
                        text_df = pd.concat([text_df, row_df])

                        page_counter += 1

                st.session_state.uploaded_df = text_df.copy()
                st.session_state.current_image_dict = current_image_dict.copy()
            doc.close()

        # --- DISPLAY ---

        if (st.session_state.uploaded_df is not None) and (st.session_state.current_image_dict is not None):
            text_df = st.session_state.uploaded_df
            current_page_number = st.session_state.current_page_number
            starting_page_number = st.session_state.starting_page_number
            max_page_number = len(text_df) + st.session_state.starting_page_number - 1
            safe_name = lab_book_name.replace(" ", "_")

            st.divider()
            st.subheader(f"Page {current_page_number}")
            
            col1_img, col2_text = st.columns(2)

            with col1_img:
                st.session_state.current_page_image = st.session_state.current_image_dict[current_page_number]
                st.image(st.session_state.current_page_image, width='stretch')

            with col2_text:
                st.session_state.current_page_text = text_df[text_df['physical_page_number']==current_page_number]['extracted_text'][0]
                input_text = st.text_area("Lab book page content", value=st.session_state.current_page_text, width='stretch', height='stretch')

            col_back_page, _, _, _, _, col_forward_page = st.columns(6)

            with col_back_page:
                if current_page_number > starting_page_number:
                    if st.button("Backward", width="stretch"):
                        text_df.loc[text_df['physical_page_number'] == current_page_number, 'extracted_text'] = input_text
                        st.session_state.current_page_number -= 1
                        st.rerun()

            with col_forward_page:
                if current_page_number < max_page_number:
                    if st.button("Forward", width="stretch"):
                        text_df.loc[text_df['physical_page_number'] == current_page_number, 'extracted_text'] = input_text
                        st.session_state.current_page_number += 1
                        st.rerun()

            # Save DataFrame to disk
            if st.button("Save Updates", type="primary"):
                text_df.loc[text_df['physical_page_number'] == current_page_number, 'extracted_text'] = input_text
                
                # Load this lab book into database
                if os.path.exists(DATABASE_TSV):
                    database_df = pd.read_csv(DATABASE_TSV, sep="\t")
                else: 
                    os.makedirs(DATABASE_DIR, exist_ok=True)
                    database_df = pd.DataFrame()
                    
                # Make sure timestamp is a datetime type
                for df in [database_df, text_df]:
                    if len(df) > 0:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])

                # Concatenate the dataframes
                combined_df = pd.concat([database_df, text_df], ignore_index=True)
                combined_df = combined_df.sort_values(by='timestamp', ascending=False)

                # Drop duplicates based on lab_book_name and scanned_page_number, keeping the first (latest)
                latest_df = combined_df.drop_duplicates(subset=['lab_book_name', 'scanned_page_number'], keep='first')
                latest_df = latest_df.reset_index(drop=True)
                latest_df.to_csv(DATABASE_TSV, index=False)

                # Save image data
                os.makedirs(IMAGE_DIR, exist_ok=True)
                for page_number in range(starting_page_number, max_page_number):
                    image_filename = f"{safe_name}_Page{page_number}.png"
                    png_path = os.path.join(IMAGE_DIR, image_filename)
                    page_img = st.session_state.current_image_dict[page_number]
                    buf = BytesIO()
                    page_img.save(buf, format="PNG")
                    with open(png_path, "wb") as f:
                        f.write(buf.getbuffer())

                st.success("✅ Updates saved")

            # Offer immediate download of the dataframe as CSV
            safe_name = lab_book_name.replace(" ", "_")
            csv_buf = BytesIO()
            st.session_state.uploaded_df.to_csv(csv_buf, sep="\t", index=False)
            csv_buf.seek(0)
            st.download_button(
                label="Download extracted text (TSV)",
                data=csv_buf.getvalue(),
                file_name=f"{safe_name}_{st.session_state.timestamp}.tsv",
                mime="text"
            )


# ----------------------------
# TAB 2: Query Interface
# ----------------------------
with tab2:
    st.subheader("🔍 Ask your lab notebook")

    if st.session_state.uploaded_df is None or st.session_state.password is None:
        "First upload a lab notebook."
    else:
        notebook_df = st.session_state.uploaded_df

        # Create a single string containing all notebook entries
        # This will not be displayed to the user
        all_notebook_text = ""
        for i, row in notebook_df.iterrows():
            all_notebook_text = "\n\n".join(
                f"Lab Notebook: {row['lab_book_name']}\n"
                f"Page Number: {row['physical_page_number']}\n"
                f"Text: {row['extracted_text']}"
                for _, row in notebook_df.iterrows()
            )

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": "Hi! 👋 What would you like to know about your lab notebook?"}
            ]

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Get user query
        query = st.chat_input("Ask a question about any lab notebook entry")

        if query:
            # Display user message
            st.chat_message("user").markdown(query)
            st.session_state.messages.append({"role": "user", "content": query})

            with st.chat_message("assistant"):
                with st.spinner("Looking up your notebook..."):

                    # ---------- Step 1: Ask GPT to find relevant entries ----------
                    # GPT will scan the single string of all notebook entries
                    prompt = f"""
                    You are an expert lab assistant. The user will ask a question about their lab notebooks.
                    Use the following lab notebook data to answer the question. The data contains multiple pages from different notebooks:

                    {all_notebook_text}

                    Question: {query}

                    Instructions:
                    1. Identify all relevant pages that answer the question.
                    2. For each relevant page, provide the lab notebook name and physical page number.
                    3. Summarize the content only enough to answer the user's question.
                    4. Keep the answer clear and concise.
                    """

                    # Call OpenAI API
                    response_obj = st.session_state.client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=500
                    )
                    response = response_obj.choices[0].message.content.strip()

                    # Display and store response
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})




# with tab3:
#     st.subheader("🔍 Search Notebook Database")

#     st.markdown(
#         """
#         Select a notebook and page number to view the corresponding scanned page.  
#         Useful for double-checking transcriptions or revisiting original data.
#         """
#     )

#     if st.session_state.uploaded_df is not None:

#         # Example: list of notebooks (replace with dynamic list from your DB or folder)
#         available_notebooks = list(st.session_state.uploaded_df['lab_book_name'].unique())
#         selected_notebook = st.selectbox("Select a lab notebook:", available_notebooks)

#         page_number = st.number_input(
#             "Enter page number:",
#             min_value=1,
#             step=1,
#             format="%d",
#             key="page_number_input",
#         )

#         st.button("🔍 View Page", key="view_page_button")

#         # Placeholder example image (replace with your real image loading logic)
#         st.divider()
#         example_image_path = "example_lab_page.png"  # replace dynamically later

#         if os.path.exists(example_image_path):
#             st.image(example_image_path, caption=f"{selected_notebook} — Page {page_number}", use_container_width=True)
#         else:
#             st.info("Image preview will appear here once linked to your notebook image directory.")

#         st.caption("Note: This feature will display a PNG or JPG of the requested page once the image directory is connected.")

