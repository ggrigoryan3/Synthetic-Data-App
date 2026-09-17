import sys
import os
from pathlib import Path
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent.resolve()))

load_dotenv()

import streamlit as st
import pandas as pd
import psycopg2
from schema_parser import parse_ddl_with_gemini
from data_generator import DataGenerationEngine, export_to_zip

st.set_page_config(page_title="Synthetic Data Platform", layout="wide")

GCP_PROJECT = os.getenv("GCP_PROJECT")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")

if not GCP_PROJECT or GCP_PROJECT == "your-gcp-project":
    st.error("Error: GCP_PROJECT is missing or improperly configured in your `.env` file!")
    st.stop()

if "generated_tables" not in st.session_state:
    st.session_state.generated_tables = {}
if "parsed_schema" not in st.session_state:
    st.session_state.parsed_schema = None

st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Go to", ["Data Generation", "Talk to your data"])

def save_tables_to_postgres(tables_dict):
    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        database=os.getenv("POSTGRES_DB", "synthetic_data"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "postgrespassword")
    )
    cursor = conn.cursor()
    for table_name, df in tables_dict.items():
        cursor.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE;")
        
        cols = ", ".join([f'"{col}" TEXT' for col in df.columns])
        cursor.execute(f'CREATE TABLE {table_name} ({cols});')
        
        for _, row in df.iterrows():
            vals = list(row.values)
            placeholders = ", ".join(["%s"] * len(vals))
            cursor.execute(f'INSERT INTO {table_name} VALUES ({placeholders})', vals)
            
    conn.commit()
    cursor.close()
    conn.close()

if app_mode == "Data Generation":
    st.header("Synthetic Data Engine")

    col1, col2 = st.columns([1, 1])

    with col1:
        uploaded_file = st.file_uploader("Upload DDL Schema (.sql, .ddl, .txt)", type=["sql", "ddl", "txt"])
        custom_instructions = st.text_area("Additional Data Generation Instructions/Prompts", 
                                           placeholder="e.g., Make order dates within 2024, set email addresses from @example.com domain...")

    with col2:
        temperature = st.slider("Model Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.05)
        row_count = st.number_input("Rows per Table", min_value=1, max_value=1000, value=10)

    if st.button("Generate Data", type="primary"):
        if uploaded_file is not None:
            ddl_content = uploaded_file.getvalue().decode("utf-8")
            
            with st.spinner("Parsing DDL Schema..."):
                schema = parse_ddl_with_gemini(ddl_content, GCP_PROJECT, GCP_LOCATION)
                st.session_state.parsed_schema = schema

            engine = DataGenerationEngine(GCP_PROJECT, GCP_LOCATION)
            
            sorted_tables = sorted(schema.tables, key=lambda x: x.dependency_order)
            
            generated_data = {}
            progress_bar = st.progress(0)
            
            for index, table in enumerate(sorted_tables):
                with st.spinner(f"Generating synthetic records for '{table.table_name}'..."):
                    df = engine.generate_table_data(
                        table_schema=table,
                        user_instructions=custom_instructions,
                        parent_data=generated_data,
                        row_count=row_count,
                        temperature=temperature
                    )
                    generated_data[table.table_name] = df
                progress_bar.progress((index + 1) / len(sorted_tables))

            st.session_state.generated_tables = generated_data
            st.success("Synthetic Data Generation Complete!")

    if st.session_state.generated_tables:
        st.divider()
        st.subheader("Data Inspector & Refinement Module")

        tabs = st.tabs(list(st.session_state.generated_tables.keys()))

        engine = DataGenerationEngine(GCP_PROJECT, GCP_LOCATION)

        for idx, tab_name in enumerate(st.session_state.generated_tables.keys()):
            with tabs[idx]:
                df_curr = st.session_state.generated_tables[tab_name]
                st.dataframe(df_curr, use_container_width=True)

                with st.form(key=f"mod_form_{tab_name}"):
                    mod_prompt = st.text_input(f"Enter feedback to adjust table '{tab_name}'", 
                                              placeholder="e.g., Multiply all prices by 1.15, make status column uppercase...")
                    submit_mod = st.form_submit_button("Submit Modification")

                    if submit_mod and mod_prompt:
                        with st.spinner(f"Updating table '{tab_name}'..."):
                            updated_df = engine.modify_table_data(df_curr, mod_prompt, temperature)
                            st.session_state.generated_tables[tab_name] = updated_df
                            st.rerun()

        st.divider()
        c1, c2 = st.columns([1, 1])

        with c1:
            zip_bytes = export_to_zip(st.session_state.generated_tables)
            st.download_button(
                label="Download Generated Dataset (.zip)",
                data=zip_bytes,
                file_name="synthetic_dataset.zip",
                mime="application/zip",
                use_container_width=True
            )

        with c2:
            if st.button("Save to System Database", use_container_width=True):
                try:
                    save_tables_to_postgres(st.session_state.generated_tables)
                    st.success("Successfully ingested dataset into PostgreSQL!")
                except Exception as e:
                    st.error(f"Failed writing to database: {e}")

elif app_mode == "Talk to your data":
    st.header("Talk to your data")
    st.info("Phase 2 Query Workspace: Select saved tables in PostgreSQL to run Natural Language queries using Gemini 2.0 Flash.")