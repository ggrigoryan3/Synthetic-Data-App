import io
import os
import sys
import zipfile
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from dotenv import load_dotenv
from google import genai
from google.genai import types
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context

sys.path.append(str(Path(__file__).parent.resolve()))
load_dotenv()

langfuse_client = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
)

class DataGenerationEngine:
    def __init__(self, project_id: str, location: str):
        self.client = genai.Client(vertexai=True, project=project_id, location=location)

    @observe(name="generate_table_data")
    def generate_table_data(
        self, 
        table_schema: Any, 
        user_instructions: str, 
        parent_data: Dict[str, pd.DataFrame], 
        row_count: int = 10,
        temperature: float = 0.7
    ) -> pd.DataFrame:
        
        fk_context = {}
        for col in table_schema.columns:
            if col.is_foreign_key and col.references_table in parent_data:
                parent_df = parent_data[col.references_table]
                ref_col = col.references_column
                if ref_col in parent_df.columns:
                    fk_context[col.name] = parent_df[ref_col].tolist()

        prompt = f"""
        Generate synthetic data for table '{table_schema.table_name}'.
        Columns & Constraints: {table_schema.model_dump_json()}
        
        Additional User Formatting/Value Instructions:
        {user_instructions}
        
        Available Foreign Key Value Pools (You MUST sample only existing values from these lists for foreign keys):
        {fk_context}
        
        Generate exactly {row_count} rows. Output JSON array of objects, where keys correspond exactly to column names.
        """
        
        langfuse_context.update_current_trace(
            name=f"generate_{table_schema.table_name}",
            tags=["synthetic-data-gen"],
            metadata={"table": table_schema.table_name, "row_count": row_count}
        )

        response = self.client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=temperature
            )
        )
        
        df = pd.read_json(io.StringIO(response.text))
        return df

    @observe(name="modify_table_data")
    def modify_table_data(self, df: pd.DataFrame, feedback_prompt: str, temperature: float = 0.3) -> pd.DataFrame:
        
        langfuse_context.update_current_trace(
            tags=["table-modification"],
            metadata={"feedback_prompt": feedback_prompt}
        )

        prompt = f"""
        Modify the following tabular dataset according to the user's instructions.
        
        Current Dataset (JSON):
        {df.to_json(orient='records')}
        
        User Instructions:
        {feedback_prompt}
        
        Return the complete updated dataset maintaining identical column names and types.
        """
        
        response = self.client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=temperature
            )
        )
        
        return pd.read_json(io.StringIO(response.text))

@observe(name="export_to_zip")
def export_to_zip(tables_dict: Dict[str, pd.DataFrame]) -> bytes:
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for table_name, df in tables_dict.items():
            csv_data = df.to_csv(index=False)
            zip_file.writestr(f"{table_name}.csv", csv_data)
    zip_buffer.seek(0)
    return zip_buffer.getvalue()