import os
import sys
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional
from dotenv import load_dotenv
from google import genai
from google.genai import types
from langfuse.decorators import observe, langfuse_context

sys.path.append(str(Path(__file__).parent.resolve()))
load_dotenv()

class ColumnSchema(BaseModel):
    name: str
    data_type: str
    is_primary_key: bool
    is_foreign_key: bool
    references_table: Optional[str] = None
    references_column: Optional[str] = None
    is_nullable: bool
    constraints_description: Optional[str] = None

class TableSchema(BaseModel):
    table_name: str
    columns: List[ColumnSchema]
    dependency_order: int = Field(description="Order of creation based on foreign key dependencies (0 for independent tables)")

class DatabaseSchema(BaseModel):
    tables: List[TableSchema]

@observe(name="parse_ddl_schema")
def parse_ddl_with_gemini(ddl_text: str, project_id: str, location: str) -> DatabaseSchema:
    client = genai.Client(vertexai=True, project=project_id, location=location)
    
    langfuse_context.update_current_trace(
        tags=["ddl-parser"],
        metadata={"ddl_length": len(ddl_text)}
    )

    prompt = f"""
    Analyze the following SQL DDL schema. Extract all tables, columns, data types, primary keys, foreign keys, 
    and nullability constraints. Calculate the `dependency_order` for each table so that tables without foreign keys 
    have order 0, tables referencing order 0 have order 1, etc.
    
    DDL Schema:
    {ddl_text}
    """
    
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=DatabaseSchema,
            temperature=0.0
        ),
    )
    
    return DatabaseSchema.model_validate_json(response.text)