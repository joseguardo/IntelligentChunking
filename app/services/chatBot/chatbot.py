from dataclasses import dataclass
from utils_chatbot import prompt_enhancer
from typing import Dict, Any, Type, Optional, List
from pydantic import BaseModel, Field, create_model
from datetime import datetime
from dotenv import load_dotenv  
import os
from openai import AsyncOpenAI, OpenAI
import asyncio

@dataclass
class StructuredField:
    name: str
    description: str
    data_type: str  # 'str', 'int', 'float', 'list', 'datetime'

class StructuredChatbot:
    def __init__(self, model: str = "gpt-4.1-mini-2025-04-14"):
        """
        Initialize the chatbot with storage for dynamic models
        """
        self.models: Dict[str, Type[BaseModel]] = {}
        self.model_registry: Dict[str, dict] = {}
        self.model = model
        self._init_openai_clients()


    
    def _init_openai_clients(self) -> None:
        """Initialize async and sync OpenAI clients."""
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.client = AsyncOpenAI(api_key=api_key)
        self.sync_client = OpenAI(api_key=api_key)  # For embeddings
        print("✅ OpenAI clients initialized")



    def _create_pydantic_model(self, fields: List[StructuredField]) -> Optional[type]:
        """Create a dynamic Pydantic BaseModel from structured fields."""
        if not fields:
            return None
        
        # Map data types to Python types
        type_mapping = {
            'str': str,
            'int': int,
            'float': float,
            'list': List[str],  # Default to List[str]
            'datetime': datetime
        }
        
        # Create field definitions for pydantic
        field_definitions = {}
        
        for field in fields:
            python_type = type_mapping.get(field.data_type, str)
            field_definitions[field.name] = (
                python_type, 
                Field(..., description=field.description)
            )
        
        # Create dynamic model
        StructuredResponse = create_model(
            'StructuredResponse',
            **field_definitions
        )
        DynamicModel = create_model('StructuredResponseList', 
                                 items=(List[StructuredResponse], Field(..., description=f"List of structured responses")))
        
        print(f"✅ Created Pydantic model with {len(fields)} fields")
        return DynamicModel
    
    def generate_prompt(self, fields: List[StructuredField], prompt: str) -> str:
        """Generate an enhanced prompt for the chatbot. 
        It takes a list of StructuredField and a base prompt, and returns a engineered prompt. 
        """

        prompt = prompt + "\n\n" + prompt_enhancer + "\n\n" + "Fields:\n"
        for field in fields:
            prompt += f"- {field.name} ({field.data_type}): {field.description}\n"
        return prompt

def generate_enhanced_prompt(self, prompt: str, output_schema) -> str:
    """Generate an enhanced prompt for the chatbot. 
    It takes a base prompt, and returns a engineered prompt. 
    """
    try: 
        completion = self.client.responses.parse(
        model=self.model,
        input= [
            {"role": "system", "content": "You are a prompt engineering expert."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": prompt
                    }
                ]
            }
        ],
        text_format=output_schema
    )
        parsed_result = completion.output_parsed 

        return parsed_result
    except Exception as e:
        print(f"❌ Error generating enhanced prompt: {e}")
        return prompt




class ResponseFields(BaseModel):
    original_text: str = Field(..., description="The original text from which the information was extracted")
    improved_prompt: str = Field(..., description="The enhanced prompt used for extraction")


# Minimalistic test
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model = "gpt-4.1-mini"  
completion = client.responses.parse(
    model=model,
    input= [
        {"role": "system", "content": "You are a prompt engineering expert."},
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": prompt_enhancer
                }
            ]
        }
    ],
    text_format= ResponseFields
)
parsed_result = completion.output_parsed 
print("Parsed Result from minimalistic test:\n", parsed_result)