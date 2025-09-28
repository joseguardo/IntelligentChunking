import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, create_model, validator
from typing import Dict, Any, List, Optional, Union, Type, get_type_hints
from decimal import Decimal
from datetime import datetime, date
from enum import Enum
import re
import json
from pathlib import Path
import inspect


class FieldMetadata:
    """Metadata about a field extracted from Excel template"""
    def __init__(self, name: str, excel_column: str, section: str = None):
        self.name = name
        self.excel_column = excel_column
        self.section = section
        self.data_type = str  # Default type
        self.is_optional = True
        self.default_value = None
        self.description = ""
        self.validation_rules = []
        self.enum_values = []
        self.nested_fields = []  # For complex objects
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'excel_column': self.excel_column,
            'section': self.section,
            'data_type': self.data_type.__name__ if hasattr(self.data_type, '__name__') else str(self.data_type),
            'is_optional': self.is_optional,
            'default_value': self.default_value,
            'description': self.description,
            'validation_rules': self.validation_rules,
            'enum_values': self.enum_values
        }


class ExcelTemplateAnalyzer:
    """Analyzes Excel templates to extract field structure and relationships"""
    
    def __init__(self):
        self.field_type_patterns = {
            r'.*amount.*|.*value.*|.*price.*|.*fee.*': Decimal,
            r'.*date.*|.*deadline.*|.*due.*|.*maturity.*': date,
            r'.*percentage.*|.*rate.*|.*margin.*|.*\%.*': Decimal,
            r'.*currency.*': str,
            r'.*email.*': str,
            r'.*phone.*|.*fax.*': str,
            r'.*account.*number.*|.*swift.*|.*bic.*': str,
            r'.*name.*|.*title.*|.*description.*': str,
            r'.*code.*|.*id.*|.*reference.*': str,
            r'.*boolean.*|.*flag.*|.*is_.*|.*has_.*': bool,
            r'.*count.*|.*number.*|.*quantity.*': int,
        }
        
        self.section_patterns = {
            r'.*identification.*|.*basic.*|.*general.*': 'identification',
            r'.*financial.*|.*amount.*|.*money.*|.*currency.*': 'financial_details',
            r'.*date.*|.*time.*|.*deadline.*|.*maturity.*': 'key_dates',
            r'.*interest.*|.*rate.*|.*margin.*|.*libor.*': 'interest_terms',
            r'.*fee.*|.*cost.*|.*charge.*': 'fee_structure',
            r'.*payment.*|.*bank.*|.*account.*|.*swift.*': 'payment_instructions',
            r'.*special.*|.*condition.*|.*covenant.*|.*waiver.*': 'special_conditions'
        }

    def analyze_excel_template(self, excel_path: str, 
                             field_column: str = 'Field Name',
                             value_column: str = 'Value',
                             section_column: str = 'Section',
                             description_column: str = 'Description') -> List[FieldMetadata]:
        """
        Analyze Excel template to extract field metadata
        
        Expected Excel structure:
        | Section | Field Name | Value | Description | Type Hint | Required |
        """
        df = pd.read_excel(excel_path)
        fields = []
        
        for idx, row in df.iterrows():
            if pd.isna(row.get(field_column)):
                continue
                
            field_name = self._clean_field_name(str(row[field_column]))
            excel_column = row.get('Excel Column', f'Col_{idx}')
            section = row.get(section_column, self._infer_section(field_name))
            
            field_meta = FieldMetadata(field_name, excel_column, section)
            field_meta.description = str(row.get(description_column, ''))
            
            # Infer data type
            field_meta.data_type = self._infer_data_type(
                field_name, 
                row.get(value_column),
                row.get('Type Hint', '')
            )
            
            # Check if required
            field_meta.is_optional = not row.get('Required', False)
            
            # Set default value
            if 'Default' in row and not pd.isna(row['Default']):
                field_meta.default_value = row['Default']
            
            # Extract validation rules
            if 'Validation' in row and not pd.isna(row['Validation']):
                field_meta.validation_rules = str(row['Validation']).split(';')
            
            # Extract enum values if specified
            if 'Enum Values' in row and not pd.isna(row['Enum Values']):
                field_meta.enum_values = [v.strip() for v in str(row['Enum Values']).split(',')]
            
            fields.append(field_meta)
        
        return fields

    def _clean_field_name(self, name: str) -> str:
        """Convert Excel field name to Python-friendly name"""
        # Remove special characters and convert to snake_case
        name = re.sub(r'[^\w\s]', '', name)
        name = re.sub(r'\s+', '_', name.strip())
        return name.lower()

    def _infer_section(self, field_name: str) -> str:
        """Infer section based on field name patterns"""
        for pattern, section in self.section_patterns.items():
            if re.search(pattern, field_name.lower()):
                return section
        return 'general'

    def _infer_data_type(self, field_name: str, sample_value: Any, type_hint: str) -> Type:
        """Infer data type from field name, sample value, and hints"""
        
        # Check explicit type hint first
        if type_hint:
            type_mapping = {
                'str': str, 'string': str,
                'int': int, 'integer': int,
                'float': float, 'decimal': Decimal,
                'bool': bool, 'boolean': bool,
                'date': date, 'datetime': datetime
            }
            if type_hint.lower() in type_mapping:
                return type_mapping[type_hint.lower()]
        
        # Infer from sample value
        if not pd.isna(sample_value):
            if isinstance(sample_value, (int, np.integer)):
                return int
            elif isinstance(sample_value, (float, np.floating)):
                return Decimal if 'amount' in field_name.lower() or 'rate' in field_name.lower() else float
            elif isinstance(sample_value, bool):
                return bool
            elif isinstance(sample_value, (datetime, pd.Timestamp)):
                return datetime
            elif isinstance(sample_value, date):
                return date
        
        # Infer from field name patterns
        for pattern, data_type in self.field_type_patterns.items():
            if re.search(pattern, field_name.lower()):
                return data_type
        
        # Default to string
        return str


class DynamicModelGenerator:
    """Generates Pydantic models dynamically from field metadata"""
    
    def __init__(self):
        self.generated_models = {}
        self.enum_cache = {}

    def generate_models_from_fields(self, fields: List[FieldMetadata], 
                                  base_model_name: str = "DynamicContract") -> Type[BaseModel]:
        """Generate nested Pydantic models from field metadata"""
        
        # Group fields by section
        sections = self._group_fields_by_section(fields)
        
        # Generate section models first
        section_models = {}
        for section_name, section_fields in sections.items():
            if section_name != 'general':
                model_name = self._to_class_name(section_name)
                section_models[section_name] = self._create_model_for_fields(
                    section_fields, model_name
                )
        
        # Create main model with section models as fields
        main_model_fields = {}
        
        # Add section models as fields
        for section_name, model_class in section_models.items():
            field_name = section_name
            main_model_fields[field_name] = (model_class, Field(..., description=f"{section_name} details"))
        
        # Add general fields directly to main model
        if 'general' in sections:
            for field_meta in sections['general']:
                field_info = self._create_field_info(field_meta)
                main_model_fields[field_meta.name] = field_info
        
        # Create main model
        main_model = create_model(
            base_model_name,
            **main_model_fields,
            __config__=type('Config', (), {
                'extra': 'allow',
                'use_enum_values': True,
                'validate_assignment': True
            })
        )
        
        # Add utility methods
        main_model = self._add_utility_methods(main_model, fields)
        
        self.generated_models[base_model_name] = main_model
        return main_model

    def _group_fields_by_section(self, fields: List[FieldMetadata]) -> Dict[str, List[FieldMetadata]]:
        """Group fields by their section"""
        sections = {}
        for field in fields:
            section = field.section or 'general'
            if section not in sections:
                sections[section] = []
            sections[section].append(field)
        return sections

    def _create_model_for_fields(self, fields: List[FieldMetadata], model_name: str) -> Type[BaseModel]:
        """Create a Pydantic model for a group of fields"""
        model_fields = {}
        
        for field_meta in fields:
            field_info = self._create_field_info(field_meta)
            model_fields[field_meta.name] = field_info
        
        # Add validators if needed
        validators = self._create_validators(fields)
        if validators:
            model_fields.update(validators)
        
        return create_model(model_name, **model_fields)

    def _create_field_info(self, field_meta: FieldMetadata) -> tuple:
        """Create field info tuple for Pydantic model creation"""
        data_type = field_meta.data_type
        
        # Handle enums
        if field_meta.enum_values:
            enum_name = f"{field_meta.name.title()}Enum"
            if enum_name not in self.enum_cache:
                enum_values = {val.upper().replace(' ', '_'): val for val in field_meta.enum_values}
                self.enum_cache[enum_name] = Enum(enum_name, enum_values)
            data_type = self.enum_cache[enum_name]
        
        # Handle optional fields
        if field_meta.is_optional:
            data_type = Optional[data_type]
        
        # Create Field with metadata
        field_kwargs = {
            'description': field_meta.description,
            'alias': field_meta.excel_column
        }
        
        if field_meta.default_value is not None:
            field_kwargs['default'] = field_meta.default_value
        elif field_meta.is_optional:
            field_kwargs['default'] = None
        else:
            field_kwargs['default'] = ...
        
        field_instance = Field(**field_kwargs)
        
        return (data_type, field_instance)

    def _create_validators(self, fields: List[FieldMetadata]) -> Dict[str, Any]:
        """Create custom validators for fields"""
        validators = {}
        
        for field_meta in fields:
            validator_funcs = []
            
            # Add validation rules
            for rule in field_meta.validation_rules:
                if rule.startswith('min:'):
                    min_val = float(rule.split(':')[1])
                    validator_funcs.append(self._create_min_validator(field_meta.name, min_val))
                elif rule.startswith('max:'):
                    max_val = float(rule.split(':')[1])
                    validator_funcs.append(self._create_max_validator(field_meta.name, max_val))
                elif rule == 'positive':
                    validator_funcs.append(self._create_positive_validator(field_meta.name))
                elif rule == 'email':
                    validator_funcs.append(self._create_email_validator(field_meta.name))
            
            # Add type-specific validators
            if field_meta.data_type == Decimal:
                validator_funcs.append(self._create_decimal_validator(field_meta.name))
            
            if validator_funcs:
                # Combine all validators for this field
                def combined_validator(cls, v, field_name=field_meta.name, validators=validator_funcs):
                    for validator_func in validators:
                        v = validator_func(v)
                    return v
                
                validators[f'validate_{field_meta.name}'] = validator(field_meta.name, allow_reuse=True)(combined_validator)
        
        return validators

    def _create_min_validator(self, field_name: str, min_val: float):
        def validate_min(v):
            if v is not None and v < min_val:
                raise ValueError(f'{field_name} must be at least {min_val}')
            return v
        return validate_min

    def _create_max_validator(self, field_name: str, max_val: float):
        def validate_max(v):
            if v is not None and v > max_val:
                raise ValueError(f'{field_name} must be at most {max_val}')
            return v
        return validate_max

    def _create_positive_validator(self, field_name: str):
        def validate_positive(v):
            if v is not None and v < 0:
                raise ValueError(f'{field_name} must be positive')
            return v
        return validate_positive

    def _create_email_validator(self, field_name: str):
        def validate_email(v):
            if v and '@' not in str(v):
                raise ValueError(f'{field_name} must be a valid email')
            return v
        return validate_email

    def _create_decimal_validator(self, field_name: str):
        def validate_decimal(v):
            if v is not None:
                try:
                    return Decimal(str(v))
                except:
                    raise ValueError(f'{field_name} must be a valid decimal')
            return v
        return validate_decimal

    def _add_utility_methods(self, model_class: Type[BaseModel], fields: List[FieldMetadata]) -> Type[BaseModel]:
        """Add utility methods to the generated model"""
        
        def to_excel_dict(self) -> Dict[str, Any]:
            """Convert model to Excel-friendly dictionary"""
            excel_dict = {}
            for field_meta in fields:
                # Get value from nested structure
                value = self._get_nested_value(field_meta.name, field_meta.section)
                if value is not None:
                    excel_dict[field_meta.excel_column] = value
            return excel_dict
        
        def _get_nested_value(self, field_name: str, section: str):
            """Get value from potentially nested structure"""
            if section and section != 'general' and hasattr(self, section):
                section_obj = getattr(self, section)
                return getattr(section_obj, field_name, None)
            else:
                return getattr(self, field_name, None)
        
        @classmethod
        def from_excel_dict(cls, excel_dict: Dict[str, Any]):
            """Create model instance from Excel dictionary"""
            model_data = {}
            
            # Group data by section
            section_data = {}
            for field_meta in fields:
                value = excel_dict.get(field_meta.excel_column)
                if value is not None:
                    section = field_meta.section or 'general'
                    if section not in section_data:
                        section_data[section] = {}
                    section_data[section][field_meta.name] = value
            
            # Create nested structures
            for section, data in section_data.items():
                if section == 'general':
                    model_data.update(data)
                else:
                    model_data[section] = data
            
            return cls(**model_data)
        
        def get_field_mapping(self) -> Dict[str, str]:
            """Get mapping of model fields to Excel columns"""
            mapping = {}
            for field_meta in fields:
                key = field_meta.name
                if field_meta.section and field_meta.section != 'general':
                    key = f"{field_meta.section}.{field_meta.name}"
                mapping[key] = field_meta.excel_column
            return mapping
        
        # Add methods to the class
        model_class.to_excel_dict = to_excel_dict
        model_class._get_nested_value = _get_nested_value
        model_class.from_excel_dict = classmethod(from_excel_dict)
        model_class.get_field_mapping = get_field_mapping
        
        return model_class

    def _to_class_name(self, name: str) -> str:
        """Convert snake_case to PascalCase"""
        return ''.join(word.capitalize() for word in name.split('_'))


class DynamicModelPipeline:
    """Main pipeline for generating models from Excel templates"""
    
    def __init__(self):
        self.analyzer = ExcelTemplateAnalyzer()
        self.generator = DynamicModelGenerator()
        self.models_cache = {}

    def process_excel_template(self, template_path: str, 
                             model_name: str = None,
                             save_schema: bool = True) -> Type[BaseModel]:
        """Process Excel template and generate Pydantic model"""
        
        if model_name is None:
            model_name = Path(template_path).stem.replace(' ', '') + 'Model'
        
        # Check cache
        if model_name in self.models_cache:
            return self.models_cache[model_name]
        
        # Analyze template
        print(f"Analyzing Excel template: {template_path}")
        fields = self.analyzer.analyze_excel_template(template_path)
        
        print(f"Found {len(fields)} fields across {len(set(f.section for f in fields))} sections")
        
        # Generate model
        print(f"Generating Pydantic model: {model_name}")
        model_class = self.generator.generate_models_from_fields(fields, model_name)
        
        # Cache the model
        self.models_cache[model_name] = model_class
        
        # Save schema if requested
        if save_schema:
            self._save_model_schema(model_class, fields, model_name)
        
        return model_class

    def _save_model_schema(self, model_class: Type[BaseModel], 
                          fields: List[FieldMetadata], 
                          model_name: str):
        """Save model schema for reference"""
        schema = {
            'model_name': model_name,
            'pydantic_schema': model_class.schema(),
            'field_metadata': [f.to_dict() for f in fields],
            'generated_at': datetime.now().isoformat()
        }
        
        with open(f"{model_name}_schema.json", 'w') as f:
            json.dump(schema, f, indent=2, default=str)
        
        print(f"Schema saved to {model_name}_schema.json")

    def batch_process_templates(self, template_dir: str) -> Dict[str, Type[BaseModel]]:
        """Process multiple Excel templates in a directory"""
        template_path = Path(template_dir)
        models = {}
        
        for excel_file in template_path.glob("*.xlsx"):
            if not excel_file.name.startswith('~'):  # Skip temp files
                model_name = excel_file.stem.replace(' ', '') + 'Model'
                try:
                    model = self.process_excel_template(str(excel_file), model_name)
                    models[model_name] = model
                    print(f"✓ Successfully processed {excel_file.name}")
                except Exception as e:
                    print(f"✗ Error processing {excel_file.name}: {e}")
        
        return models

    def create_sample_template(self, output_path: str):
        """Create a sample Excel template showing the expected format"""
        sample_data = [
            {
                'Section': 'identification',
                'Field Name': 'Transaction Name',
                'Excel Column': 'B2',
                'Description': 'Name of the transaction',
                'Type Hint': 'str',
                'Required': True,
                'Default': None,
                'Validation': None,
                'Enum Values': None
            },
            {
                'Section': 'identification',
                'Field Name': 'Borrower',
                'Excel Column': 'B3',
                'Description': 'Primary borrower entity',
                'Type Hint': 'str',
                'Required': True,
                'Default': None,
                'Validation': None,
                'Enum Values': None
            },
            {
                'Section': 'financial_details',
                'Field Name': 'Initial Amount',
                'Excel Column': 'B5',
                'Description': 'Initial loan amount',
                'Type Hint': 'decimal',
                'Required': True,
                'Default': None,
                'Validation': 'positive',
                'Enum Values': None
            },
            {
                'Section': 'financial_details',
                'Field Name': 'Currency',
                'Excel Column': 'B6',
                'Description': 'Currency of the loan',
                'Type Hint': 'str',
                'Required': True,
                'Default': 'USD',
                'Validation': None,
                'Enum Values': 'USD, EUR, GBP, JPY'
            },
            {
                'Section': 'key_dates',
                'Field Name': 'Signing Date',
                'Excel Column': 'B8',
                'Description': 'Date when loan was signed',
                'Type Hint': 'date',
                'Required': True,
                'Default': None,
                'Validation': None,
                'Enum Values': None
            }
        ]
        
        df = pd.DataFrame(sample_data)
        df.to_excel(output_path, index=False)
        print(f"Sample template created at {output_path}")


# Example usage
if __name__ == "__main__":
    # Create pipeline
    pipeline = DynamicModelPipeline()
    
    # Create sample template
    #pipeline.create_sample_template("sample_loan_template.xlsx")
    
    # Example of processing a template (you would need an actual Excel file)
    model = pipeline.process_excel_template("Caledonia Facility A.xlsx", "LoanContract")
    
    # Example of using the generated model
    # contract_data = {"identification": {"transaction_name": "Test Loan"}}
    # contract = model(**contract_data)
    # excel_dict = contract.to_excel_dict()
    
    print("Dynamic model pipeline created successfully!")
    print("Use pipeline.create_sample_template() to create a template")
    print("Use pipeline.process_excel_template() to generate models")