from chatbot import StructuredChatbot, StructuredField


def test_basemodel_creation(): 
    # Create test fields
    fields = [
        StructuredField("name", "Person's name", "str"),
        StructuredField("age", "Person's age", "int")
    ]

    # Test the model creation
    chatbot = StructuredChatbot()
    model = chatbot._create_pydantic_model(fields)

    # Validate it works
    if model:
        instance = model(items=[{"name": "John", "age": 30}, {"name":"Jose", "age":48}, {"name":"Ana", "age":25}])
        print(type(instance))
        items = instance.items
        print(f"✅ Test passed - model created with {len(items)} elements and validated successfully")
        print(f"Data: {instance.items}")
    else:
        print("❌ Test failed - model creation returned None")


test_basemodel_creation()