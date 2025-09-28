import weaviate
import os
import asyncio
import uuid
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from weaviate.classes.init import Auth
from weaviate.classes.config import (
    Configure,
    DataType,
    Property,
    ReplicationDeletionStrategy,
    VectorDistances,
    VectorFilterStrategy,
    Tokenization,
    Reconfigure,
)
from weaviate.classes.tenants import Tenant
from mistralai import Mistral 


class ContractClause(BaseModel):
    contract_id: str = Field(..., description="Unique identifier for the contract")
    contract_name: str = Field(..., description="Name of the contract")
    clause_name: str = Field(..., description="Name of the clause")
    clause_text: str = Field(..., description="Text content of the clause")

class Contract(BaseModel):
    clauses : list[ContractClause] = Field(..., description="List of clauses in the contract")



class WeaviateAsyncManager:
    def __init__(self):
        load_dotenv()  # Load environment variables from .env file
        # Best practice: store your credentials in environment variables
        weaviate_url = os.environ["WEAVIATE_URL"]
        weaviate_api_key = os.environ["WEAVIATE_API_KEY"]
        openai_api_key = os.environ["OPENAI_API_KEY"]

        # Create async client (but don't connect yet)
        self.async_client = weaviate.use_async_with_weaviate_cloud(
            cluster_url=weaviate_url,
            auth_credentials=Auth.api_key(weaviate_api_key),
            headers={"X-OpenAI-Api-Key": openai_api_key}
        )

    async def __aenter__(self):
        """Async context manager entry"""
        await self.async_client.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - ensures proper cleanup"""
        await self.close()

    async def connect(self):
        """Manually connect to Weaviate"""
        await self.async_client.connect()

    async def close(self):
        """Properly close the async connection"""
        if hasattr(self, 'async_client') and self.async_client:
            await self.async_client.close()

    async def create_weaviate_collection(self):
        try: 
            await self.async_client.collections.create(
                name="ContractClausesWithTenancy", ## Implies that for each user we create a new shard
                description="A collection of contracts and their clauses", 
                properties=[
                    Property(name="contract_id", data_type=DataType.TEXT, description="Unique identifier for the contract", tokenization=Tokenization.WORD, indexSearchable=True),
                    Property(name="contract_name", data_type=DataType.TEXT, description="Name of the contract", tokenization=Tokenization.WORD, indexSearchable=True),
                    Property(name="clause_name", data_type=DataType.TEXT, description="Name of the clause", tokenization=Tokenization.WORD, indexSearchable=True),
                    Property(name="clause_text", data_type=DataType.TEXT, description="Text content of the clause", tokenization=Tokenization.WORD, indexSearchable=True),
                ],
                multi_tenancy_config=Configure.multi_tenancy(enabled=True), 
                
                vector_config=Configure.Vectors.text2vec_openai(
                    name="clause_text_vector",
                    model="text-embedding-3-large",
                    dimensions=1024,
                    source_properties=["clause_name", "clause_text"],
                    vector_index_config=Configure.VectorIndex.dynamic(
                        distance_metric=VectorDistances.COSINE,   # Default and robust for most text models
                        threshold=10000,                          # Switch to HNSW after 10,000 objects
                        flat=Configure.VectorIndex.flat(
                            quantizer=Configure.VectorIndex.Quantizer.bq()  # Optional: use BQ for faster flat search
                        ),
                        hnsw=Configure.VectorIndex.hnsw(
                            max_connections=32,                   # Example HNSW config, adjust as needed
                            quantizer=Configure.VectorIndex.Quantizer.rq()  # Optional: use RQ for HNSW compression
                        )
                    )
                ),
                replication_config=Configure.replication(
                    factor=1,
                    async_enabled=True,
                    deletion_strategy=ReplicationDeletionStrategy.TIME_BASED_RESOLUTION,
                ),
                reranker_config=Configure.Reranker.cohere(),
                generative_config=Configure.Generative.openai(
                    model="gpt-5-mini-2025-08-07"
                )
            )
            print("Async collection created successfully!")
        except Exception as e:
            print(f"Error creating collection: {e}")
        
        response = await self.async_client.collections.list_all(simple=False)
        return response

    async def delete_weaviate_collection(self, collection_name: str):
        try: 
            await self.async_client.collections.delete(collection_name)
            print(f"Collection '{collection_name}' deleted successfully!")
        except Exception as e:
            print(f"Error deleting collection: {e}")
        response = await self.async_client.collections.list_all(simple=False)
        return response

    async def create_object(self, collection_name: str, clause: ContractClause, tenant: str = None):
        try: 
            if tenant:
                print(f"Inserting object for tenant: {tenant}")
                collection = self.async_client.collections.use(collection_name).with_tenant(tenant)
            else:
                collection = self.async_client.collections.use(collection_name)
            
            uuid = await collection.data.insert({
                    "contract_id": clause.contract_id,
                    "contract_name": clause.contract_name,
                    "clause_name": clause.clause_name,
                    "clause_text": clause.clause_text,
                })
            print(f"Async object created successfully with UUID: {uuid}")
        except Exception as e:
            print(f"Error creating object: {e}")
            return None
        return uuid

    async def create_batch_objects(self, collection_name: str, contract: Contract, tenant: str = None, batch_size: int = 100):
        try: 
            if tenant:
                print(f"Inserting objects for tenant: {tenant}")
                collection = self.async_client.collections.use(collection_name).with_tenant(tenant)
            else:
                collection = self.async_client.collections.use(collection_name)

            # For async, we use insert_many for batch operations
            objects_to_insert = []
            for clause in contract.clauses:
                objects_to_insert.append({
                    "contract_id": clause.contract_id,
                    "contract_name": clause.contract_name,
                    "clause_name": clause.clause_name,
                    "clause_text": clause.clause_text,
                })
            
            response = await collection.data.insert_many(objects_to_insert)
            print(f"Async batch import successful! Imported {len(objects_to_insert)} articles.")
                
        except Exception as e:
            print(f"Error creating batch of objects: {e}")
            return None
        return response

    async def read_all_objects(self, collection_name: str):
        try: 
            collection = self.async_client.collections.use(collection_name)
            
            # For async, we fetch all objects at once
            response = await collection.query.fetch_objects()
            objects = []
            
            for item in response.objects:
                objects.append(item.properties)
                print(f"Contract ID: {item.properties.get('contract_id')}")
                print(f"Contract Name: {item.properties.get('contract_name')}")
                print(f"Clause Name: {item.properties.get('clause_name')}")
                print(f"Clause Text: {item.properties.get('clause_text')}")
                print("---")
            
            print(f"Total objects retrieved: {len(objects)}")
        except Exception as e:
            print(f"Error reading objects: {e}")
            return None
        return objects

    async def async_query(self, collection_name:str, query:str, limit:int, tenant: str = None):
        try:
            if tenant:
                print(f"Querying collection for tenant: {tenant}")
                collection=self.async_client.collections.use(name=collection_name).with_tenant(tenant)
            else:   
                print("Querying collection without tenant")
                collection=self.async_client.collections.use(name=collection_name)
            response = await collection.query.hybrid(query=query, limit=limit)
        except Exception as e:
            print(f"Error querying collection: {e}")
            return None
        response = response.objects
        results = []
        for item in response:
            results.append(item.properties)

        return results
    
    async def update_auto_tenant_generation(self, collection_name: str, enabled: bool):
        try:
            collection = self.async_client.collections.use(collection_name)
            await collection.config.update(
                multi_tenancy_config=Reconfigure.multi_tenancy(auto_tenant_creation=enabled)
            )
            print(f"Auto tenant generation set to {enabled} for collection '{collection_name}'")
        except Exception as e:
            print(f"Error updating auto tenant generation: {e}")
            return None
        response = await self.async_client.collections.list_all(simple=False)
        return response

    async def create_tenants(self, collection_name: str, tenants: list[Tenant]):
        try:
            collection = self.async_client.collections.use(collection_name)
            await collection.tenants.create(tenants)
            for tenant in tenants:
                print(f"Tenant '{tenant.name}' created in collection '{collection_name}'")
        except Exception as e:
            print(f"Error creating tenants: {e}")
            return None
        response = await collection.tenants.get()
        return response

    async def load_chunks_from_folder(self, folder_path: str, collection_name: str, tenant: str, contract_name: str):
        """
        Load all chunk files from a folder and create batch objects in Weaviate

        Args:
            folder_path: Path to the folder containing chunk files
            collection_name: Name of the Weaviate collection
            tenant: Tenant identifier (e.g., "tenantA")
            contract_name: Name of the contract (e.g., "Sfacilities")
        """
        try:
            # Generate a random contract ID
            contract_id = str(uuid.uuid4())
            print(f"Generated contract ID: {contract_id}")

            # Get all .md files from the folder
            if not os.path.exists(folder_path):
                print(f"Error: Folder path '{folder_path}' does not exist")
                return None

            chunk_files = [f for f in os.listdir(folder_path) if f.endswith('.md')]
            print(f"Found {len(chunk_files)} chunk files")

            # Create ContractClause objects for each chunk
            clauses = []
            for chunk_file in chunk_files:
                file_path = os.path.join(folder_path, chunk_file)

                # Read the chunk content
                with open(file_path, 'r', encoding='utf-8') as file:
                    chunk_content = file.read()

                # Extract clause name from filename (remove .md extension)
                clause_name = os.path.splitext(chunk_file)[0]

                # Create ContractClause object
                clause = ContractClause(
                    contract_id=contract_id,
                    contract_name=contract_name,
                    clause_name=clause_name,
                    clause_text=chunk_content
                )
                clauses.append(clause)
                print(f"Loaded chunk: {clause_name}")

            # Create Contract object with all clauses
            contract = Contract(clauses=clauses)

            # Use create_batch_objects to insert all chunks
            response = await self.create_batch_objects(
                collection_name=collection_name,
                contract=contract,
                tenant=tenant
            )

            print(f"Successfully loaded {len(clauses)} chunks for contract '{contract_name}' with ID '{contract_id}' for tenant '{tenant}'")
            return response

        except Exception as e:
            print(f"Error loading chunks from folder: {e}")
            return None



# Example usage:
def main():
    async def run():
        async with WeaviateAsyncManager() as manager:
            query = "What is the name of the company that is getting the facility?"
            results = await manager.async_query(collection_name="ContractClausesWithTenancy", query=query, limit=2, tenant="tenantA")
            print("Query Results:")
            print(results)

    asyncio.run(run())

if __name__ == "__main__":
    main()