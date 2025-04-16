import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from openai import AzureOpenAI
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional

# Load environment variables
load_dotenv()

# Azure OpenAI settings
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")


class VectorStore:
    def __init__(self, embeddings_dir=None):
        """Initialize the vector store."""
        self.metadata_df = None
        self.embeddings = None
        self.json_data_dir = None

        # Initialize Azure OpenAI client
        self.client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            api_version="2023-05-15",
            azure_endpoint=AZURE_OPENAI_ENDPOINT
        )

        if embeddings_dir:
            self.load_from_directory(embeddings_dir)

    def load_from_directory(self, embeddings_dir):
        """Load embeddings and metadata from a directory."""
        # Define paths
        metadata_csv = os.path.join(embeddings_dir, 'embeddings_metadata.csv')
        embeddings_npy = os.path.join(embeddings_dir, 'embeddings.npy')
        json_data_dir = os.path.join(embeddings_dir, 'json_data')

        # Load metadata
        if os.path.exists(metadata_csv):
            self.metadata_df = pd.read_csv(metadata_csv)
            print(f"Loaded metadata for {len(self.metadata_df)} documents")
        else:
            raise FileNotFoundError(f"Metadata file not found: {metadata_csv}")

        # Load embeddings
        if os.path.exists(embeddings_npy):
            self.embeddings = np.load(embeddings_npy)
            print(f"Loaded {len(self.embeddings)} embeddings")
        else:
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_npy}")

        # Set JSON data directory
        if os.path.exists(json_data_dir):
            self.json_data_dir = json_data_dir
        else:
            print(f"Warning: JSON data directory not found: {json_data_dir}")

        # Validate lengths
        if len(self.metadata_df) != len(self.embeddings):
            raise ValueError(
                f"Mismatch between metadata ({len(self.metadata_df)} records) and embeddings ({len(self.embeddings)} vectors)")

    def create_query_embedding(self, query_text):
        """Create an embedding for a query text."""
        try:
            response = self.client.embeddings.create(
                input=query_text,
                model=EMBEDDING_DEPLOYMENT
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error creating query embedding: {e}")
            return None

    def _cosine_similarity(self, a, b):
        """Calculate cosine similarity between two vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def search(self, query_text, top_k=5, filter_criteria=None):
        """
        Search for similar documents to the query text.

        Args:
            query_text (str): The query text to search for.
            top_k (int): The number of top results to return.
            filter_criteria (dict): Metadata filters to apply before similarity search.
                                    Example: {'hmo': 'maccabi', 'tier': 'gold'}

        Returns:
            List of dictionaries containing search results.
        """
        # Create query embedding
        query_embedding = self.create_query_embedding(query_text)
        if query_embedding is None:
            return []

        # Apply metadata filters if provided
        filtered_indices = None
        if filter_criteria:
            # Start with all indices
            filtered_indices = np.arange(len(self.metadata_df))

            # Apply each filter
            for key, value in filter_criteria.items():
                if key in self.metadata_df.columns:
                    # Find indices that match this criteria
                    key_indices = self.metadata_df[self.metadata_df[key] == value].index.values
                    # Intersection with existing filtered indices
                    filtered_indices = np.intersect1d(filtered_indices, key_indices)

        # If no filtered indices left or no filters applied, use all indices
        if filtered_indices is None or len(filtered_indices) == 0:
            if filter_criteria:
                print("Warning: No documents match the filter criteria")
                return []
            filtered_indices = np.arange(len(self.metadata_df))

        # Calculate similarities only for filtered documents
        similarities = []
        for idx in filtered_indices:
            similarity = self._cosine_similarity(query_embedding, self.embeddings[idx])
            similarities.append((idx, similarity))

        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Get top k results
        top_indices = [idx for idx, _ in similarities[:top_k]]

        # Build result list
        results = []
        for i, idx in enumerate(top_indices):
            # Get metadata
            metadata = self.metadata_df.iloc[idx].to_dict()

            # Load full JSON data if available
            json_data = None
            if self.json_data_dir:
                json_file = os.path.join(self.json_data_dir, f"{metadata['id']}.json")
                if os.path.exists(json_file):
                    with open(json_file, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)

            # Add to results
            result = {
                'metadata': metadata,
                'similarity': similarities[filtered_indices.tolist().index(idx)][1],
                'json_data': json_data
            }
            results.append(result)

        return results

    def get_document_by_id(self, doc_id):
        """Get a document by its ID."""
        if self.metadata_df is None:
            return None

        if doc_id in self.metadata_df['id'].values:
            # Get metadata
            metadata = self.metadata_df[self.metadata_df['id'] == doc_id].iloc[0].to_dict()

            # Load full JSON data if available
            json_data = None
            if self.json_data_dir:
                json_file = os.path.join(self.json_data_dir, f"{doc_id}.json")
                if os.path.exists(json_file):
                    with open(json_file, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)

            return {
                'metadata': metadata,
                'json_data': json_data
            }

        return None

    def filter_by_metadata(self, criteria):
        """
        Get all documents that match the given metadata criteria.

        Args:
            criteria (dict): Metadata filters to apply.
                             Example: {'hmo': 'maccabi', 'tier': 'gold'}

        Returns:
            List of matching document IDs.
        """
        if self.metadata_df is None:
            return []

        filtered_df = self.metadata_df.copy()

        for key, value in criteria.items():
            if key in filtered_df.columns:
                filtered_df = filtered_df[filtered_df[key] == value]

        return filtered_df['id'].tolist()

# TODO: delete
# Example usage when run as a script
# if __name__ == "__main__":
#     # Define paths
#     BASE_DIR = Path(__file__).resolve().parent.parent
#     EMBEDDINGS_DIR = BASE_DIR / 'data' / 'embeddings'
#
#     print(f"Loading embeddings from: {EMBEDDINGS_DIR}")
#
#     # Load vector store
#     try:
#         vector_store = VectorStore(EMBEDDINGS_DIR)
#
#         # Example search with filter
#         query = "כמה עולה שיאצו?"
#         filter_criteria = {
#             'hmo': 'maccabi',
#             'tier': 'gold'
#         }
#
#         results = vector_store.search(query, top_k=10, filter_criteria=filter_criteria)
#
#         # Print results
#         print(f"\nSearch results for: '{query}' with filter {filter_criteria}:")
#         for i, result in enumerate(results):
#             print(f"\nResult {i + 1}:")
#             print(f"  ID: {result['metadata']['id']}")
#             print(f"  Service: {result['metadata']['service']}")
#             print(f"  Similarity: {result['similarity']:.4f}")
#
#             # Print snippet of text
#             text = result['metadata']['text']
#             snippet = text[:200] + "..." if len(text) > 200 else text
#             print(f"  Text snippet: {snippet}")
#     except Exception as e:
#         print(f"Error loading vector store: {e}")