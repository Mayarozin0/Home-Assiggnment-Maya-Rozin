import os
from pathlib import Path
from vector_store import VectorStore

# Initialize the vector store
BASE_DIR = Path(__file__).resolve().parent.parent
EMBEDDINGS_DIR = BASE_DIR / 'data' / 'embeddings'

# Singleton pattern for the vector store
_vector_store = None


def get_vector_store():
    """Get the vector store instance (initialize if needed)."""
    global _vector_store
    if _vector_store is None:
        try:
            _vector_store = VectorStore(EMBEDDINGS_DIR)
        except Exception as e:
            import logging
            logging.error(f"Error initializing vector store: {e}")
            raise
    return _vector_store


def search_knowledge_base(query, hmo, tier, top_k=6):
    """
    Search the knowledge base for relevant information.

    Args:
        query: The user's query
        hmo: The user's health fund
        tier: The user's insurance tier
        top_k: Number of results to return

    Returns:
        A dictionary with search results
    """
    try:
        # Map Hebrew values to English for filtering
        hmo_mapping = {
            "מכבי": "maccabi",
            "מאוחדת": "meuhedet",
            "כללית": "clalit"
        }

        tier_mapping = {
            "זהב": "gold",
            "כסף": "silver",
            "ארד": "bronze"
        }

        # Convert HMO and tier to English for filtering
        hmo_en = hmo_mapping.get(hmo, hmo.lower())
        tier_en = tier_mapping.get(tier, tier.lower())

        # Filter criteria
        filter_criteria = {
            "hmo": hmo_en,
            "tier": tier_en
        }

        # Get vector store instance
        vector_store = get_vector_store()

        # Search for relevant documents
        results = vector_store.search(query, top_k=top_k, filter_criteria=filter_criteria)

        # Format the results for the chatbot
        formatted_results = []
        for result in results:
            metadata = result.get('metadata', {})
            json_data = result.get('json_data', {})

            formatted_result = {
                "category": json_data.get('category', ''),
                "description": json_data.get('description', ''),
                "hmo": json_data.get('hmo', ''),
                "tier": json_data.get('tier', ''),
                "similarity": result.get('similarity', 0),
                "services": json_data.get('services', []),
                "contact": json_data.get('contact', {})
            }
            formatted_results.append(formatted_result)

        return {
            "results": formatted_results,
            "count": len(formatted_results),
            "query": query,
            "hmo": hmo,
            "tier": tier
        }
    except Exception as e:
        import logging
        logging.error(f"Error searching knowledge base: {e}")
        return {
            "results": [],
            "count": 0,
            "query": query,
            "hmo": hmo,
            "tier": tier,
            "error": str(e)
        }