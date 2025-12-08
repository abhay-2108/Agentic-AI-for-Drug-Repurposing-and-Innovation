from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, Dict, Any, List, Optional
import json
import re
import math
from dataclasses import dataclass

@dataclass
class PharmaceuticalDocument:
    """Represents a pharmaceutical document with metadata."""
    id: str
    title: str
    content: str
    data_source: str  # pubmed, clinical_trials, patents, market_data, internal
    compound_name: Optional[str] = None
    mechanism_of_action: Optional[str] = None
    therapeutic_indication: Optional[str] = None
    metadata: Dict[str, Any] = None

class SearchQuery(BaseModel):
    """Input schema for Pharmaceutical Semantic Search Tool."""
    query: str = Field(..., description="Text query for semantic search (compound name, mechanism of action, therapeutic indication, etc.)")
    documents: List[Dict[str, Any]] = Field(default=[], description="List of pharmaceutical documents to search. Each document should have 'id', 'title', 'content', 'data_source' fields and optional metadata.")
    data_sources: Optional[List[str]] = Field(default=None, description="Filter by data source types: pubmed, clinical_trials, patents, market_data, internal")
    top_k: int = Field(default=5, description="Number of top results to return")
    similarity_threshold: float = Field(default=0.0, description="Minimum similarity score threshold (0.0 to 1.0)")
    include_metadata: bool = Field(default=True, description="Whether to include document metadata in results")

class PharmaceuticalSemanticSearchTool(BaseTool):
    """Tool for semantic search across pharmaceutical text data using built-in text similarity."""

    name: str = "pharmaceutical_semantic_search"
    description: str = (
        "Performs in-memory semantic search across pharmaceutical text data using built-in text similarity algorithms. "
        "Supports searching by compound names, mechanisms of action, therapeutic indications, and other pharmaceutical concepts. "
        "Returns similarity-ranked results with confidence scores and metadata filtering capabilities."
    )
    args_schema: Type[BaseModel] = SearchQuery

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _normalize_text(self, text: str) -> str:
        """Normalize text for better comparison."""
        if not text:
            return ""
        # Convert to lowercase and remove extra whitespace
        text = text.lower().strip()
        # Remove special characters but keep alphanumeric and spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        return text

    def _get_word_frequency(self, text: str) -> Dict[str, int]:
        """Get word frequency dictionary from text."""
        words = self._normalize_text(text).split()
        freq = {}
        for word in words:
            if len(word) > 2:  # Ignore very short words
                freq[word] = freq.get(word, 0) + 1
        return freq

    def _compute_tf_idf_similarity(self, query: str, document: str) -> float:
        """Compute TF-IDF based similarity between query and document."""
        try:
            query_words = set(self._normalize_text(query).split())
            doc_words = set(self._normalize_text(document).split())
            
            if not query_words or not doc_words:
                return 0.0
            
            # Calculate Jaccard similarity as a simple approximation
            intersection = len(query_words.intersection(doc_words))
            union = len(query_words.union(doc_words))
            
            if union == 0:
                return 0.0
            
            jaccard_sim = intersection / union
            
            # Weight by query word coverage
            query_coverage = intersection / len(query_words) if query_words else 0
            
            # Combine Jaccard similarity with query coverage
            final_score = (jaccard_sim + query_coverage) / 2
            
            return min(final_score, 1.0)
            
        except Exception:
            return 0.0

    def _compute_cosine_similarity(self, query: str, document: str) -> float:
        """Compute cosine similarity using term frequency vectors."""
        try:
            query_freq = self._get_word_frequency(query)
            doc_freq = self._get_word_frequency(document)
            
            if not query_freq or not doc_freq:
                return 0.0
            
            # Get all unique words
            all_words = set(query_freq.keys()).union(set(doc_freq.keys()))
            
            if not all_words:
                return 0.0
            
            # Create frequency vectors
            query_vector = [query_freq.get(word, 0) for word in all_words]
            doc_vector = [doc_freq.get(word, 0) for word in all_words]
            
            # Calculate dot product
            dot_product = sum(q * d for q, d in zip(query_vector, doc_vector))
            
            # Calculate magnitudes
            query_magnitude = math.sqrt(sum(q * q for q in query_vector))
            doc_magnitude = math.sqrt(sum(d * d for d in doc_vector))
            
            if query_magnitude == 0 or doc_magnitude == 0:
                return 0.0
            
            # Calculate cosine similarity
            cosine_sim = dot_product / (query_magnitude * doc_magnitude)
            return max(0.0, min(cosine_sim, 1.0))
            
        except Exception:
            return 0.0

    def _compute_combined_similarity(self, query: str, document: str) -> float:
        """Compute combined similarity score using multiple methods."""
        try:
            # Get both similarity scores
            tf_idf_score = self._compute_tf_idf_similarity(query, document)
            cosine_score = self._compute_cosine_similarity(query, document)
            
            # Exact phrase matching bonus
            query_normalized = self._normalize_text(query)
            doc_normalized = self._normalize_text(document)
            phrase_bonus = 0.0
            
            if query_normalized in doc_normalized:
                phrase_bonus = 0.3
            elif any(word in doc_normalized for word in query_normalized.split() if len(word) > 3):
                phrase_bonus = 0.1
            
            # Combine scores with weights
            combined_score = (tf_idf_score * 0.4 + cosine_score * 0.4 + phrase_bonus * 0.2)
            
            return min(combined_score, 1.0)
            
        except Exception:
            return 0.0

    def _parse_documents(self, documents_data: List[Dict[str, Any]]) -> List[PharmaceuticalDocument]:
        """Parse document data into PharmaceuticalDocument objects."""
        documents = []
        for doc_data in documents_data:
            try:
                doc = PharmaceuticalDocument(
                    id=doc_data.get('id', ''),
                    title=doc_data.get('title', ''),
                    content=doc_data.get('content', ''),
                    data_source=doc_data.get('data_source', 'unknown'),
                    compound_name=doc_data.get('compound_name'),
                    mechanism_of_action=doc_data.get('mechanism_of_action'),
                    therapeutic_indication=doc_data.get('therapeutic_indication'),
                    metadata=doc_data.get('metadata', {})
                )
                documents.append(doc)
            except Exception:
                continue
        return documents

    def _filter_by_data_source(self, documents: List[PharmaceuticalDocument], 
                              data_sources: Optional[List[str]]) -> List[PharmaceuticalDocument]:
        """Filter documents by data source types."""
        if not data_sources:
            return documents
        
        valid_sources = {'pubmed', 'clinical_trials', 'patents', 'market_data', 'internal'}
        filtered_sources = [source for source in data_sources if source in valid_sources]
        
        return [doc for doc in documents if doc.data_source in filtered_sources]

    def _create_search_text(self, doc: PharmaceuticalDocument) -> str:
        """Create searchable text from document fields."""
        search_parts = [doc.title, doc.content]
        
        if doc.compound_name:
            search_parts.append(f"Compound: {doc.compound_name}")
        if doc.mechanism_of_action:
            search_parts.append(f"Mechanism: {doc.mechanism_of_action}")
        if doc.therapeutic_indication:
            search_parts.append(f"Indication: {doc.therapeutic_indication}")
            
        return " ".join(filter(None, search_parts))

    def _run(self, query: str, documents: List[Dict[str, Any]], 
             data_sources: Optional[List[str]] = None, top_k: int = 5, 
             similarity_threshold: float = 0.0, include_metadata: bool = True) -> str:
        """Execute pharmaceutical semantic search."""
        
        try:
            # Validate inputs
            if not query.strip():
                return json.dumps({"error": "Query cannot be empty", "results": []})
            
            if not documents:
                # Return empty results instead of error if no documents provided
                return json.dumps({
                    "query": query,
                    "results": [],
                    "message": "No documents provided for search."
                })
            
            # Parse and filter documents
            parsed_docs = self._parse_documents(documents)
            if not parsed_docs:
                return json.dumps({"error": "No valid documents found", "results": []})
            
            filtered_docs = self._filter_by_data_source(parsed_docs, data_sources)
            if not filtered_docs:
                return json.dumps({"error": "No documents match the specified data sources", "results": []})
            
            # Prepare document texts and compute similarities
            results = []
            for doc in filtered_docs:
                search_text = self._create_search_text(doc)
                similarity = self._compute_combined_similarity(query, search_text)
                
                if similarity >= similarity_threshold:
                    result = {
                        "id": doc.id,
                        "title": doc.title,
                        "content": doc.content[:500] + "..." if len(doc.content) > 500 else doc.content,
                        "data_source": doc.data_source,
                        "similarity_score": float(similarity),
                        "rank": 0  # Will be updated after sorting
                    }
                    
                    if include_metadata:
                        result["compound_name"] = doc.compound_name
                        result["mechanism_of_action"] = doc.mechanism_of_action
                        result["therapeutic_indication"] = doc.therapeutic_indication
                        result["metadata"] = doc.metadata or {}
                    
                    results.append(result)
            
            # Sort by similarity score (descending) and limit to top_k
            results.sort(key=lambda x: x["similarity_score"], reverse=True)
            results = results[:top_k]
            
            # Update ranks after sorting
            for i, result in enumerate(results):
                result["rank"] = i + 1
            
            # Prepare response
            response = {
                "query": query,
                "total_documents_searched": len(filtered_docs),
                "results_found": len(results),
                "top_k": top_k,
                "similarity_threshold": similarity_threshold,
                "data_sources_used": data_sources or ["all"],
                "results": results
            }
            
            return json.dumps(response, indent=2)
            
        except Exception as e:
            error_msg = f"Search failed: {str(e)}"
            return json.dumps({"error": error_msg, "results": []})