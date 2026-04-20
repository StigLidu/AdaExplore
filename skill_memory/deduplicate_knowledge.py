#!/usr/bin/env python3
"""
Script to deduplicate a knowledge base file where each line is a knowledge item.
Uses embeddings (sentence-transformers) or LLM judgment to detect semantically similar duplicates.
Preserves the order of first occurrence of each unique line.
"""

import argparse
import sys
import os
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from agent.inference_server import create_inference_server, query_inference_server
from tqdm import tqdm

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_embeddings_sentence_transformers(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """Get embeddings using sentence-transformers."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers not installed. Install with: pip install sentence-transformers"
        )
    
    print(f"Loading sentence-transformers model: {model_name}")
    model = SentenceTransformer(model_name)
    
    print("Computing embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings


def llm_judge_duplicate_batch(
    candidate_text: str, 
    existing_texts: List[str], 
    existing_unique_indices: List[int],
    server_type: str = "azure", 
    model_name: str = "gpt-5-mini",
    existing_server: callable = None
) -> Optional[Tuple[int, float]]:
    """
    Use LLM to judge if a candidate knowledge item is a duplicate of any existing items.
    Compares the candidate against all existing items in a single LLM call.
    
    Args:
        candidate_text: The new knowledge item to check
        existing_texts: List of existing unique knowledge items
        existing_unique_indices: List of indices in unique_indices corresponding to existing_texts
        server_type: LLM server type
        model_name: LLM model name
    
    Returns:
        None if no duplicate found, or (matched_unique_index, confidence) tuple if duplicate found
        matched_unique_index is the index in existing_unique_indices (0-based)
    """
    if not existing_texts:
        return None
    
    # Use agent.inference_server for LLM queries
    try:
        if existing_server is None:
            server = create_inference_server(server_type)
        else:
            server = existing_server
    except Exception as e:
        raise ImportError(f"Cannot create inference server: {e}")
    
    # Build prompt with all existing items
    existing_items_text = "\n\n".join([
        f"Existing Item {idx + 1}:\n{text}" 
        for idx, text in enumerate(existing_texts)
    ])
    
    prompt = f"""You are an expert at identifying duplicate knowledge items. Your task is to determine if a new knowledge statement is a duplicate of any existing knowledge statements, even if worded differently.

Compare the NEW item against ALL existing items. Is the new item a duplicate of any existing item?

Important:
- matched_index should be the 1-based index (1, 2, 3, ...) of the existing item that matches, or null if no match
- Only mark as duplicate if they convey essentially the same information
- If multiple items seem similar, choose the one with the highest similarity
- Be strict: only mark as duplicate if the core meaning is identical

EXISTING Knowledge Items:
{existing_items_text}

NEW Knowledge Item to Check:
{candidate_text}

Respond with ONLY a JSON object in this exact format:
{{"is_duplicate": true/false, "matched_index": <1-based index of matched item or null>, "confidence": 0.0-1.0, "reason": "brief explanation"}}
"""

    try:
        response = query_inference_server(
            server=server,
            model_name=model_name,
            prompt=prompt,
            max_completion_tokens=1000,  # Increased for batch comparison
            temperature=1.0
        )
    except Exception as e:
        print(f"Warning: LLM batch query failed: {e}. Treating as non-duplicate.")
        return None
    
    # Parse response
    try:
        import json
        import re
        # Extract JSON from response
        json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            is_dup = result.get("is_duplicate", False)
            matched_idx = result.get("matched_index")
            confidence = result.get("confidence", 0.5)
            
            if is_dup and matched_idx is not None:
                # Convert 1-based index to 0-based index in existing_unique_indices
                if 1 <= matched_idx <= len(existing_unique_indices):
                    # Return the index in existing_unique_indices (0-based)
                    return matched_idx - 1, float(confidence)
            
            return None
        else:
            # Fallback: try to extract index from response
            response_lower = response.lower()
            if "true" in response_lower and "duplicate" in response_lower:
                # Try to find a number in the response
                numbers = re.findall(r'\d+', response)
                if numbers:
                    matched_idx = int(numbers[0])
                    if 1 <= matched_idx <= len(existing_unique_indices):
                        return matched_idx - 1, 0.7
            return None
    except Exception as e:
        print(f"Warning: Failed to parse LLM batch response: {e}. Treating as non-duplicate.")
        return None


def deduplicate_knowledge(
    input_file: str,
    output_file: str = None,
    in_place: bool = False,
    method: str = "exact",
    similarity_threshold: float = 0.95,
    embedding_model: str = "all-MiniLM-L6-v2",
    llm_server_type: str = "azure",
    llm_model: str = "gpt-5-mini",
    hybrid_mode: bool = False,
    hybrid_threshold_low: float = 0.70,
    hybrid_threshold_high: float = 0.98
):
    """
    Remove duplicate lines from a knowledge base file using exact matching, semantic similarity, or LLM judgment.
    
    Args:
        input_file: Path to the input knowledge base file
        output_file: Path to the output file (if None and not in_place, appends '_deduplicated' to input filename)
        in_place: If True, overwrites the input file with deduplicated content
        method: Deduplication method - 'exact', 'semantic', or 'llm'
        similarity_threshold: Cosine similarity threshold for semantic deduplication (0.0-1.0)
        embedding_model: Model name for sentence-transformers (e.g., 'all-MiniLM-L6-v2', 'all-mpnet-base-v2')
        llm_server_type: Server type for LLM ('azure', 'openai', etc.)
        llm_model: Model name for LLM (e.g., 'gpt-4o-mini', 'claude-3-haiku')
        hybrid_mode: If True, use embedding for fast filtering, then LLM for edge cases
        hybrid_threshold_low: Lower threshold for hybrid mode (below this = definitely not duplicate)
        hybrid_threshold_high: Upper threshold for hybrid mode (above this = definitely duplicate, between = ask LLM)
    """
    input_path = Path(input_file)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Read all lines
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    original_count = len(lines)
    
    # Filter out empty lines and prepare texts
    non_empty_lines = []
    line_indices = []
    texts = []
    
    for i, line in enumerate(lines):
        line_stripped = line.rstrip('\n\r')
        line_normalized = line_stripped.strip()
        if line_normalized:
            non_empty_lines.append(line)
            line_indices.append(i)
            texts.append(line_normalized)
    
    if method == "exact":
        # Exact string matching
        seen = set()
        unique_indices = []
        duplicates = []
        
        for idx, text in enumerate(texts):
            if text not in seen:
                seen.add(text)
                unique_indices.append(line_indices[idx])
            else:
                duplicates.append((line_indices[idx] + 1, text))
        
        # Reconstruct unique lines in original order
        unique_lines = []
        unique_indices_set = set(unique_indices)
        for i, line in enumerate(lines):
            if i in unique_indices_set or not line.strip():
                unique_lines.append(line)

    elif method == "llm":
        # Pure LLM-based deduplication (batch mode for efficiency)
        print(f"Using LLM ({llm_server_type}/{llm_model}) for deduplication")
        print("Using batch mode: comparing each new item against all existing items in one call.")
        
        unique_indices = []
        unique_texts = []  # Store texts for batch comparison
        duplicates = []
        llm_calls = 0
        
        for i in tqdm(range(len(texts)), desc="Deduplicating with LLM (batch)"):
            # Get existing unique texts and their indices
            existing_texts = [texts[j] for j in unique_indices]
            existing_line_indices = [line_indices[j] for j in unique_indices]
            
            if existing_texts:
                # Batch compare: one LLM call for all existing items
                llm_calls += 1
                result = llm_judge_duplicate_batch(
                    texts[i], 
                    existing_texts, 
                    list(range(len(unique_indices))),  # Pass indices 0, 1, 2, ... for mapping
                    llm_server_type, 
                    llm_model
                )
                
                if result is not None:
                    # Found a duplicate
                    matched_unique_idx, confidence = result
                    print(f"{texts[i]} is a duplicate of {texts[matched_unique_idx]} with confidence {confidence}")
                    # matched_unique_idx is the index in unique_indices
                    matched_text_idx = unique_indices[matched_unique_idx]
                    matched_line_idx = line_indices[matched_text_idx]
                    duplicates.append((
                        line_indices[i] + 1, texts[i], 
                        matched_line_idx + 1, texts[matched_text_idx], 
                        confidence
                    ))
                else:
                    # No duplicate found, add to unique
                    unique_indices.append(i)
                    unique_texts.append(texts[i])
            else:
                # First item, always unique
                unique_indices.append(i)
                unique_texts.append(texts[i])
        
        print(f"Total LLM calls made: {llm_calls} (much fewer than {len(texts) * (len(texts) - 1) // 2} pairwise comparisons!)")
        
        # Reconstruct unique lines in original order
        unique_lines = []
        unique_indices_set = {line_indices[idx] for idx in unique_indices}
        for i, line in enumerate(lines):
            if i in unique_indices_set or not line.strip():
                unique_lines.append(line)
    
    else:  # semantic or hybrid
        # Semantic similarity using embeddings (with optional LLM refinement)
        if hybrid_mode:
            print(f"Using HYBRID mode: embeddings for fast filtering + LLM for edge cases")
            print(f"  Embedding similarity < {hybrid_threshold_low}: definitely not duplicate")
            print(f"  Embedding similarity > {hybrid_threshold_high}: definitely duplicate")
            print(f"  Embedding similarity between {hybrid_threshold_low}-{hybrid_threshold_high}: ask LLM")
        else:
            print(f"Using sentence-transformers embeddings for semantic deduplication")
        
        embeddings = get_embeddings_sentence_transformers(texts, embedding_model)
        
        # Normalize embeddings for cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        print(f"Finding duplicates...")
        unique_indices = []
        unique_texts = []  # Store texts for batch LLM comparison
        duplicates = []
        llm_calls = 0
        
        for i in tqdm(range(len(texts)), desc="Deduplicating"):
            is_duplicate = False
            current_embedding = embeddings[i]
            
            if hybrid_mode:
                # Hybrid mode: collect candidates for LLM batch comparison
                llm_candidates = []  # List of (j, similarity) pairs that need LLM judgment
                high_similarity_match = None  # Track highest similarity above threshold
                
                for j in unique_indices:
                    similarity = cosine_similarity(current_embedding, embeddings[j])
                    
                    if similarity < hybrid_threshold_low:
                        # Definitely not duplicate, skip
                        continue
                    elif similarity >= hybrid_threshold_high:
                        # Definitely duplicate based on embedding
                        is_duplicate = True
                        duplicates.append((line_indices[i] + 1, texts[i], line_indices[j] + 1, texts[j], similarity))
                        break
                    else:
                        # Edge case: collect for batch LLM judgment
                        llm_candidates.append((j, similarity))
                
                # If no high similarity match found, use LLM batch comparison for edge cases
                if not is_duplicate and llm_candidates:
                    # Get texts of candidates
                    candidate_texts = [texts[j] for j, _ in llm_candidates]
                    candidate_indices = list(range(len(llm_candidates)))
                    
                    llm_calls += 1
                    result = llm_judge_duplicate_batch(
                        texts[i],
                        candidate_texts,
                        candidate_indices,
                        llm_server_type,
                        llm_model
                    )
                    
                    if result is not None:
                        matched_candidate_idx, confidence = result
                        matched_j, _ = llm_candidates[matched_candidate_idx]
                        is_duplicate = True
                        duplicates.append((line_indices[i] + 1, texts[i], line_indices[matched_j] + 1, texts[matched_j], confidence))
            else:
                # Pure embedding mode
                for j in unique_indices:
                    similarity = cosine_similarity(current_embedding, embeddings[j])
                    if similarity >= similarity_threshold:
                        is_duplicate = True
                        duplicates.append((line_indices[i] + 1, texts[i], line_indices[j] + 1, texts[j], similarity))
                        break
            
            if not is_duplicate:
                unique_indices.append(i)
                unique_texts.append(texts[i])
        
        if hybrid_mode:
            print(f"Total LLM calls made (for edge cases): {llm_calls}")
        
        # Reconstruct unique lines in original order
        unique_lines = []
        unique_indices_set = {line_indices[idx] for idx in unique_indices}
        for i, line in enumerate(lines):
            if i in unique_indices_set or not line.strip():
                unique_lines.append(line)
    
    unique_count = len(unique_lines)
    removed_count = original_count - unique_count
    
    # Determine output file
    if in_place:
        output_path = input_path
    elif output_file:
        output_path = Path(output_file)
    else:
        if method == "llm":
            suffix = "_llm_dedup"
        elif hybrid_mode:
            suffix = "_hybrid_dedup"
        elif method == "semantic":
            suffix = "_semantic_dedup"
        else:
            suffix = "_deduplicated"
        output_path = input_path.parent / f"{input_path.stem}{suffix}{input_path.suffix}"
    
    # Write deduplicated content
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(unique_lines)
    
    # Print statistics
    print(f"\n{'='*60}")
    print(f"Original lines: {original_count}")
    print(f"Unique lines: {unique_count}")
    print(f"Removed duplicates: {removed_count}")
    print(f"Output written to: {output_path}")
    print(f"{'='*60}")
    
    if duplicates:
        print(f"\nFirst 10 duplicate pairs (showing line numbers from original file):")
        for dup_info in duplicates[:10]:
            if method == "exact":
                line_num, content = dup_info
                print(f"  Line {line_num}: {content[:80]}..." if len(content) > 80 else f"  Line {line_num}: {content}")
            else:
                line_num1, content1, line_num2, content2, similarity = dup_info
                print(f"  Line {line_num1} (similarity: {similarity:.3f}) vs Line {line_num2}:")
                print(f"    '{content1[:60]}...'")
                print(f"    '{content2[:60]}...'")
        if len(duplicates) > 10:
            print(f"  ... and {len(duplicates) - 10} more duplicates")


def main():
    parser = argparse.ArgumentParser(
        description="Deduplicate a knowledge base file where each line is a knowledge item. "
                    "Supports exact matching, semantic similarity (embeddings), LLM judgment, or hybrid mode.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Exact string matching (fast, default)
  python deduplicate_knowledge.py knowledge.txt
  
  # Semantic deduplication with sentence-transformers (local, free)
  python deduplicate_knowledge.py knowledge.txt --method semantic
  
  # Pure LLM-based deduplication (slow but most accurate)
  python deduplicate_knowledge.py knowledge.txt --method llm
  
  # Hybrid mode: embeddings for fast filtering + LLM for edge cases (RECOMMENDED)
  python deduplicate_knowledge.py knowledge.txt --method semantic --hybrid
  
  # Adjust similarity threshold (higher = more strict)
  python deduplicate_knowledge.py knowledge.txt --method semantic --threshold 0.98
  
  # Use a different sentence-transformers model
  python deduplicate_knowledge.py knowledge.txt --method semantic --model all-mpnet-base-v2
        """
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to the input knowledge base file'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Path to the output file (default: input_file_deduplicated.txt)'
    )
    parser.add_argument(
        '-i', '--in-place',
        action='store_true',
        help='Overwrite the input file with deduplicated content'
    )
    parser.add_argument(
        '-m', '--method',
        type=str,
        choices=['exact', 'semantic', 'llm'],
        default='exact',
        help='Deduplication method: exact (string matching), semantic (embedding-based), or llm (LLM judgment)'
    )
    parser.add_argument(
        '-t', '--threshold',
        type=float,
        default=0.95,
        help='Cosine similarity threshold for semantic deduplication (0.0-1.0, default: 0.95)'
    )
    parser.add_argument(
        '--hybrid',
        action='store_true',
        help='Hybrid mode: use embeddings for fast filtering, then LLM for edge cases (recommended)'
    )
    parser.add_argument(
        '--llm-server',
        type=str,
        default='azure',
        help='LLM server type (default: azure). Options: azure, openai, etc.'
    )
    parser.add_argument(
        '--llm-model',
        type=str,
        default='gpt-5-mini',
        help='LLM model name (default: gpt-5-mini). Examples: gpt-5-mini, claude-3-haiku, etc.'
    )
    parser.add_argument(
        '--hybrid-threshold-low',
        type=float,
        default=0.85,
        help='Lower threshold for hybrid mode (below this = definitely not duplicate, default: 0.85)'
    )
    parser.add_argument(
        '--hybrid-threshold-high',
        type=float,
        default=0.98,
        help='Upper threshold for hybrid mode (above this = definitely duplicate, default: 0.98)'
    )
    
    args = parser.parse_args()
    
    # Validate thresholds
    if args.method == 'semantic' and not (0.0 <= args.threshold <= 1.0):
        print("Error: similarity threshold must be between 0.0 and 1.0", file=sys.stderr)
        sys.exit(1)
    if args.hybrid and not (0.0 <= args.hybrid_threshold_low < args.hybrid_threshold_high <= 1.0):
        print("Error: hybrid thresholds must satisfy 0.0 <= low < high <= 1.0", file=sys.stderr)
        sys.exit(1)
    
    try:
        deduplicate_knowledge(
            args.input_file,
            args.output,
            args.in_place,
            method=args.method,
            similarity_threshold=args.threshold,
            llm_server_type=args.llm_server,
            llm_model=args.llm_model,
            hybrid_mode=args.hybrid,
            hybrid_threshold_low=args.hybrid_threshold_low,
            hybrid_threshold_high=args.hybrid_threshold_high
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()