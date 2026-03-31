import logging
from typing import List, Dict, Any, Optional
from datasets import load_dataset, Dataset

logger = logging.getLogger(__name__)

def load_dataset_for_benchmarking(
    dataset_name: str,
    split: str,
    limit: int = None,
    prompt_field: str = None,
    answer_field: str = None,
    subset: str = None
) -> List[Dict[str, Any]]:
    """
    Loads and formats a dataset for benchmarking.

    Args:
        dataset_name: The name of the dataset on Hugging Face Hub.
        split: The dataset split to use (e.g., 'test').
        limit: The maximum number of samples to return.
        prompt_field: The field name containing the problem/question.
        answer_field: The field name containing the answer/solution.
        subset: Optional subset name for datasets with multiple configurations.

    Returns:
        A list of formatted problems.
    """
    logger.info(f"Loading dataset {dataset_name} (split: {split}, subset: {subset})")
    try:
        dataset: Dataset
        if subset:
            dataset = load_dataset(dataset_name, name=subset, split=split)
        else:
            dataset = load_dataset(dataset_name, split=split)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

    # Determine field names if not provided
    if prompt_field is None:
        prompt_field = _find_prompt_field(dataset)
    if answer_field is None:
        answer_field = _find_answer_field(dataset)

    logger.info(f"Using prompt field: {prompt_field}, answer field: {answer_field}")

    formatted_problems = []
    for i, example in enumerate(dataset):
        if limit and i >= limit:
            break
        
        # Extract prompt and answer
        if prompt_field in example and answer_field in example:
            prompt = example[prompt_field]
            answer = example[answer_field]
        else:
            logger.warning(f"Skipping example {i} due to missing fields. Available: {example.keys()}")
            continue

        formatted_id = example.get("task_id") or example.get("id") or f"{dataset_name}-{split}-{i}"
        formatted_problems.append({
            "id": formatted_id,
            "prompt": prompt,
            "reference_answer": answer,
        })
        
    logger.info(f"Loaded {len(formatted_problems)} problems.")
    return formatted_problems

def _find_prompt_field(dataset: Dataset) -> str:
    """Find the field name containing the problem/question."""
    prompt_fields = ["prompt", "problem", "question"]
    for field in prompt_fields:
        if field in dataset.column_names:
            return field
    raise ValueError(f"Could not find prompt field in dataset. Available columns: {dataset.column_names}")

def _find_answer_field(dataset: Dataset) -> str:
    """Find the field name containing the answer/solution."""
    answer_fields = ["answer", "canonical_solution", "solution", "reference_answer"]
    for field in answer_fields:
        if field in dataset.column_names:
            return field
    raise ValueError(f"Could not find answer field in dataset. Available columns: {dataset.column_names}")