import json
import logging
import os
import time
from pathlib import Path
import yaml
import sys

from tqdm import tqdm

from fasttts import FastTTS
from benchmark_config import build_benchmark_config_from_yaml
from dataset_utils import load_dataset_for_benchmarking

_log_level = getattr(logging, os.environ.get("FASTTTS_LOG_LEVEL", "INFO").upper(), logging.INFO)
logging.basicConfig(level=_log_level, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_benchmark_from_config(config_path: str):
    """
    Runs one or more benchmarks as specified in a YAML config file.
    """
    with open(config_path, 'r') as f:
        yaml_obj = yaml.safe_load(f)

    # Support either a single config or a list of configs
    configs = yaml_obj if isinstance(yaml_obj, list) else [yaml_obj]

    for config_dict in configs:
        config = build_benchmark_config_from_yaml(config_dict)
        logger.info(f"Starting benchmark: {config.name}")
        logger.info(f"  Dataset: {config.dataset_name}")
        if config.dataset_subset:
            logger.info(f"  Subset: {config.dataset_subset}")
        if config.prompt_field or config.answer_field:
            logger.info(f"  Custom fields -> prompt: {config.prompt_field}, answer: {config.answer_field}")
        logger.info(f"  Beam width: {config.search_config.beam_width}")
        logger.info(f"  N: {config.search_config.n}")
        logger.info(f"  Iterations: {config.search_config.num_iterations}")
        logger.info(f"  Spec diff: {config.enable_spec_diff}")
        logger.info(f"  Offload enabled: {config.offload_enabled}")
        logger.info(f"  Prefix aware scheduling: {config.enable_prefix_aware_scheduling}")

        # --- Load Dataset ---
        problems = load_dataset_for_benchmarking(
            dataset_name=config.dataset_name,
            split=config.dataset_split,
            limit=config.limit,
            prompt_field=config.prompt_field,
            answer_field=config.answer_field,
            subset=config.dataset_subset,
        )
        if not problems:
            logger.error("No problems loaded. Exiting.")
            continue

        # --- Initialize FastTTS ---
        logger.info("Initializing FastTTS...")
        fast_tts = FastTTS(config.fasttts_config)

        try:
            # --- Run Inference ---
            output_dir = Path(config.output_dir)
            output_dir.mkdir(exist_ok=True)
            config_suffix = f"_bw{config.search_config.beam_width}_n{config.search_config.n}_iter{config.search_config.num_iterations}"
            if config.enable_spec_diff:
                config_suffix += "_specdiff"
            if config.offload_enabled:
                config_suffix += "_offload"
            output_file = output_dir / f"{config.name}{config_suffix}_results.jsonl"
            existing_ids = set()
            existing_results = []
            if output_file.exists():
                logger.info(f"Output file {output_file} already exists. Loading existing results...")
                with open(output_file, "r") as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            # Support both old and new format
                            if isinstance(data, dict) and "id" in data:
                                existing_ids.add(data["id"])
                                existing_results.append(data)
                        except Exception as e:
                            logger.warning(f"Skipping invalid line in output file: {e}")

            logger.info(f"Running inference on {len(problems)} problems...")
            logger.info(f"Results will be saved to: {output_file}")
            start_time = time.time()
            new_results = []
            for problem in tqdm(problems, desc=f"Benchmarking {config.name}"):
                qid = problem["id"]
                if qid in existing_ids:
                    logger.info(f"Skipping problem {qid} - already processed")
                    continue
                all_solutions = fast_tts.search(
                    problems=[problem["prompt"]],
                    search_config=config.search_config
                )
                # Save with question id
                result = {"id": qid, "prompt": problem["prompt"], "reference_answer": problem.get("reference_answer"), "solutions": all_solutions}
                new_results.append(result)
                existing_ids.add(qid)
                
                # Write results incrementally to avoid losing progress if process is killed
                all_results = existing_results + new_results
                all_results.sort(key=lambda x: str(x["id"]))
                with open(output_file, "w") as f:
                    for res in all_results:
                        f.write(json.dumps(res) + "\n")
            end_time = time.time()
            total_time = end_time - start_time
            logger.info("--- Benchmark Summary ---")
            logger.info(f"Dataset: {config.dataset_name}")
            logger.info(f"Problems attempted: {len(problems)}")
            logger.info(f"Total time: {total_time:.2f}s")
            logger.info(f"Average time per problem: {total_time/len(problems):.2f}s")
            logger.info(f"Results saved to: {output_file}")
            logger.info("-------------------------")
        except Exception as e:
            logger.error(f"Error running benchmark: {e}")
            raise e
        finally:
            fast_tts.shutdown()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_benchmarks.py <config_file.yaml>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    run_benchmark_from_config(config_path)