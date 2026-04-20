# Collect error messages from logs and summarize them into a skill memory file.

import os
import argparse
from tqdm import tqdm
import sys
import random
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.inference_server import create_inference_server, query_inference_server
from skill_memory.deduplicate_knowledge import llm_judge_duplicate_batch

PROMPT_TEMPLATE = """You are an assistant that extracts *minimal, log-grounded constraints* from Triton kernel error messages
and records them as one-line "You cannot ..." rules in a skill memory file.

Given the code block and the error message:

- Summarize the error into **exactly one sentence**
- The sentence MUST begin with "You cannot ..."
- The sentence MUST describe the **minimal prohibited action** implied by the error
- The sentence MUST NOT include:
  - suggestions, fixes, or alternatives
  - reasons, explanations, or consequences
  - assumptions beyond what is directly implied by the error message
- Do NOT generalize beyond the specific scope indicated by the error
  (e.g., inside a Triton kernel, at compile time, for constexpr parameters, etc.)
- **Do NOT guess, infer, or hallucinate a rule**
- **If the error message does not clearly imply a single, well-defined prohibited action,
  output exactly: `no guidance`**

Only use information that is directly supported by the error message.

<Code Block>
{code_block}
</Code Block>

<Error Message>
{error_message}
</Error Message>

Now write the one-sentence constraint.
"""

def parse_args():
    parser = argparse.ArgumentParser(description="Collect error messages from logs and summarize them into a skill memory file")
    parser.add_argument("file", type=str, nargs="?", default=None,
                        help="Optional: Process a single metrics file instead of batch processing")
    parser.add_argument("--log-dir", type=str, default="outputs", 
                        help="Directory containing log files (default: outputs)")
    parser.add_argument("--server", type=str, default="azure", 
                        help="Inference server type (default: azure)")
    parser.add_argument("--model-name", type=str, default="gpt-5-mini", 
                        help="Model name to use for inference (default: gpt-5-mini)")
    parser.add_argument("--knowledge-store-path", type=str, default="knowledge_store.txt", 
                        help="Path to store the skill memory file (default: knowledge_store.txt)")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for shuffling logs (default: 42)")
    parser.add_argument("--max-logs", type=int, default=3000, 
                        help="Maximum number of logs to process (default: 3000)")
    parser.add_argument("--filter-max-difference", action="store_true", 
                        help="Filter out logs containing 'max_difference' (default: False)")
    parser.add_argument("--progress_file", type=str, default=None, 
                        help="Path to store the progress (default: None)")
    return parser.parse_args()

def check_error_exists(file: str, filter_max_difference: bool = True):
    with open(file, "r") as f:
        content = f.read()
        if "correctness=False" not in content or (filter_max_difference and "max_difference" in content):
            return False, "Not a CE or RE error" # only collect experience from CE and RE
    return True, "Error exists"

def collect_experience_from_single_log(file: str, server: callable, model_name: str, filter_max_difference: bool = False):
    if file.endswith("_metrics.txt"):
        with open(file, "r") as f:
            content = f.read()
            if "correctness=False" not in content or (filter_max_difference and "max_difference" in content):
                return False, "Not a CE or RE error" # only collect experience from CE and RE
    elif file.endswith("_metrics.json"):
        metrics = json.load(open(file, "r"))
        if metrics is None:
            return False, "Metrics file not found"
        if metrics["correctness"] == True or (filter_max_difference and "max_difference" in str(metrics)):
            return False, "Not a CE or RE error" # only collect experience from CE and RE
        content = str(metrics)
    else:
        return False, "Unknown metrics file format"
    code_file = file.replace("_metrics.txt", ".py").replace("_metrics.json", ".py")
    if not os.path.exists(code_file):
        print(f"Warning: Code file not found: {code_file}, skipping...")
        return False, "Code file not found"
    with open(code_file, "r") as f:
        code_block = f.read()
    error_message = content
    prompt = PROMPT_TEMPLATE.format(code_block=code_block, error_message=error_message)
    response = query_inference_server(server, model_name, prompt)
    if "no guidance" in response.lower():
        return False, "No guidance"
    return True, response.strip().split("\n")[-1]

def update_memory(memory_path: str, log_path: str, server: callable, model_name: str, filter_max_difference: bool = False):
    files = os.listdir(log_path)
    log_path_list = []
    for file in files:
        if (file.endswith("_metrics.txt") or file.endswith("_metrics.json")) and not "best_metrics" in file and not "step_0" in file:
            log_path_list.append(os.path.join(log_path, file))
    knowledge_base = []
    score_base = []
    if os.path.exists(memory_path):
        with open(memory_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                assert "||" in line, "Knowledge store file is not formatted correctly"
                knowledge_base_item, score = line.strip().split("||")
                knowledge_base.append(knowledge_base_item.strip())
                score_base.append(float(score.strip()))


    print("-" * 100
          + "\nskill memory"
          + "\n"
          + "-" * 100
          + "\n")
    print("example skill memory entry:")
    print(knowledge_base[0] if len(knowledge_base) > 0 else "No skill memory")
    print("example score base:")
    print(score_base[0] if len(score_base) > 0 else "No score base")

    for log_path in log_path_list:
        try:
            success, response = collect_experience_from_single_log(
                file=log_path, server=server, model_name=model_name, filter_max_difference=filter_max_difference)
            if success:
                dup_flag = False
                if len(knowledge_base) > 0:
                    result = llm_judge_duplicate_batch(
                        candidate_text=response, 
                        existing_texts=knowledge_base, 
                        existing_unique_indices=list(range(len(knowledge_base))), 
                        model_name=model_name,
                        existing_server=server
                    )
                    if result is not None:
                        matched_unique_idx, confidence = result
                        if confidence > 0.5:
                            score_base[matched_unique_idx] += 1
                            dup_flag = True
                if not dup_flag:
                    knowledge_base.append(response)
                    score_base.append(1.0)
                print("response:")
                print(response)
                print("dup_flag:", dup_flag)
                print("-" * 100)
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise e

    with open(memory_path, "w") as f:
        for knowledge_base_item, score in zip(knowledge_base, score_base):
            f.write(f"{knowledge_base_item}||{score}\n")

if __name__ == "__main__":
    args = parse_args()
    log_dir = args.log_dir
    logs = sorted([log for log in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, log))])
    server = create_inference_server(args.server)
    model_name = args.model_name

    if args.file:
        # single log
        success, response = collect_experience_from_single_log(
            file=args.file, server=server, model_name=model_name, filter_max_difference=args.filter_max_difference)
        if success:
            print(response)
        else:
            print(f"Error: {response}")
        exit()

    # load all log paths
    log_path_list = []
    for log in logs:
        files = os.listdir(os.path.join(log_dir, log))
#        if not log.split("_")[-1].isdigit() or int(log.split("_")[-1]) > 20: continue # skip logs after 20 steps
        for file in files:
            if (file.endswith("_metrics.txt") or file.endswith("_metrics.json")) and not file.endswith("_best_metrics.txt") and not "step_0" in file:
                log_path_list.append(os.path.join(log_dir, log, file))

    if args.progress_file is not None:
        with open(args.progress_file, "r") as f:
            processed_log_path_list = f.readlines()
            processed_log_path_list = [line.strip() for line in processed_log_path_list]
    else:
        processed_log_path_list = []

    random.seed(args.seed)
    random.shuffle(log_path_list)

    print(f"Total {len(log_path_list)} logs to process")
    total_error_count = 0
    for log_path in log_path_list: 
        success, response = check_error_exists(log_path, args.filter_max_difference)
        if success:
            total_error_count += 1
    print(f"Total {total_error_count} error logs to process")

    # Load the existing skill memory file.
    knowledge_base = []
    score_base = []
    knowledge_store_path = args.knowledge_store_path
    if os.path.exists(knowledge_store_path):
        with open(knowledge_store_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                knowledge_base_item, score = line.split("||")
                knowledge_base.append(knowledge_base_item.strip())
                score_base.append(float(score.strip()))

    log_path_list = log_path_list[:args.max_logs]
    new_log_path_list = []
    processed_count = 0
    pbar = tqdm(enumerate(log_path_list), total=len(log_path_list), desc="Processing logs")
    for idx, log_path in pbar:
        if log_path in processed_log_path_list:
            continue
        new_log_path_list.append(log_path)
        try:
            success, response = collect_experience_from_single_log(
                file=log_path, server=server, model_name=model_name, filter_max_difference=args.filter_max_difference)
            if success:
                dup_flag = False
                if len(knowledge_base) > 0:
                    result = llm_judge_duplicate_batch(
                        candidate_text=response, 
                        existing_texts=knowledge_base, 
                        existing_unique_indices=list(range(len(knowledge_base))), 
                        existing_server=server
                    )
                    if result is not None:
                        matched_unique_idx, confidence = result
                        if confidence > 0.5:
                            score_base[matched_unique_idx] += 1
                            dup_flag = True
                if not dup_flag:
                    knowledge_base.append(response)
                    score_base.append(1.0)
            else:
                tqdm.write(f"Warning: {response}")
        except Exception as e:
            tqdm.write(f"Warning: Error processing {log_path}: {e}")

        processed_count += 1
        # Update progress bar with skill memory size.
        pbar.set_postfix({"KB size": len(knowledge_base)})
        # Persist the skill memory every 100 logs.
        if processed_count % 100 == 0:
            tqdm.write(f"Processed {processed_count} logs, {len(knowledge_base)} knowledge items collected")
            # Update the skill memory file.
            with open(knowledge_store_path, "w") as f:
                for knowledge_base_item, score in zip(knowledge_base, score_base):
                    f.write(f"{knowledge_base_item}||{score}\n")
            # add the new log path list to the progress file
            if args.progress_file is not None:
                with open(args.progress_file, "a") as f:
                    f.write("\n".join(new_log_path_list) + "\n")
                new_log_path_list = []
    
    # Save the final skill memory file.
    with open(knowledge_store_path, "w") as f:
        for knowledge_base_item, score in zip(knowledge_base, score_base):
            f.write(f"{knowledge_base_item}||{score}\n")
    
    if args.progress_file is not None:
        with open(args.progress_file, "a") as f:
            f.write("\n".join(new_log_path_list) + "\n")
    
    print(f"Processed {processed_count} logs, {len(knowledge_base)} knowledge items collected")