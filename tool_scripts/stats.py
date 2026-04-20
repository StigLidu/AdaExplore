import argparse
import os
import ast
import math
import shutil
import json
import numpy as np
import matplotlib.pyplot as plt

source_mapping = {
    "KB": "KernelBench",
}

level_mapping = {
    "l1": "level1",
    "l2": "level2",
    "l3": "level3",
    "l4": "level4",
}

SPEEDUP_CLAMP = (0.0, 10.0)

def parse_args():
    parser = argparse.ArgumentParser()
    # The log folder to process
    # Folder structure: agent-type_knowledge-base_task-level_step_date-time
    parser.add_argument(
        "--log_folder",
        type=str,
        nargs="+",
        default=None,
        help="One or more log folder paths to process",
    )
    parser.add_argument("--range_end", type=int, default=100)
    parser.add_argument("--draw_curve", action="store_true")
    parser.add_argument("--save_eval_task_ids_path", type=str, default=None)
    parser.add_argument("--load_eval_task_ids_path", type=str, default=None)
    parser.add_argument("--best_metrics_file", type=str, default=None)
    parser.add_argument("--step", type=int, default=None, help="Step suffix for metrics files")
    parser.add_argument("--force_refind", action="store_true", default=False, help="Force refind the best kernel")
    parser.add_argument("--filter_meaningless", action="store_true", default=False, help="Filter out meaningless kernels")
    parser.add_argument("--meaningless_list_path", type=str, default="datasets/meaningless.txt", help="Path to the meaningless list")
    parser.add_argument("--verbose", action="store_true", default=False, help="Verbose each task's metrics")
    args = parser.parse_args()
    return args


def load_eval_task_ids(path):
    if path is None:
        return None
    with open(path, "r") as f:
        return [(int(l), int(p)) for l, p in (line.strip().split() for line in f)]


def parse_metrics_txt(content):
    """Parse metrics from txt format."""
    compiled = content.split("compiled=")[1].split(" ")[0] == "True"
    correctness = content.split("correctness=")[1].split(" ")[0] == "True"
    fast_p = 0
    if compiled and correctness:
        runtime = ast.literal_eval(content.split("runtime_stats=")[1].split("\n")[0])
        fast_p = runtime.get("fast_p", 0)
    return compiled, correctness, fast_p


def parse_metrics_json(content):
    """Parse metrics from json format."""
    compiled = content.get("compiled", False)
    correctness = content.get("correctness", False)
    fast_p = content.get("runtime_stats", {}).get("fast_p", 0) if compiled and correctness else 0
    return compiled, correctness, fast_p


def parse_metrics_file(file_path):
    """Parse metrics from either txt or json file. Returns (compiled, correctness, fast_p, raw_content)."""
    with open(file_path, "r") as f:
        raw = f.read()
    if file_path.endswith(".json"):
        compiled, correctness, fast_p = parse_metrics_json(json.loads(raw))
    else:
        compiled, correctness, fast_p = parse_metrics_txt(raw)
    return compiled, correctness, fast_p, raw


def extract_step(filename, log_folder, total_steps):
    """Extract step number from filename based on agent type."""
    if "proposal" in filename:
        if "PS" in log_folder or "IRL" in log_folder or "IRB" in log_folder:
            return int(filename.split("proposal_")[1].split("_")[0])
        elif "SQUARE" in log_folder:
            sqrt_step = int(np.sqrt(total_steps) + 0.1)
            return (int(filename.split("proposal_")[1].split("_")[0]) - 1) * sqrt_step + 1
        elif "IRS" in log_folder:
            return 1
        else:
            raise ValueError(f"log folder does not match the file name pattern: {filename}")
    elif "tune" in filename:
        parts = filename.split("tune_")[1].split("_")
        if "IRS" in log_folder:
            return int(parts[1])
        elif "SQUARE" in log_folder:
            sqrt_step = int(np.sqrt(total_steps) + 0.1)
            return (int(parts[0]) - 1) * sqrt_step + int(parts[1]) + 1
        else:
            raise ValueError(f"log folder does not match the file name pattern: {filename}")
    elif "step_" in filename and ("MCTS" in log_folder or "MH" in log_folder or "CHAIN" in log_folder):
        return int(filename.split("step_")[1].split("_")[0])
    else:
        raise ValueError(f"log folder does not match the file name pattern: {filename}")

def has_eval_error(content):
    """Check if content contains evaluation errors."""
    errors = ["torch.OutOfMemoryError", "No space left on device", "Remote evaluation request failed", "evaluation failed after 3 attempts"]
    return any(err in str(content) for err in errors)


def process_log_folder(
    log_folder: str,
    args,
    to_eval_task_ids,
    step_acc_map: dict,
    step_prefix_map: dict,
    test_idxs: list,
    eval_task_ids: list,
    total_steps: int,
):
    """Process a single log folder and update statistics."""
    correct_count = 0
    sum_speedup = 0
    total_count = 0
    log_sum_speedup = 0
    log_sum_speedup_with_incorrect = 0
    bug_eval_count = 0
    count_1_2 = 0
    count_2 = 0
    prefix_best_data = {}
    if args.filter_meaningless:
        with open(args.meaningless_list_path, "r") as f:
            meaningless_list = [line.strip() for line in f.readlines()]

    for result_folder in os.listdir(log_folder):
        folder_path = os.path.join(log_folder, result_folder)
        if not os.path.isdir(folder_path):
            continue

        no_place_tag = False
        if result_folder.strip("/").endswith("_BUG"):
            if result_folder.replace("_BUG", "") in os.listdir(log_folder):
                continue
            else:
                print("WARNING: Bug folder does not have a corresponding normal folder, use the bug folder as the normal folder")
                no_place_tag = True
    
        # if the folder is empty, remove it
        if os.path.isdir(folder_path) and not os.listdir(folder_path):
            print(f"Removing empty folder: {folder_path}")
            shutil.rmtree(folder_path)
            continue
        
        level, problem_id = map(int, result_folder.split("_")[:2])

        if problem_id > args.range_end:
            continue
        
        if args.filter_meaningless:
            filter_out = False
            data_source = source_mapping.get(log_folder.split("/")[-1].split("_")[2].split("-")[0], None)
            data_level = level_mapping.get(log_folder.split("/")[-1].split("_")[2].split("-")[1], None)
            if data_source is None or data_level is None:
                print(f"WARNING: Data source or level not found for {log_folder}")
            else:
                for meaningless_kernel in meaningless_list:
                    if f"{data_source}/{data_level}/{problem_id}_" in meaningless_kernel:
                        filter_out = True
                        break
            if filter_out:
                continue

        # Find best metrics file
        best_metrics_path = next(
            (os.path.join(folder_path, f) for f in os.listdir(folder_path) if args.best_metrics_file in f),
            None,
        )

        if to_eval_task_ids is not None and (level, problem_id) not in to_eval_task_ids:
            continue

        test_idxs.append((level, problem_id))
        eval_task_ids.append((level, problem_id))
        total_count += 1
        ood_tag = True

        # Parse step metrics
        for file in os.listdir(folder_path):
            is_metrics = ("metrics.txt" in file or "metrics.json" in file) and "best_metrics" not in file
            if not is_metrics:
                continue
            
            step = extract_step(file, log_folder, total_steps)
            file_path = os.path.join(folder_path, file)
            _, _, fast_p, raw = parse_metrics_file(file_path)
            
            if has_eval_error(raw) and ood_tag:
                bug_eval_count += 1
                ood_tag = False
            
            step_prefix = file.rsplit("_metrics.", 1)[0]
            fast_p = min(fast_p, SPEEDUP_CLAMP[1])
            step_acc_map.setdefault(step, {})[(level, problem_id)] = fast_p
            step_prefix_map.setdefault(step, {})[(level, problem_id)] = step_prefix

        # Parse best metrics
        if best_metrics_path is not None and not args.force_refind:
            _, _, best_fast_p, _ = parse_metrics_file(best_metrics_path)
        else:
            # Use max of step_acc_map for first `step` steps
            best_step = np.argmax(
                [step_acc_map.get(s, {}).get((level, problem_id), 0) for s in range(1, args.step + 1)]
            ) + 1
            best_fast_p = step_acc_map.get(best_step, {}).get((level, problem_id), 0)
            best_step_prefix = step_prefix_map.get(best_step, {}).get((level, problem_id), "")
            # Copy the best kernel and metrics
            shutil.copy(os.path.join(folder_path, f"{best_step_prefix}.py"),
                        os.path.join(folder_path, f"global_best_kernel_{args.step}.py"))
            for ext in [".txt", ".json"]:
                src = os.path.join(folder_path, f"{best_step_prefix}_metrics{ext}")
                if os.path.exists(src):
                    shutil.copy(src, os.path.join(folder_path, f"global_best_metrics_{args.step}{ext}"))
                    break
            else:
                raise ValueError(f"No metrics file found for {best_step_prefix}")

            prefix_best_fast_p = []
            current_best_fast_p = 0
            for prefix_checkpoint in range(1, args.step + 1):
                current_best_fast_p = max(current_best_fast_p, step_acc_map.get(prefix_checkpoint, {}).get((level, problem_id), 0))
                prefix_best_fast_p.append(current_best_fast_p)
            prefix_best_data[f"{level}_{problem_id}"] = prefix_best_fast_p

        if best_fast_p > 0:
            best_fast_p = min(best_fast_p, SPEEDUP_CLAMP[1])
            correct_count += 1
            sum_speedup += best_fast_p
            log_sum_speedup += math.log(best_fast_p)
            log_sum_speedup_with_incorrect += math.log(best_fast_p)
            if best_fast_p > 1.2:
                count_1_2 += 1
            if best_fast_p > 2:
                count_2 += 1
        else:
            log_sum_speedup_with_incorrect += math.log(0.1)

        if args.verbose:
            print(f"Task ID: {level} {problem_id}, Fast P: {best_fast_p}")

        # Handle bug folder
        if not ood_tag and not no_place_tag:
            bug_folder = os.path.join(log_folder, f"{level}_{problem_id}_BUG")
            if os.path.exists(bug_folder):
                shutil.rmtree(bug_folder)
            os.rename(folder_path, bug_folder)
            print(f"Bug Task ID: {level} {problem_id}")

    return {
        "correct_count": correct_count,
        "sum_speedup": sum_speedup,
        "total_count": total_count,
        "log_sum_speedup": log_sum_speedup,
        "log_sum_speedup_with_incorrect": log_sum_speedup_with_incorrect,
        "bug_eval_count": bug_eval_count,
        "count_1_2": count_1_2,
        "count_2": count_2,
        "prefix_best_data": prefix_best_data,
    }


def draw_curves(log_folder: str, test_idxs: list, step_acc_map: dict, args):
    """Draw and save performance curves for a log folder."""
    total_steps = max(step_acc_map.keys()) + 1 if step_acc_map else 1
    all_speedup_list = np.zeros((len(test_idxs), total_steps))
    max_update_count, max_update_idx = 0, None

    for i, (level, problem_id) in enumerate(test_idxs):
        speedup_list = np.zeros(total_steps)
        update_count = 0
        
        for k in range(1, total_steps):
            if k in step_acc_map and (level, problem_id) in step_acc_map[k]:
                speedup_list[k] = step_acc_map[k][(level, problem_id)]
            else:
                print(f"[Warning] Step {k} of {level} {problem_id} has no data")
                speedup_list[k] = speedup_list[k - 1]
        
        for k in range(2, total_steps):
            if speedup_list[k] > speedup_list[k - 1]:
                update_count += 1
            speedup_list[k] = max(speedup_list[k], speedup_list[k - 1])
        
        all_speedup_list[i] = speedup_list
        if update_count >= max_update_count:
            max_update_count, max_update_idx = update_count, (level, problem_id)

    avg_speedup_list = all_speedup_list.mean(axis=0)
    avg_none_zero_count = (all_speedup_list > 0).mean(axis=0)
    print(f"Avg none zero count: {avg_none_zero_count}")
    print(f"Max update count: {max_update_count}")
    print(f"Max update count idx: {max_update_idx}")
    os.makedirs(f"results/{log_folder.split('/')[-1]}", exist_ok=True)
    save_path = f"results/{log_folder.split('/')[-1]}"
    plt.figure()
    plt.plot(avg_speedup_list)
    plt.savefig(f"{save_path}/speedup.png")
    print(f"Saved speedup curve to {save_path}/speedup.png")
    plt.figure()
    plt.plot(avg_none_zero_count)
    plt.savefig(f"{save_path}/accuracy.png")
    print(f"Saved accuracy curve to {save_path}/accuracy.png")
    np.savetxt(f"{save_path}/speedup.csv", avg_speedup_list, delimiter=" ")
    np.savetxt(f"{save_path}/accuracy.csv", avg_none_zero_count, delimiter=" ")


def main():
    args = parse_args()
    to_eval_task_ids = load_eval_task_ids(args.load_eval_task_ids_path)
    
    if args.log_folder is None:
        args.log_folder = [log_folder.split("/")[-1] for log_folder in os.listdir("outputs") if log_folder.count("_") >= 4 and "SYN" not in log_folder]
        args.log_folder = [os.path.join("outputs", log_folder) for log_folder in args.log_folder]
    
    # Record incomplete folders
    partial_folders = []
    
    # Process each log folder
    for log_folder in args.log_folder:
        print(f"\n{'='*60}")
        log_folder = log_folder.strip("/")
        print(f"Processing log folder: {log_folder}")
        print(f"{'='*60}")

        # [TODO]:
        if "FIT" in log_folder:
            print(f"Skipping {log_folder} because it is a FIT folder")
            continue
        
        if args.step is None:
            try:
                total_steps = int(log_folder.split("/")[-1].split("_")[-2])
            except ValueError:
                print(f"Warning: Could not extract number of steps from log folder name {log_folder}")
                total_steps = ""
            args.step = total_steps
        if args.best_metrics_file is None:
            on_sf_name = f"global_best_metrics_on_sf_{args.step}.json"
            found_on_sf = any(
                os.path.exists(os.path.join(log_folder, d, on_sf_name))
                for d in os.listdir(log_folder)
                if os.path.isdir(os.path.join(log_folder, d))
            )
            if found_on_sf:
                args.best_metrics_file = on_sf_name
            else:
                args.best_metrics_file = f"global_best_metrics_{args.step}.json"
        print(f"Using {args.best_metrics_file} as the best metrics file")
        print("Determined step number from log folder name: ", args.step)
        
        # Statistics for this folder
        step_acc_map = {}
        step_prefix_map = {}
        test_idxs = []
        eval_task_ids = []
        
        stats = process_log_folder(
            log_folder, args, to_eval_task_ids, step_acc_map, step_prefix_map, test_idxs, eval_task_ids, args.step
        )
        
        # Print statistics for this folder
        tc, cc = stats["total_count"], stats["correct_count"]
        if tc > 0:
            print(f"\nStatistics for {log_folder}:")
            print(f"  Total: {tc}, Correct: {cc}, Accuracy: {cc/tc:.4f}")
            print(f"  Avg speedup: {stats['sum_speedup']/tc:.4f}, "
                  f"GM(10): {math.exp(stats['log_sum_speedup_with_incorrect']/tc):.4f}")
            if cc > 0:
                print(f"  Avg speedup (correct only): {stats['sum_speedup']/cc:.4f}, "
                      f"GM(10) (correct only): {math.exp(stats['log_sum_speedup']/cc):.4f}")
            print(f"  Bug eval count: {stats['bug_eval_count']}")
            print(f"  Count 1.2: {stats['count_1_2']}")
            print(f"  Count 2: {stats['count_2']}")
        else:
            print(f"No valid results found in {log_folder}")

        if stats['prefix_best_data']:
            prefix_path = os.path.join(log_folder, "best_prefix.json")
            with open(prefix_path, "w") as f:
                json.dump(stats['prefix_best_data'], f)
            print(f"  Saved prefix best data to {prefix_path}")

        if stats['bug_eval_count'] > 0 or stats['total_count'] < args.range_end:
            partial_folders.append(log_folder)
        
        # Draw curve for this folder
        if args.draw_curve and test_idxs:
            draw_curves(log_folder, test_idxs, step_acc_map, args)
        
        # Save eval task ids for this folder
        if args.save_eval_task_ids_path and eval_task_ids:
            save_path = args.save_eval_task_ids_path
            # If multiple folders, append folder name to save path
            if len(args.log_folder) > 1:
                folder_name = log_folder.split('/')[-1]
                save_path = save_path.replace('.txt', f'_{folder_name}.txt')
            with open(save_path, "w") as f:
                f.writelines(f"{level} {problem_id}\n" for level, problem_id in eval_task_ids)
    
    print("incomplete folders: ", partial_folders)


if __name__ == "__main__":
    main()