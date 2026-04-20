import argparse
from agent.inference_server import query_inference_server
from agentprompt.proposer_prompt import (
    generate_proposer_prompt,
    generate_pool_prompt_dual,
)
from src.utils import extract_first_code
from src.eval import eval_kernel_against_ref, wrapped_eval_kernel_against_ref
from agentprompt.reviser_prompt import generate_reviser_prompt
from agentprompt.tuner_prompt import generate_tuner_prompt
from agent.utils import extract_edits, str_replace

def _use_performance_metric(args: argparse.Namespace) -> bool:
    test_source = str(getattr(args, "test_source", "KB")).upper()
    return test_source not in {"TBG"}

def single_small_step(ref_arch_src: str, inference_server: str, previous_kernels: list, previous_metrics: list, args: argparse.Namespace):
    # Reviser Agent (if disabled, no reviser agent will be used)
    # Check if all previous kernels are wrong - if so, skip reviewer
    all_previous_wrong = (
        len(previous_metrics) > 0 and 
        all(not metric.correctness for metric in previous_metrics)
    )
    
    if (not args.disable_reviewer) and (not all_previous_wrong or getattr(args, 'force_reviser', False)):
        # TODO: currently, only filter wrong attempts for tuner agent, not for reviser agent, to prevent repeating the same mistakes
        reviser_prompt = generate_reviser_prompt(
            task_params=args.task_params,
            custom_triton_kernels=previous_kernels[-1] if len(previous_kernels) > 0 else None, 
            run_info=previous_metrics[-1] if len(previous_metrics) > 0 else None,
            experience_guidance_path=args.general_memory_path,
            knowledge_1_threshold=args.knowledge_1_threshold,
        )
        reviser_output = query_inference_server(
            server=inference_server,
            model_name=args.model_name,
            prompt=reviser_prompt,
            max_completion_tokens=args.max_completion_tokens,
        )
    else:
        reviser_prompt = "No tuning guidance provided."
        reviser_output = "Please fix or improve the performance of the custom kernels based on the execution logs."

    # Tuner Agent
    tuner_prompt = generate_tuner_prompt(
        task_params=args.task_params,
        previous_kernels=previous_kernels, 
        previous_metrics=previous_metrics, 
        tuning_guidance=reviser_output,
        experience_guidance_path=args.general_memory_path,
        knowledge_1_threshold=args.knowledge_1_threshold,
        filter_wrong_attempts=getattr(args, 'filter_wrong_attempts', False),
    )

    tuner_output = query_inference_server(
        server=inference_server,
        model_name=args.model_name,
        prompt=tuner_prompt,
        max_completion_tokens=args.max_completion_tokens,
    )
    tuned_kernel = previous_kernels[-1]
    edits = extract_edits(tuner_output)
    for old_str, new_str in edits:
        tuned_kernel = str_replace(tuned_kernel, old_str, new_str)

    measure_performance = _use_performance_metric(args)
    tuned_metrics = wrapped_eval_kernel_against_ref(
        ref_arch_src,
        tuned_kernel,
        measure_performance=measure_performance,
        num_correct_trials=5,
        num_perf_trials=100 if measure_performance else 1,
        backend="triton",
        dtype_str=args.dtype_str,
        # Use the assigned GPU id (single int), not the list of available GPUs
        device=args.gpu_id,
        use_remote_eval=getattr(args, 'use_remote_eval', False),
        remote_eval_url=getattr(args, 'remote_eval_url', "http://127.0.0.1:12017"),
        test_source=getattr(args, 'test_source', 'KB'),
        level=getattr(args, 'level', None),
        problem_id=getattr(args, 'problem_id', None),
        gpu_name=getattr(args, 'gpu_name', None),
    )
    #print(f"Tuned Metrics: {tuned_metrics}")
    #print(f"Tuned Metrics 2: {tuned_metrics_2}")
    #assert tuned_metrics == tuned_metrics_2, f"Tuned Metrics: {tuned_metrics} != Tuned Metrics 2: {tuned_metrics_2}"

    # Construct combined prompt for small step (reviser_prompt + separator + tuner_prompt)
    prompt = reviser_prompt + "\n" + "-" * 100 + "\n" + tuner_prompt
    
    logs = {
        "reviser_prompt": reviser_prompt,
        "reviser_output": reviser_output,
        "tuner_prompt": tuner_prompt,
        "tuner_output": tuner_output,
        "tuned_kernel": tuned_kernel,
        "tuned_metrics": tuned_metrics,
        "prompt": prompt,
    }
    return tuned_kernel, tuned_metrics, logs

def single_large_step(
    ref_arch_src: str,
    inference_server: str,
    kernel_pool: list,
    metrics_pool: list,
    args: argparse.Namespace,
    *,
    context_ids: list[int] | None = None,
    elite_kernel_pool: list | None = None,
    elite_metrics_pool: list | None = None,
    elite_context_ids: list[int] | None = None,
):
    # previously experienced kernels and metrics
    pool_prompt = generate_pool_prompt_dual(
        kernel_pool=kernel_pool,
        metrics_pool=metrics_pool,
        kernel_pool_ids=context_ids,
        elite_kernel_pool=elite_kernel_pool,
        elite_metrics_pool=elite_metrics_pool,
        elite_pool_ids=elite_context_ids,
    )
    # Proposer Agent
    proposer_prompt = generate_proposer_prompt(
        task=args.test_source,
        task_params=args.task_params,
        experience_guidance_path=args.general_memory_path, 
        pool_prompt=pool_prompt,
        knowledge_1_threshold=args.knowledge_1_threshold,
    )
    proposer_output = query_inference_server(
        server=inference_server,
        model_name=args.model_name,
        prompt=proposer_prompt,
        max_completion_tokens=args.max_completion_tokens,
    )
    proposal_kernel = extract_first_code(proposer_output, ["python", "cpp"])
    measure_performance = _use_performance_metric(args)
    proposal_metrics = wrapped_eval_kernel_against_ref(
        ref_arch_src,
        proposal_kernel,
        measure_performance=measure_performance,
        num_correct_trials=5,
        num_perf_trials=100 if measure_performance else 1,
        backend="triton",
        dtype_str=args.dtype_str,
        # Use the assigned GPU id (single int), not the list of available GPUs
        device=args.gpu_id,
        use_remote_eval=getattr(args, 'use_remote_eval', False),
        remote_eval_url=getattr(args, 'remote_eval_url', "http://127.0.0.1:12017"),
        test_source=getattr(args, 'test_source', 'KB'),
        level=getattr(args, 'level', None),
        problem_id=getattr(args, 'problem_id', None),
        gpu_name=getattr(args, 'gpu_name', None),
    )
    # For large step, prompt is just the proposer_prompt
    prompt = proposer_prompt
    
    logs = {
        "proposer_prompt": proposer_prompt,
        "proposal_kernel": proposal_kernel,
        "proposal_metrics": proposal_metrics,
        "prompt": prompt,
    }
    return proposal_kernel, proposal_metrics, logs

def dummy_small_step(ref_arch_src: str, inference_server: str, previous_kernels: list, previous_metrics: list, args: argparse.Namespace):
    """
    Dummy small step that does nothing, returns empty kernel and zero metrics.
    """
    from src.eval import KernelExecResult
    
    empty_kernel = ""
    zero_metrics = KernelExecResult(
        compiled=False,
        correctness=False,
        metadata={},
        runtime=-1.0,
        runtime_stats={}
    )
    logs = {
        "dummy": True,
        "action": "dummy_small_step",
        "prompt": "",
    }
    return empty_kernel, zero_metrics, logs

def dummy_large_step(ref_arch_src: str, inference_server: str, kernel_pool: list, metrics_pool: list, args: argparse.Namespace):
    """
    Dummy large step that does nothing, returns empty kernel and zero metrics.
    """
    from src.eval import KernelExecResult
    
    empty_kernel = ""
    zero_metrics = KernelExecResult(
        compiled=False,
        correctness=False,
        metadata={},
        runtime=-1.0,
        runtime_stats={}
    )
    logs = {
        "dummy": True,
        "action": "dummy_large_step",
        "prompt": "",
    }
    return empty_kernel, zero_metrics, logs
