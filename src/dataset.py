################################################################################
# Helpers for Dataset
################################################################################

import os
import random
import re
import hashlib
import json
from src.utils import read_file

REPO_TOP_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)
KERNEL_BENCH_PATH = os.path.join(REPO_TOP_PATH, "datasets", "KernelBench")
SYNTHESIZED_DATA_PATH = os.path.join(REPO_TOP_PATH, "datasets", "KernelBench_syn")
FLASHINFER_TRACE_PATH = os.path.join(REPO_TOP_PATH, "datasets", "flashinfer-trace")
MLSYS26_CONTEST_PATH = os.path.join(REPO_TOP_PATH, "datasets", "mlsys26-contest")
TRITONBENCH_PATH = os.path.join(REPO_TOP_PATH, "datasets", "TritonBench")
TRITONBENCH_G_PATH = os.path.join(TRITONBENCH_PATH, "data", "TritonBench_G_v1")
TRITONBENCH_G_FLAT_PATH = os.path.join(REPO_TOP_PATH, "datasets", "TritonBench_G_v1")


def assign_problem_hash(problem_path: str) -> list[int]:
    """
    Assign a unique hash to a problem in the dataset
    """
    with open(problem_path, "r") as f:
        problem_src = f.read()
    return get_code_hash(problem_src)


def get_code_hash(problem_src: str) -> str:
    """
    Assign a unique hash to some piece of code
    Important to strip out the comments and whitespace as they are not functionally part of the code
    """
    # Remove multi-line comments first
    problem_src = re.sub(r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'', "", problem_src)
    # Remove inline comments and all whitespace
    cleaned_problem_src = re.sub(r"#.*$|\s+", "", problem_src, flags=re.MULTILINE)
    # hash only on code
    return hashlib.md5(cleaned_problem_src.encode()).hexdigest()


def construct_problem_dataset_from_problem_dir(problem_dir: str) -> list[str]:
    """
    Construct a list of relative paths to all the python files in the problem directory
    Sorted by the numerical prefix of the filenames
    """
    DATASET = []

    for file_name in os.listdir(problem_dir):
        if file_name.endswith(".py"):
            # TODO: revisit later to satisfy eval harnes
            relative_path = os.path.join(problem_dir, file_name)
            DATASET.append(relative_path)

    # Sort the DATASET based on the numerical prefix of the filenames
    DATASET.sort(key=lambda x: int(os.path.basename(x).split("_")[0].split(".")[0]))

    return DATASET


def construct_kernelbench_dataset(level: int, suffix: str = "") -> list[str]:
    return construct_problem_dataset_from_problem_dir(
        os.path.join(KERNEL_BENCH_PATH + suffix, f"level{level}")
    )


def construct_synthesized_data_dataset(version: int) -> list[str]:
    return construct_problem_dataset_from_problem_dir(
        os.path.join(SYNTHESIZED_DATA_PATH, f"syn_v{version}")
    )


def get_tritonbench_g_dataset_root() -> str:
    """
    Return TritonBench-G dataset root path.

    Resolution order:
    1. $TRITONBENCH_G_PATH
    2. datasets/TritonBench/data/TritonBench_G_v1
    3. datasets/TritonBench_G_v1
    """
    candidates = []
    env_root = os.environ.get("TRITONBENCH_G_PATH")
    if env_root:
        candidates.append(env_root)
    candidates.extend([TRITONBENCH_G_PATH, TRITONBENCH_G_FLAT_PATH])

    for root in candidates:
        if os.path.isdir(root):
            return root
    # Return the canonical default path even if it does not exist; caller can raise a clear error.
    return TRITONBENCH_G_PATH


def construct_tritonbench_g_dataset(dataset_root: str | None = None) -> list[str]:
    """
    Construct sorted list of TritonBench-G python task files.
    """
    root = dataset_root or get_tritonbench_g_dataset_root()
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"TritonBench-G dataset root does not exist: {root}. "
            "Set TRITONBENCH_G_PATH or place data under datasets/TritonBench/data/TritonBench_G_v1."
        )
    return sorted(os.path.join(root, f) for f in os.listdir(root) if f.endswith(".py"))


def resolve_tritonbench_g_problem_path(
    problem_id: str, dataset_root: str | None = None
) -> str:
    """
    Resolve TritonBench-G problem_id to concrete .py path.

    Supports:
    - 1-indexed integer id (e.g. "1")
    - filename with or without suffix (e.g. "add_example" / "add_example.py")
    """
    dataset = construct_tritonbench_g_dataset(dataset_root=dataset_root)
    pid = str(problem_id).strip()

    if pid.isdigit():
        idx = int(pid) - 1
        if idx < 0 or idx >= len(dataset):
            raise ValueError(
                f"TritonBench-G problem_id out of range: {problem_id}. "
                f"Valid range is 1..{len(dataset)}"
            )
        return dataset[idx]

    filename = pid if pid.endswith(".py") else f"{pid}.py"
    for path in dataset:
        if os.path.basename(path) == filename:
            return path

    raise ValueError(f"TritonBench-G problem not found: {problem_id}")


KERNELBENCH_LEVEL_1_DATASET = construct_kernelbench_dataset(level=1)
KERNELBENCH_LEVEL_2_DATASET = construct_kernelbench_dataset(level=2)
KERNELBENCH_LEVEL_3_DATASET = construct_kernelbench_dataset(level=3)

################################################################################
# Eval on Subsets of KernelBench
################################################################################


def get_kernelbench_subset(
    level: int, num_subset_problems: int = 10, random_seed: int = 42
) -> tuple[list[str], list[int]]:
    """
    Get a random subset of problems from the KernelBench dataset
    """

    full_dataset = construct_kernelbench_dataset(level)

    random.seed(random_seed)
    num_subset_problems = min(num_subset_problems, len(full_dataset))
    subset_indices = random.sample(range(len(full_dataset)), num_subset_problems)

    subset = sorted([full_dataset[i] for i in subset_indices])
    return subset, subset_indices


################################################################################
# Representative subsets of KernelBench
# use this if you want to iterate on methods without the hassle of running the full dataset
# problem_ids are 1-indexed (logical index)
################################################################################

level1_representative_subset = [
    "1_Square_matrix_multiplication_.py",
    "3_Batched_matrix_multiplication.py",
    "6_Matmul_with_large_K_dimension_.py",
    "18_Matmul_with_transposed_both.py",
    "23_Softmax.py",
    "26_GELU_.py",
    "33_BatchNorm.py",
    "36_RMSNorm_.py",
    "40_LayerNorm.py",
    "42_Max_Pooling_2D.py",
    "48_Mean_reduction_over_a_dimension.py",
    "54_conv_standard_3D__square_input__square_kernel.py",
    "57_conv_transposed_2D__square_input__square_kernel.py",
    "65_conv_transposed_2D__square_input__asymmetric_kernel.py",
    "77_conv_transposed_3D_square_input_square_kernel___padded____dilated____strided__.py",
    "82_conv_depthwise_2D_square_input_square_kernel.py",
    "86_conv_depthwise_separable_2D.py",
    "87_conv_pointwise_2D.py",
]

level1_representative_subset_problem_ids = [
    1,
    3,
    6,
    18,
    23,
    26,
    33,
    36,
    40,
    42,
    48,
    54,
    57,
    65,
    77,
    82,
    86,
    87,
]

level2_representative_subset = [
    "1_Conv2D_ReLU_BiasAdd.py",
    "2_ConvTranspose2d_BiasAdd_Clamp_Scaling_Clamp_Divide.py",
    "8_Conv3d_Divide_Max_GlobalAvgPool_BiasAdd_Sum.py",
    "18_Matmul_Sum_Max_AvgPool_LogSumExp_LogSumExp.py",
    "23_Conv3d_GroupNorm_Mean.py",
    "28_BMM_InstanceNorm_Sum_ResidualAdd_Multiply.py",
    "33_Gemm_Scale_BatchNorm.py",
    "43_Conv3d_Max_LogSumExp_ReLU.py",
]

level2_representative_subset_problem_ids = [1, 2, 8, 18, 23, 28, 33, 43]

level3_representative_subset = [
    "1_MLP.py",
    "5_AlexNet.py",
    "8_ResNetBasicBlock.py",
    "11_VGG16.py",
    "20_MobileNetV2.py",
    "21_EfficientNetMBConv.py",
    "33_VanillaRNN.py",
    "38_LTSMBidirectional.py",
    "43_MinGPTCausalAttention.py",
]

level3_representative_subset_problem_ids = [1, 5, 8, 11, 20, 33, 38, 43]


################################################################################
# FlashInfer Trace Dataset
################################################################################

FLASHINFER_TRACE_OP_TYPES = [
    "gemm",
    "gqa_paged",
    "gqa_ragged",
    "mla_paged",
    "moe",
    "rmsnorm",
    "sampling",
]

# Test-source aliases that use FlashInfer-style datasets.
FLASHINFER_STYLE_DATASET_ROOTS = {
    "FIT": FLASHINFER_TRACE_PATH,
    "MLSYS": MLSYS26_CONTEST_PATH,
}


def get_flashinfer_style_dataset_root(test_source: str) -> str:
    """Return dataset root path for FlashInfer-style datasets (FIT/MLSYS)."""
    return FLASHINFER_STYLE_DATASET_ROOTS.get(test_source, FLASHINFER_TRACE_PATH)


def construct_flashinfer_trace_dataset(
    op_type: str, dataset_root: str = FLASHINFER_TRACE_PATH
) -> list[str]:
    """Return sorted list of problem names for given op_type."""
    op_dir = os.path.join(dataset_root, "definitions", op_type)
    return sorted(f[:-5] for f in os.listdir(op_dir) if f.endswith(".json"))


def load_flashinfer_trace_definition(
    op_type: str,
    problem_name: str,
    dataset_root: str = FLASHINFER_TRACE_PATH,
) -> dict:
    """Load definition JSON for given op_type and problem_name."""
    path = os.path.join(dataset_root, "definitions", op_type, f"{problem_name}.json")
    with open(path, "r") as f:
        return json.load(f)


def flashinfer_trace_definition_to_ref_arch_src(definition: dict) -> str:
    """Return the reference implementation from definition."""
    return definition.get("reference", "")
