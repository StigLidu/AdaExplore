def run_info_to_prompt(run_info: dict) -> str:
    prompt = f"""
    Run Info:
    {run_info}
    """
    return prompt