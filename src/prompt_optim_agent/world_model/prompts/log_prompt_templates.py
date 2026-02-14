# log prompts

forward_log_template = """---------------\tforward\t----------------
cur_prompt:\n{cur_prompt}
labels:  {labels}
preds:   {preds}
correct: {correct}
acc:     {acc}
"""

error_string_log_template = """--------------\terror_string\t-----------------
{error_string}
"""

correct_string_log_template = """--------------\tcorrect_string\t-----------------
{correct_string}
"""

gradient_log_template = """---------------\tcal_gradient\t----------------
gradient_prompt:\n{gradient_prompt}
gradient:\n{gradient}
"""

gradient_summary_template = """---------------\tsummary_feedback_template\t----------------
summary_prompt:\n{prompt}
summary:\n{summary}
"""

correct_gradient_log_template = """---------------\tcal_correct_gradient\t----------------
gradient_prompt:\n{gradient_prompt}
gradient:\n{gradient}
"""

optimize_log_template = """-------------\toptimize\t---------------
optimize_prompt:\n{optimize_prompt}
response:\n{response}
optimized_prompt:\n{optimized_prompt}
"""

self_evaluate_log_template = """-------------\tself_eval\t---------------
self_eval_prompt:\n{self_eval_prompt}
response:\n{response}
self_eval_score:\n{self_eval_score}
"""
