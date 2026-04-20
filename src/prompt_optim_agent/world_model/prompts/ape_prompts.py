# Prompt templates for APE (Automatic Prompt Engineer)
# Based on: Zhou et al., "Large Language Models Are Human-Level Prompt Engineers" (ICLR 2023)
# https://arxiv.org/abs/2211.01910

# --- Generation Templates ---
# These are used to ask the LLM to infer an instruction from input-output demonstrations.

forward_generation_template = """I gave a friend an instruction and five inputs. The friend read the instruction and wrote an output for every one of the inputs.\nHere are the input-output pairs:\n\n{demos}\n\nThe instruction was """

insert_generation_template = """I instructed my friend to {instruction_placeholder}. The friend read the instruction and wrote an output for every one of the inputs.\nHere are the input-output pairs:\n\n{demos}"""

insert_generation_template_v2 = """Professor Smith was given the following instructions: {instruction_placeholder}\nHere are the Professor's responses:\n\n{demos}"""

# --- Demo Formatting ---

demo_template = "Input: {input}\nOutput: {output}"

demo_template_qa = "Q: {input}\nA: {output}"
