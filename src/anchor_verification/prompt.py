from dataclasses import dataclass

@dataclass
class PromptManager:
    instruction_prompt = """
Given the program below, improve its performance:

### Program:
{slow_code}

### Optimized Version:
"""


    cot_prompt: str = """
Given the program, generate an efficiency improvement strategy to enhance its performance.

### slower program:
{slow_code}

### strategy:
LLMs generated potential strategy."""


    inference_prompt: str = """
Given the program below, improve its performance:

### Program:
{slow_code}

### Optimized Version:
"""


    anchor_testcase_input = """
Given the program below, please explain and analyze its functionality, and provide 3 testcase inputs that fully consider boundary conditions and code coverage. Note that only the testcase inputs are required.

### Program:
{slow_code}

### Explanation:
{Your explanation here}

### Test case Inputs:
{Your testcase inputs}
"""


    anchor_refinment = """
You are a code expert, and your task is to correct the functionally incorrect code based on test cases and execution feedback. Analyze the issues, apply the necessary fixes, and ensure the corrected code meets the expected functionality and pass the testcase.

### Incorrect Program:
{code}

### Explanation:
{explanation}

### Testcase:
{Testcase}

### Feedback from execution:
{Feedback}

### Your corrected code version:
"""

    selfdebug_prompt = """
Given the program below, improve its performance:

### Program:
{slow_code}

### Optimized Version:
"""

    direct_testcase_prompt = """
Given the program below, please explain and analyze its functionality, and generate three comprehensive test cases that thoroughly cover boundary conditions and all code paths. Each testcase should include the input and the corresponding expected output.

### Program:
{slow_code}

### Explanation:
{Your explanation here}

### Test case:
{Your testcase}
"""
    
    
    def get_prompt(self, prompt_name):
        return self._prompts.get(prompt_name, "Prompt not found")