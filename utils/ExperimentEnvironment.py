from io import StringIO
import sys
import traceback
import re
from typing import List
from IPython import embed


from maia_api import System, Tools

class ExperimentEnvironment:
    '''Executes and stores the variables of maia's experiments'''

    def __init__(self, system: System, tools: Tools, global_vars: dict):
        self.system = system
        self.tools = tools
        self.experiment_vars = global_vars

        self.experiment_vars["system"] = system
        self.experiment_vars["tools"] = tools

    # Parse maia's code
    def get_code(self, maia_experiment: str)->List[str]:
        '''Parses code from maia's experiment. There may be multiple code blocks.
        A block is defined by "```python ```"."
        
        '''
        # Extract the code blocks
        pattern = r"```python(.*?)```"
        unstripped_maia_code = re.findall(pattern, maia_experiment, re.DOTALL)
        # Remove leading and trailing whitespaces
        maia_code = [code.strip() for code in unstripped_maia_code]

        if len(maia_code) == 0:
            raise ValueError("No code blocks found in the experiment.")
        
        return maia_code

    # Run the code on python
    def execute_experiment(self, maia_experiment: str)->str:
        code_blocks = self.get_code(maia_experiment)

        out = StringIO()
        err = StringIO()
        # Store original stdout and stderr
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        for code in code_blocks:
            try:
                # Redirect stdout and stderr
                sys.stdout = out
                sys.stderr = err
                
                # Execute the code with the system and tools objects, as well as any
                # variables defined in previous experiments
                exec(compile(code, 'code', 'exec'), globals(), self.experiment_vars)
            except Exception as e:
                # Capture traceback for exceptions 
                traceback.print_exc(file=err)
                # Output error
                err.write(str(e))
                # Stop execution
                break

        # Restore original stdout and stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        # Get the captured output
        output = ""
        if out.getvalue() != "":
            output += f"Standard Output:\n{out.getvalue()}"
        if err.getvalue() != "":
            output += f"\n\nStandard Error:\n{err.getvalue()}"

        return output