from abc import ABC, abstractmethod
import os
import json
from functools import lru_cache

from agentless.repair.repair import construct_topn_file_context
from agentless.util.compress_file import get_skeleton
from agentless.util.postprocess_data import extract_code_blocks, extract_locs_for_files
from agentless.util.preprocess_data import (
    correct_file_paths,
    get_full_file_paths_and_classes_and_functions,
    get_repo_files,
    line_wrap_content,
    show_project_structure,
)

MAX_CONTEXT_LENGTH = 128000


class FL(ABC):
    def __init__(self, instance_id, structure, problem_statement, **kwargs):
        self.structure = structure
        self.instance_id = instance_id
        self.problem_statement = problem_statement

    @abstractmethod
    def localize(self, top_n=1, mock=False) -> tuple[list, list, list, any]:
        pass


class LLMFL(FL):
    # File description cache to store descriptions by repository and file path
    file_descriptions_cache = {}
    
    obtain_relevant_files_prompt = """
Please look through the following GitHub problem description and Repository structure and provide a list of files that one would need to edit to fix the problem.

### GitHub Problem Description ###
{problem_statement}

###

### Repository Structure with File Descriptions ###
{structure_with_descriptions}

###

Please only provide the full path and return at most 5 files.
The returned files should be separated by new lines ordered by most to least important and wrapped with ```
For example:
```
file1.py
file2.py
```
"""

    obtain_irrelevant_files_prompt = """
Please look through the following GitHub problem description and Repository structure and provide a list of folders that are irrelevant to fixing the problem.
Note that irrelevant folders are those that do not need to be modified and are safe to ignored when trying to solve this problem.

### GitHub Problem Description ###
{problem_statement}

###

### Repository Structure with File Descriptions ###
{structure_with_descriptions}

###

Please only provide the full path.
Remember that any subfolders will be considered as irrelevant if you provide the parent folder.
Please ensure that the provided irrelevant folders do not include any important files needed to fix the problem
The returned folders should be separated by new lines and wrapped with ```
For example:
```
folder1/
folder2/folder3/
folder4/folder5/
```
"""

    file_content_template = """
### File: {file_name} ###
{file_content}
"""
    file_content_in_block_template = """
### File: {file_name} ###
```python
{file_content}
```
"""

    generate_file_description_prompt = """
Please provide a concise two-sentence description of the following Python file based on its content. The description should summarize the purpose and main functionality of the file.

```python
{file_content}
```
"""

    obtain_relevant_code_combine_top_n_prompt = """
Please review the following GitHub problem description and relevant files, and provide a set of locations that need to be edited to fix the issue.
The locations can be specified as class names, function or method names, or exact line numbers that require modification.

### GitHub Problem Description ###
{problem_statement}

###
{file_contents}

###

Please provide the class name, function or method name, or the exact line numbers that need to be edited.
The possible location outputs should be either "class", "function" or "line".

### Examples:
```
full_path1/file1.py
line: 10
class: MyClass1
line: 51

full_path2/file2.py
function: MyClass2.my_method
line: 12

full_path3/file3.py
function: my_function
line: 24
line: 156
```

Return just the location(s) wrapped with ```.
"""

    obtain_relevant_code_combine_top_n_no_line_number_prompt = """
Please review the following GitHub problem description and relevant files, and provide a set of locations that need to be edited to fix the issue.
The locations can be specified as class, method, or function names that require modification.

### GitHub Problem Description ###
{problem_statement}

###
{file_contents}

###

Please provide the class, method, or function names that need to be edited.
### Examples:
```
full_path1/file1.py
function: my_function1
class: MyClass1

full_path2/file2.py
function: MyClass2.my_method
class: MyClass3

full_path3/file3.py
function: my_function2
```

Return just the location(s) wrapped with ```.
"""
    obtain_relevant_functions_and_vars_from_compressed_files_prompt_more = """
Please look through the following GitHub Problem Description and the Skeleton of Relevant Files.
Identify all locations that need inspection or editing to fix the problem, including directly related areas as well as any potentially related global variables, functions, and classes.
For each location you provide, either give the name of the class, the name of a method in a class, the name of a function, or the name of a global variable.

### GitHub Problem Description ###
{problem_statement}

### Skeleton of Relevant Files ###
{file_contents}

###

Please provide the complete set of locations as either a class name, a function name, or a variable name.
Note that if you include a class, you do not need to list its specific methods.
You can include either the entire class or don't include the class name and instead include specific methods in the class.
### Examples:
```
full_path1/file1.py
function: my_function_1
class: MyClass1
function: MyClass2.my_method

full_path2/file2.py
variable: my_var
function: MyClass3.my_method

full_path3/file3.py
function: my_function_2
function: my_function_3
function: MyClass4.my_method_1
class: MyClass5
```

Return just the locations wrapped with ```.
"""

    obtain_relevant_functions_and_vars_from_raw_files_prompt = """
Please look through the following GitHub Problem Description and Relevant Files.
Identify all locations that need inspection or editing to fix the problem, including directly related areas as well as any potentially related global variables, functions, and classes.
For each location you provide, either give the name of the class, the name of a method in a class, the name of a function, or the name of a global variable.

### GitHub Problem Description ###
{problem_statement}

### Relevant Files ###
{file_contents}

###

Please provide the complete set of locations as either a class name, a function name, or a variable name.
Note that if you include a class, you do not need to list its specific methods.
You can include either the entire class or don't include the class name and instead include specific methods in the class.
### Examples:
```
full_path1/file1.py
function: my_function_1
class: MyClass1
function: MyClass2.my_method

full_path2/file2.py
variable: my_var
function: MyClass3.my_method

full_path3/file3.py
function: my_function_2
function: my_function_3
function: MyClass4.my_method_1
class: MyClass5
```

Return just the locations wrapped with ```.
"""

    def __init__(
        self,
        instance_id,
        structure,
        problem_statement,
        model_name,
        backend,
        logger,
        **kwargs,
    ):
        super().__init__(instance_id, structure, problem_statement)
        self.max_tokens = 300
        self.model_name = model_name
        self.backend = backend
        self.logger = logger

        # Short Description Model 
        from agentless.util.model import make_model
        self.desc_model = make_model(
            model=self.model_name,
            backend=self.backend,
            logger=self.logger,
            max_tokens=100,  # Short description
            temperature=0,
            batch_size=1,
        )
        
    def _get_file_description_cache_key(self, repo, file_path):
        """Generate a unique key for caching file descriptions."""
        return f"{repo}:{file_path}"
        
    def _generate_file_description(self, file_path, file_content):
        """Generate a description for a file using the language model."""
        
        self.logger.info(f"Generating description for file: {file_path}")
        
        message = self.generate_file_description_prompt.format(
            file_content=file_content
        ).strip()
        
        traj = self.desc_model.codegen(message, num_samples=1)[0]
        description = traj["response"].strip()
        
        # Ensure the description is not too long
        if len(description.split('.')) > 2:
            sentences = description.split('.')
            description = '.'.join(sentences[:2]) + '.'
            
        return description
    
    def _get_file_description(self, repo, file_path, file_content):
        """Get a file description, either from cache or by generating it."""
        cache_key = self._get_file_description_cache_key(repo, file_path)
        
        # Check if description exists in the cache
        if cache_key in self.file_descriptions_cache:
            self.logger.info(f"Using cached description for {file_path}")
            return self.file_descriptions_cache[cache_key]
        
        # Generate a description
        description = self._generate_file_description(file_path, file_content)
        
        # Cache the description
        self.file_descriptions_cache[cache_key] = description
        
        # Optionally save to disk for persistence
        cache_dir = os.path.join("cache", "file_descriptions")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, "descriptions.json")
        
        try:
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    disk_cache = json.load(f)
            else:
                disk_cache = {}
                
            disk_cache[cache_key] = description
            
            with open(cache_file, 'w') as f:
                json.dump(disk_cache, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save file description to disk: {e}")
        
        return description
    
    def _build_structure_with_descriptions(self, structure):
        """Add file descriptions to the repository structure representation."""
        import re
        
        # Try to extract repo name from instance_id for cache key
        repo = self.instance_id.split('__')[0] if '__' in self.instance_id else self.instance_id
        
        # Get all files in the repository
        files, _, _ = get_full_file_paths_and_classes_and_functions(structure)
        
        # Basic structure representation
        basic_structure = show_project_structure(structure).strip()
        
        # Enhance with descriptions
        structure_with_descriptions = basic_structure
        
        for file_content in files:
            file_path = file_content[0]
            content = "\n".join(file_content[1])
            
            if file_path.endswith(".py"):  # Only process Python files
                description = self._get_file_description(repo, file_path, content)
                
                # Add the description after the file name in the structure
                file_name = file_path.split('/')[-1]
                file_name_pattern = r'(\s+' + re.escape(file_name) + r')(\n)'
                replacement = r'\1 - ' + description + r'\2'
                structure_with_descriptions = re.sub(
                    file_name_pattern, 
                    replacement, 
                    structure_with_descriptions
                )
        
        return structure_with_descriptions

    def _parse_model_return_lines(self, content: str) -> list[str]:
        if content:
            return content.strip().split("\n")

    def localize_irrelevant(self, top_n=1, mock=False):
        from agentless.util.api_requests import num_tokens_from_messages
        from agentless.util.model import make_model

        structure_with_descriptions = self._build_structure_with_descriptions(self.structure)
        
        message = self.obtain_irrelevant_files_prompt.format(
            problem_statement=self.problem_statement,
            structure_with_descriptions=structure_with_descriptions,
        ).strip()
        self.logger.info(f"prompting with message:\n{message}")
        self.logger.info("=" * 80)

        if mock:
            self.logger.info("Skipping querying model since mock=True")
            traj = {
                "prompt": message,
                "usage": {
                    "prompt_tokens": num_tokens_from_messages(message, self.model),
                },
            }
            return [], {"raw_output_loc": ""}, traj

        model = make_model(
            model=self.model_name,
            backend=self.backend,
            logger=self.logger,
            max_tokens=2048,  # self.max_tokens,
            temperature=0,
            batch_size=1,
        )
        traj = model.codegen(message, num_samples=1)[0]
        traj["prompt"] = message
        raw_output = traj["response"]

        files, classes, functions = get_full_file_paths_and_classes_and_functions(
            self.structure
        )

        f_files = []
        filtered_files = []

        model_identified_files_folder = self._parse_model_return_lines(raw_output)
        # remove any none folder none files
        model_identified_files_folder = [
            x
            for x in model_identified_files_folder
            if x.endswith("/") or x.endswith(".py")
        ]

        for file_content in files:
            file_name = file_content[0]
            if any([file_name.startswith(x) for x in model_identified_files_folder]):
                filtered_files.append(file_name)
            else:
                f_files.append(file_name)

        self.logger.info(raw_output)

        return (
            f_files,
            {
                "raw_output_files": raw_output,
                "found_files": f_files,
                "filtered_files": filtered_files,
            },
            traj,
        )

    def localize(self, top_n=1, mock=False) -> tuple[list, list, list, any]:
        from agentless.util.api_requests import num_tokens_from_messages
        from agentless.util.model import make_model

        found_files = []

        structure_with_descriptions = self._build_structure_with_descriptions(self.structure)
        
        message = self.obtain_relevant_files_prompt.format(
            problem_statement=self.problem_statement,
            structure_with_descriptions=structure_with_descriptions,
        ).strip()
        self.logger.info(f"prompting with message:\n{message}")
        self.logger.info("=" * 80)
        if mock:
            self.logger.info("Skipping querying model since mock=True")
            traj = {
                "prompt": message,
                "usage": {
                    "prompt_tokens": num_tokens_from_messages(message, self.model_name),
                },
            }
            return [], {"raw_output_loc": ""}, traj

        model = make_model(
            model=self.model_name,
            backend=self.backend,
            logger=self.logger,
            max_tokens=self.max_tokens,
            temperature=0,
            batch_size=1,
        )
        traj = model.codegen(message, num_samples=1)[0]
        traj["prompt"] = message
        raw_output = traj["response"]
        model_found_files = self._parse_model_return_lines(raw_output)

        files, classes, functions = get_full_file_paths_and_classes_and_functions(
            self.structure
        )

        # sort based on order of appearance in model_found_files
        found_files = correct_file_paths(model_found_files, files)

        self.logger.info(raw_output)

        return (
            found_files,
            {"raw_output_files": raw_output},
            traj,
        )

    def localize_function_from_compressed_files(
        self,
        file_names,
        mock=False,
        temperature=0.0,
        keep_old_order=False,
        compress_assign: bool = False,
        total_lines=30,
        prefix_lines=10,
        suffix_lines=10,
    ):
        from agentless.util.api_requests import num_tokens_from_messages
        from agentless.util.model import make_model

        file_contents = get_repo_files(self.structure, file_names)
        compressed_file_contents = {
            fn: get_skeleton(
                code,
                compress_assign=compress_assign,
                total_lines=total_lines,
                prefix_lines=prefix_lines,
                suffix_lines=suffix_lines,
            )
            for fn, code in file_contents.items()
        }
        contents = [
            self.file_content_in_block_template.format(file_name=fn, file_content=code)
            for fn, code in compressed_file_contents.items()
        ]
        file_contents = "".join(contents)
        template = (
            self.obtain_relevant_functions_and_vars_from_compressed_files_prompt_more
        )
        message = template.format(
            problem_statement=self.problem_statement, file_contents=file_contents
        )
        self.logger.info(f"prompting with message:")
        self.logger.info("\n" + message)

        def message_too_long(message):
            return (
                num_tokens_from_messages(message, self.model_name) >= MAX_CONTEXT_LENGTH
            )

        while message_too_long(message) and len(contents) > 1:
            self.logger.info(f"reducing to \n{len(contents)} files")
            contents = contents[:-1]
            file_contents = "".join(contents)
            message = template.format(
                problem_statement=self.problem_statement, file_contents=file_contents
            )  # Recreate message

        if message_too_long(message):
            raise ValueError(
                "The remaining file content is too long to fit within the context length"
            )
        self.logger.info(f"prompting with message:\n{message}")
        self.logger.info("=" * 80)

        if mock:
            self.logger.info("Skipping querying model since mock=True")
            traj = {
                "prompt": message,
                "usage": {
                    "prompt_tokens": num_tokens_from_messages(
                        message,
                        self.model_name,
                    ),
                },
            }
            return {}, {"raw_output_loc": ""}, traj

        model = make_model(
            model=self.model_name,
            backend=self.backend,
            logger=self.logger,
            max_tokens=self.max_tokens,
            temperature=temperature,
            batch_size=1,
        )
        traj = model.codegen(message, num_samples=1)[0]
        traj["prompt"] = message
        raw_output = traj["response"]

        model_found_locs = extract_code_blocks(raw_output)
        model_found_locs_separated = extract_locs_for_files(
            model_found_locs, file_names, keep_old_order
        )

        self.logger.info(f"==== raw output ====")
        self.logger.info(raw_output)
        self.logger.info("=" * 80)
        self.logger.info(f"==== extracted locs ====")
        for loc in model_found_locs_separated:
            self.logger.info(loc)
        self.logger.info("=" * 80)

        return model_found_locs_separated, {"raw_output_loc": raw_output}, traj

    def localize_function_from_raw_text(
        self,
        file_names,
        mock=False,
        temperature=0.0,
        keep_old_order=False,
    ):
        from agentless.util.api_requests import num_tokens_from_messages
        from agentless.util.model import make_model

        file_contents = get_repo_files(self.structure, file_names)
        raw_file_contents = {fn: code for fn, code in file_contents.items()}
        contents = [
            self.file_content_template.format(file_name=fn, file_content=code)
            for fn, code in raw_file_contents.items()
        ]
        file_contents = "".join(contents)
        template = self.obtain_relevant_functions_and_vars_from_raw_files_prompt
        message = template.format(
            problem_statement=self.problem_statement, file_contents=file_contents
        )
        self.logger.info(f"prompting with message:")
        self.logger.info("\n" + message)

        def message_too_long(message):
            return (
                num_tokens_from_messages(message, self.model_name) >= MAX_CONTEXT_LENGTH
            )

        while message_too_long(message) and len(contents) > 1:
            self.logger.info(f"reducing to \n{len(contents)} files")
            contents = contents[:-1]
            file_contents = "".join(contents)
            message = template.format(
                problem_statement=self.problem_statement, file_contents=file_contents
            )  # Recreate message

        if message_too_long(message):
            raise ValueError(
                "The remaining file content is too long to fit within the context length"
            )
        self.logger.info(f"prompting with message:\n{message}")
        self.logger.info("=" * 80)

        if mock:
            self.logger.info("Skipping querying model since mock=True")
            traj = {
                "prompt": message,
                "usage": {
                    "prompt_tokens": num_tokens_from_messages(
                        message,
                        self.model_name,
                    ),
                },
            }
            return {}, {"raw_output_loc": ""}, traj

        model = make_model(
            model=self.model_name,
            backend=self.backend,
            logger=self.logger,
            max_tokens=self.max_tokens,
            temperature=temperature,
            batch_size=1,
        )
        traj = model.codegen(message, num_samples=1)[0]
        traj["prompt"] = message
        raw_output = traj["response"]

        model_found_locs = extract_code_blocks(raw_output)
        model_found_locs_separated = extract_locs_for_files(
            model_found_locs, file_names, keep_old_order
        )

        self.logger.info(f"==== raw output ====")
        self.logger.info(raw_output)
        self.logger.info("=" * 80)
        self.logger.info(f"==== extracted locs ====")
        for loc in model_found_locs_separated:
            self.logger.info(loc)
        self.logger.info("=" * 80)

        return model_found_locs_separated, {"raw_output_loc": raw_output}, traj

    def localize_line_from_coarse_function_locs(
        self,
        file_names,
        coarse_locs,
        context_window: int,
        add_space: bool,
        sticky_scroll: bool,
        no_line_number: bool,
        temperature: float = 0.0,
        num_samples: int = 1,
        mock=False,
        keep_old_order=False,
    ):
        from agentless.util.api_requests import num_tokens_from_messages
        from agentless.util.model import make_model

        file_contents = get_repo_files(self.structure, file_names)
        topn_content, file_loc_intervals = construct_topn_file_context(
            coarse_locs,
            file_names,
            file_contents,
            self.structure,
            context_window=context_window,
            loc_interval=True,
            add_space=add_space,
            sticky_scroll=sticky_scroll,
            no_line_number=no_line_number,
        )
        if no_line_number:
            template = self.obtain_relevant_code_combine_top_n_no_line_number_prompt
        else:
            template = self.obtain_relevant_code_combine_top_n_prompt
        message = template.format(
            problem_statement=self.problem_statement, file_contents=topn_content
        )
        self.logger.info(f"prompting with message:\n{message}")
        self.logger.info("=" * 80)

        def message_too_long(message):
            return (
                num_tokens_from_messages(message, self.model_name) >= MAX_CONTEXT_LENGTH
            )

        while message_too_long(message) and len(coarse_locs) > 1:
            self.logger.info(f"reducing to \n{len(coarse_locs)} files")
            coarse_locs.popitem()
            topn_content, file_loc_intervals = construct_topn_file_context(
                coarse_locs,
                file_names,
                file_contents,
                self.structure,
                context_window=context_window,
                loc_interval=True,
                add_space=add_space,
                sticky_scroll=sticky_scroll,
                no_line_number=no_line_number,
            )
            message = template.format(
                problem_statement=self.problem_statement, file_contents=topn_content
            )

        if message_too_long(message):
            raise ValueError(
                "The remaining file content is too long to fit within the context length"
            )

        if mock:
            self.logger.info("Skipping querying model since mock=True")
            traj = {
                "prompt": message,
                "usage": {
                    "prompt_tokens": num_tokens_from_messages(message, self.model_name),
                },
            }
            return [], {"raw_output_loc": ""}, traj

        model = make_model(
            model=self.model_name,
            backend=self.backend,
            logger=self.logger,
            max_tokens=self.max_tokens,
            temperature=temperature,
            batch_size=num_samples,
        )
        raw_trajs = model.codegen(
            message, num_samples=num_samples, prompt_cache=num_samples > 1
        )

        # Merge trajectories
        raw_outputs = [raw_traj["response"] for raw_traj in raw_trajs]
        traj = {
            "prompt": message,
            "response": raw_outputs,
            "usage": {  # merge token usage
                "completion_tokens": sum(
                    raw_traj["usage"]["completion_tokens"] for raw_traj in raw_trajs
                ),
                "prompt_tokens": sum(
                    raw_traj["usage"]["prompt_tokens"] for raw_traj in raw_trajs
                ),
            },
        }
        model_found_locs_separated_in_samples = []
        for raw_output in raw_outputs:
            model_found_locs = extract_code_blocks(raw_output)
            model_found_locs_separated = extract_locs_for_files(
                model_found_locs, file_names, keep_old_order
            )
            model_found_locs_separated_in_samples.append(model_found_locs_separated)

            self.logger.info(f"==== raw output ====")
            self.logger.info(raw_output)
            self.logger.info("=" * 80)
            self.logger.info(f"==== extracted locs ====")
            for loc in model_found_locs_separated:
                self.logger.info(loc)
            self.logger.info("=" * 80)
        self.logger.info("==== Input coarse_locs")
        coarse_info = ""
        for fn, found_locs in coarse_locs.items():
            coarse_info += f"### {fn}\n"
            if isinstance(found_locs, str):
                coarse_info += found_locs + "\n"
            else:
                coarse_info += "\n".join(found_locs) + "\n"
        self.logger.info("\n" + coarse_info)
        if len(model_found_locs_separated_in_samples) == 1:
            model_found_locs_separated_in_samples = (
                model_found_locs_separated_in_samples[0]
            )

        return (
            model_found_locs_separated_in_samples,
            {"raw_output_loc": raw_outputs},
            traj,
        )

    def localize_line_from_raw_text(
        self,
        file_names,
        mock=False,
        temperature=0.0,
        num_samples=1,
        keep_old_order=False,
    ):
        from agentless.util.api_requests import num_tokens_from_messages
        from agentless.util.model import make_model

        file_contents = get_repo_files(self.structure, file_names)
        raw_file_contents = {
            fn: line_wrap_content(code) for fn, code in file_contents.items()
        }
        contents = [
            self.file_content_template.format(file_name=fn, file_content=code)
            for fn, code in raw_file_contents.items()
        ]
        file_contents = "".join(contents)
        template = self.obtain_relevant_code_combine_top_n_prompt
        message = template.format(
            problem_statement=self.problem_statement, file_contents=file_contents
        )
        self.logger.info(f"prompting with message:")
        self.logger.info("\n" + message)

        def message_too_long(message):
            return (
                num_tokens_from_messages(message, self.model_name) >= MAX_CONTEXT_LENGTH
            )

        while message_too_long(message) and len(contents) > 1:
            self.logger.info(f"reducing to \n{len(contents)} files")
            contents = contents[:-1]
            file_contents = "".join(contents)
            message = template.format(
                problem_statement=self.problem_statement, file_contents=file_contents
            )  # Recreate message

        if message_too_long(message):
            raise ValueError(
                "The remaining file content is too long to fit within the context length"
            )
        self.logger.info(f"prompting with message:\n{message}")
        self.logger.info("=" * 80)

        if mock:
            self.logger.info("Skipping querying model since mock=True")
            traj = {
                "prompt": message,
                "usage": {
                    "prompt_tokens": num_tokens_from_messages(
                        message,
                        self.model_name,
                    ),
                },
            }
            return {}, {"raw_output_loc": ""}, traj

        model = make_model(
            model=self.model_name,
            backend=self.backend,
            logger=self.logger,
            max_tokens=self.max_tokens,
            temperature=temperature,
            batch_size=num_samples,
        )
        raw_trajs = model.codegen(message, num_samples=num_samples)

        # Merge trajectories
        raw_outputs = [raw_traj["response"] for raw_traj in raw_trajs]
        traj = {
            "prompt": message,
            "response": raw_outputs,
            "usage": {  # merge token usage
                "completion_tokens": sum(
                    raw_traj["usage"]["completion_tokens"] for raw_traj in raw_trajs
                ),
                "prompt_tokens": sum(
                    raw_traj["usage"]["prompt_tokens"] for raw_traj in raw_trajs
                ),
            },
        }
        model_found_locs_separated_in_samples = []
        for raw_output in raw_outputs:
            model_found_locs = extract_code_blocks(raw_output)
            model_found_locs_separated = extract_locs_for_files(
                model_found_locs, file_names, keep_old_order
            )
            model_found_locs_separated_in_samples.append(model_found_locs_separated)

            self.logger.info(f"==== raw output ====")
            self.logger.info(raw_output)
            self.logger.info("=" * 80)
            self.logger.info(f"==== extracted locs ====")
            for loc in model_found_locs_separated:
                self.logger.info(loc)
            self.logger.info("=" * 80)

        if len(model_found_locs_separated_in_samples) == 1:
            model_found_locs_separated_in_samples = (
                model_found_locs_separated_in_samples[0]
            )

        return (
            model_found_locs_separated_in_samples,
            {"raw_output_loc": raw_outputs},
            traj,
        )
