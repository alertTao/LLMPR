import logging
import os
from typing import Any, Callable, Dict, List, Optional

import jsonlines
from tqdm import tqdm
import pandas as pd
from .prompts_template import PRTGChatPrompts, PRTGPrompts


class DataPreprocessor:
    prompt_constructors: Dict[str, Callable] = {
        "simple": PRTGPrompts.zero_shot_simple,
    }
    prompt_constructors_chat: Dict[str, Callable] = {
        "simple": PRTGChatPrompts.zero_shot_simple,
    }

    def __init__(self,
                 prompt_configuration: str
                 ):
        self._prompt_configuration = prompt_configuration

    def process(
            self,
            prompt_configuration: str,
            description: Optional[str] = None,
            commit_messages: Optional[str] = None,
            linked_issue_titles: Optional[str] = None,
    ) -> str:
        prompt = DataPreprocessor.prompt_constructors[prompt_configuration](
            description=description,
            commit_messages=commit_messages,
            linked_issue_titles=linked_issue_titles,
        )
        return prompt

    def process_file(
            self, input_path: str, output_path: str, use_cache: bool, limit_test: int
    ) -> None:
        if use_cache and os.path.exists(output_path):
            logging.info("Found preprocessed data!")
        else:
            logging.info("Start processing data!")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            open(output_path, "w").close()

            chunk = []
            with jsonlines.open(input_path, "r") as reader:
                for i, line in enumerate(reader):
                    chunk.append(line)
            df = pd.DataFrame(chunk)

            chunk: List[Dict[str, Any]] = []
            for i, row in tqdm(df.iterrows(), "Processing prompts from input file"):
                if i >= limit_test:
                    break
                prompt = {
                    "prompt": self.process(
                        prompt_configuration=self._prompt_configuration,
                        description=row['description'],
                        commit_messages=row['commit_messages'],
                        linked_issue_titles=row['linked_issue_titles'],
                    ),
                    "title": row['title'],
                }
                chunk.append(prompt)

            with jsonlines.open(output_path, "a") as writer:
                writer.write_all(chunk)
            logging.info("Finish processing prompts!")
