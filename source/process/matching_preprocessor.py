import logging
import os
from typing import Any, Callable, Dict, List, Optional

import jsonlines
from tqdm import tqdm
import pandas as pd
from .prompts_matching import PRTGPromptsMatching, PRTGChatPromptsMatching


class DataPreprocessorMatching:
    prompt_constructors: Dict[str, Callable] = {
        "simple": PRTGPromptsMatching.one_shot_simple,
    }
    prompt_constructors_chat: Dict[str, Callable] = {
        "simple": PRTGChatPromptsMatching.one_shot_simple,
    }

    def __init__(self,
                 prompt_configuration: str
                 ):
        self._prompt_configuration = prompt_configuration

    def process(
            self,
            prompt_configuration: str,
            example_description: Optional[str] = None,
            example_commit_messages: Optional[str] = None,
            example_linked_issue_titles: Optional[str] = None,
            example_title: Optional[str] = None,
            description: Optional[str] = None,
            commit_messages: Optional[str] = None,
            linked_issue_titles: Optional[str] = None,
    ) -> str:
        prompt = DataPreprocessorMatching.prompt_constructors[prompt_configuration](
            example_description=example_description,
            example_commit_messages=example_commit_messages,
            example_linked_issue_titles=example_linked_issue_titles,
            example_title=example_title,
            description=description,
            commit_messages=commit_messages,
            linked_issue_titles=linked_issue_titles,
        )
        return prompt

    def process_file(
            self, input_path: str, output_path: str, matching_path: str, limit_test: int, use_cache: bool
    ) -> None:
        if use_cache and os.path.exists(output_path):
            logging.info("Found preprocessed data!")
        else:
            logging.info("Start processing data!")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            open(output_path, "w").close()

            matchs = []
            with jsonlines.open(matching_path, "r") as reader:
                for i, line in enumerate(reader):
                    matchs.append(line)
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
                        example_description=matchs[i]['description'],
                        example_commit_messages=matchs[i]['commit_messages'],
                        example_linked_issue_titles=matchs[i]['linked_issue_titles'],
                        example_title=matchs[i]['title'],
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
