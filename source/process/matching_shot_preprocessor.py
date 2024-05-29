import logging
import os
from typing import Any, Callable, Dict, List, Optional

import jsonlines
from tqdm import tqdm
import pandas as pd
from .prompts_matching_shot import PRTGPromptsMatchingShot, PRTGChatPromptsMatchingShot


class DataPreprocessorMatchingShot:
    prompt_constructors: Dict[str, Callable] = {
        "simple": PRTGPromptsMatchingShot.few_shot_simple,
    }
    prompt_constructors_chat: Dict[str, Callable] = {
        "simple": PRTGChatPromptsMatchingShot.few_shot_simple,
    }

    def __init__(self,
                 prompt_configuration: str
                 ):
        self._prompt_configuration = prompt_configuration

    def process(
            self,
            prompt_configuration: str,
            example_descriptions: Optional[list[str]] = None,
            example_commit_messages: Optional[list[str]] = None,
            example_linked_issue_titles: Optional[list[str]] = None,
            example_titles: Optional[list[str]] = None,
            description: Optional[str] = None,
            commit_messages: Optional[str] = None,
            linked_issue_titles: Optional[str] = None,
    ) -> str:
        prompt = DataPreprocessorMatchingShot.prompt_constructors[prompt_configuration](
            example_descriptions=example_descriptions,
            example_commit_messages=example_commit_messages,
            example_linked_issue_titles=example_linked_issue_titles,
            example_titles=example_titles,
            description=description,
            commit_messages=commit_messages,
            linked_issue_titles=linked_issue_titles,
        )
        return prompt

    def process_file(
            self, input_path: str, output_path: str, matching_dir: str, input_path_file: str, limit_test: int,
            shot: int, use_cache: bool
    ) -> None:
        if use_cache and os.path.exists(output_path):
            logging.info("Found preprocessed data!")
        else:
            logging.info("Start processing data!")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            open(output_path, "w").close()

            first_match = []
            with jsonlines.open(f"{matching_dir}/first_{input_path_file}", "r") as reader:
                for i, line in enumerate(reader):
                    first_match.append(line)
            second_match = []
            with jsonlines.open(f"{matching_dir}/second_{input_path_file}", "r") as reader:
                for i, line in enumerate(reader):
                    second_match.append(line)
            third_match = []
            with jsonlines.open(f"{matching_dir}/third_{input_path_file}", "r") as reader:
                for i, line in enumerate(reader):
                    third_match.append(line)
            fourth_match = []
            with jsonlines.open(f"{matching_dir}/fourth_{input_path_file}", "r") as reader:
                for i, line in enumerate(reader):
                    fourth_match.append(line)
            fifth_match = []
            with jsonlines.open(f"{matching_dir}/fifth_{input_path_file}", "r") as reader:
                for i, line in enumerate(reader):
                    fifth_match.append(line)
            chunk = []
            with jsonlines.open(input_path, "r") as reader:
                for i, line in enumerate(reader):
                    chunk.append(line)
            df = pd.DataFrame(chunk)

            chunk: List[Dict[str, Any]] = []
            for i, row in tqdm(df.iterrows(), "Processing prompts from input file"):
                if i >= limit_test:
                    break
                descriptions = [first_match[i]['description']]
                commit_messages = [first_match[i]['commit_messages']]
                linked_issue_titles = [first_match[i]['linked_issue_titles']]
                titles = [first_match[i]['title']]
                if shot > 1:
                    descriptions.append(second_match[i]['description'])
                    commit_messages.append(second_match[i]['commit_messages'])
                    linked_issue_titles.append(second_match[i]['linked_issue_titles'])
                    titles.append(second_match[i]['title'])
                if shot > 2:
                    descriptions.append(third_match[i]['description'])
                    commit_messages.append(third_match[i]['commit_messages'])
                    linked_issue_titles.append(third_match[i]['linked_issue_titles'])
                    titles.append(third_match[i]['title'])
                if shot > 3:
                    descriptions.append(fourth_match[i]['description'])
                    commit_messages.append(fourth_match[i]['commit_messages'])
                    linked_issue_titles.append(fourth_match[i]['linked_issue_titles'])
                    titles.append(fourth_match[i]['title'])
                if shot > 4:
                    descriptions.append(fifth_match[i]['description'])
                    commit_messages.append(fifth_match[i]['commit_messages'])
                    linked_issue_titles.append(fifth_match[i]['linked_issue_titles'])
                    titles.append(fifth_match[i]['title'])
                prompt = {
                    "prompt": self.process(
                        prompt_configuration=self._prompt_configuration,
                        example_descriptions=descriptions,
                        example_commit_messages=commit_messages,
                        example_linked_issue_titles=linked_issue_titles,
                        example_titles=titles,
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
