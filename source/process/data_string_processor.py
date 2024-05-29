
import logging
import os
import re


import jsonlines
import pandas as pd


class StringPreprocessor:

    def __init__(self,
                 prompt_configuration: str
                 ):
        self._prompt_configuration = prompt_configuration

    def process_file(
            self, input_path: str, output_path: str, use_cache: bool
    ) -> None:
        if use_cache and os.path.exists(output_path):
            logging.info("Found StringPreprocessed data!")
        else:
            logging.info("Start StringPreprocessed data!")
            df = pd.read_csv(input_path)
            print(df['text'].size)
            texts = df['text']
            titles = df['summary']
            chunk = []
            with jsonlines.open(output_path, "w") as file:
                for i in range(len(texts)):
                    description = ""
                    commit_messages = ""
                    linked_issue_titles = ""
                    desc_content = re.findall(r'<desc>(.*?)</desc>', texts[i], re.DOTALL)
                    cmt_content = re.findall(r'<cmt>(.*?)</cmt>', texts[i], re.DOTALL)
                    iss_content = re.findall(r'<iss>(.*?)</iss>', texts[i], re.DOTALL)
                    for desc in desc_content:
                        description += desc
                    for cmt in cmt_content:
                        commit_messages += cmt
                    for iss in iss_content:
                        linked_issue_titles += iss

                    pre_data = {
                        "description": description,
                        "commit_messages": commit_messages,
                        "linked_issue_titles": linked_issue_titles,
                        "title": titles[i]
                    }
                    # pre_data = PreData(description, commit_messages, linked_issue_title, titles[i])
                    chunk.append(pre_data)
                file.write_all(chunk)
            logging.info("Finish StringProcessing data!")
