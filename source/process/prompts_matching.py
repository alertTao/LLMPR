from typing import Dict, List


class PRTGPromptsMatching:

    @staticmethod
    def one_shot_simple(example_description: str, example_commit_messages: str, example_linked_issue_titles: str,
                        example_title: str, description: str, commit_messages: str, linked_issue_titles: str,
                        **kwargs) -> str:
        prompt = "Please write a Pull Request title for given information with the help of an example"
        prompt += (f"Example:\n"
                   f"Description:{example_description}\n"
                   f"Commit Messages:{example_commit_messages}\n"
                   f"Linked Issue Titles:{example_linked_issue_titles}\n"
                   f"Pull Request Title:{example_title}\n")

        prompt += (f"You need to write:\n"
                   f"Description:{description}\n"
                   f"Commit Messages:{commit_messages}\n"
                   f"Linked Issue Titles:{linked_issue_titles}\n"
                   f"Pull Request Title:")
        return prompt


class PRTGChatPromptsMatching:

    @staticmethod
    def one_shot_simple(example_description: str, example_commit_messages: str, example_linked_issue_titles: str,
                        example_title: str, description: str, commit_messages: str, linked_issue_titles: str,
                        **kwargs) -> List[
        Dict[str, str]]:
        return [
            {
                "role": "system",
                "content": "You are a helpful programming assistant that creates Pull Request Title for given information.",
            },
            {"role": "user",
             "content": PRTGPromptsMatching.one_shot_simple(example_description=example_description,
                                                            example_commit_messages=example_commit_messages,
                                                            example_linked_issue_titles=example_linked_issue_titles,
                                                            example_title=example_title, description=description,
                                                            commit_messages=commit_messages,
                                                            linked_issue_titles=linked_issue_titles)},
        ]
