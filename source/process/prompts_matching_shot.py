from typing import Dict, List


class PRTGPromptsMatchingShot:

    @staticmethod
    def few_shot_simple(example_descriptions: list[str], example_commit_messages: list[str],
                        example_linked_issue_titles: list[str],
                        example_titles: list[str], description: str, commit_messages: str, linked_issue_titles: str,
                        **kwargs) -> str:
        # prompt = "write a Pull Request title for some given information.Pay attention to some similar PR\n"
        prompt = "Please write a Pull Request title for given information with the help of some examples."
        for i in range(len(example_descriptions)):
            prompt += (f"Example [{i}]:\n"
                       f"Description:{example_descriptions[i]}\n"
                       f"Commit Messages:{example_commit_messages[i]}\n"
                       f"Linked Issue Titles:{example_linked_issue_titles[i]}\n"
                       f"Pull Request Title:{example_titles[i]}\n")

        prompt += (f"You need to write:\n"
                   f"Description:{description}\n"
                   f"Commit Messages:{commit_messages}\n"
                   f"Linked Issues Titles:{linked_issue_titles}\n"
                   f"Pull Request Title:")
        return prompt


class PRTGChatPromptsMatchingShot:

    @staticmethod
    def few_shot_simple(example_descriptions: list[str], example_commit_messages: list[str],
                        example_linked_issue_titles: list[str],
                        example_titles: list[str], description: str, commit_messages: str, linked_issue_titles: str,
                        **kwargs) -> List[Dict[str, str]]:
        return [
            {
                "role": "system",
                "content": "You are a helpful programming assistant that creates Pull Request Title for given information.",
            },
            {"role": "user",
             "content": PRTGPromptsMatchingShot.few_shot_simple(example_descriptions=example_descriptions,
                                                                example_commit_messages=example_commit_messages,
                                                                example_linked_issue_titles=example_linked_issue_titles,
                                                                example_titles=example_titles, description=description,
                                                                commit_messages=commit_messages,
                                                                linked_issue_titles=linked_issue_titles)},
        ]
