from typing import Dict, List


class PRTGPrompts:

    @staticmethod
    def zero_shot_simple(description: str, commit_messages: str, linked_issue_titles: str, **kwargs) -> str:
        return (
            f"Please write a Pull Request Title in English for the given information.\n"
            f"Description:{description}\n"
            f"Commit Messages:{commit_messages}\n"
            f"Linked Issue Titles:{linked_issue_titles}\n"
            f"Pull Request Title:")


class PRTGChatPrompts:

    @staticmethod
    def zero_shot_simple(description: str, commit_messages: str, linked_issue_titles: str, **kwargs) -> List[
        Dict[str, str]]:
        return [
            {
                "role": "system",
                "content": "You are a helpful programming assistant that creates Pull Request Title for given information.",
            },
            {"role": "user",
             "content": PRTGPrompts.zero_shot_simple(description=description, commit_messages=commit_messages,
                                                     linked_issue_titles=linked_issue_titles)},
        ]
