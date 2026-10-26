from __future__ import annotations

from abc import abstractmethod, ABC
from typing import Dict


class Plugin(ABC):
    """
    A plugin interface which can be used to create plugins for the ChatGPT API.
    """

    @abstractmethod
    def get_source_name(self) -> str:
        """
        Return the name of the source of the plugin.
        """
        pass

    @abstractmethod
    def get_spec(self) -> [Dict]:
        """
        Function specs in the form of JSON schema as specified in the OpenAI documentation:
        https://platform.openai.com/docs/api-reference/chat/create#chat/create-functions
        """
        pass

    def get_progress_message_key(self, function_name) -> str | None:
        """
        Key in translations.json of a message shown to the user while the function runs, or None
        """
        return None

    @abstractmethod
    async def execute(self, function_name, helper, **kwargs) -> Dict:
        """
        Execute the plugin and return a JSON serializable response
        """
        pass
