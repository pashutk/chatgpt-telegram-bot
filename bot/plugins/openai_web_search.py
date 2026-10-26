from typing import Dict

import httpx

from .plugin import Plugin


class OpenAIWebSearchPlugin(Plugin):
    """
    A plugin to search the web with OpenAI's built-in web_search tool (Responses API), using OPENAI_MODEL
    """
    def get_source_name(self) -> str:
        return "OpenAI Web Search"

    def get_spec(self) -> [Dict]:
        return [{
            "name": "web_search",
            "description": "Search the web and get an answer with sources. Use it for recent events, news, prices, "
                           "exchange rates, schedules, facts about people and anything you are not sure about.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The question to research, in the user's language, with all needed context"
                    }
                },
                "required": ["query"],
            },
        }]

    def get_progress_message_key(self, function_name) -> str:
        return 'web_search_progress'

    async def execute(self, function_name, helper, **kwargs) -> Dict:
        body = {
            'model': helper.config['model'],
            'input': kwargs['query'],
            'tools': [{'type': 'web_search'}],
        }
        if helper.config['reasoning_effort']:
            body['reasoning'] = {'effort': helper.config['reasoning_effort']}

        # the pinned openai SDK predates the Responses API, so call the endpoint directly
        response = await helper.client.post('/responses', cast_to=httpx.Response, body=body)
        output = response.json()['output']
        answer = ''.join(content.get('text', '')
                         for item in output if item['type'] == 'message'
                         for content in item['content'])
        return {'answer': answer}
