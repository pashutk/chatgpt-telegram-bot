import os
import uuid
from typing import Dict

from .plugin import Plugin


class ImageGenerationPlugin(Plugin):
    """
    A plugin to draw images with the configured OpenAI image model (IMAGE_MODEL)
    """
    def get_source_name(self) -> str:
        return "OpenAI Images"

    def get_spec(self) -> [Dict]:
        return [{
            "name": "generate_image",
            "description": "Draw a new image from a text description: a picture, illustration, greeting card, "
                           "poster, drawing, sketch, diagram or chart. Use it whenever the user asks to draw, "
                           "paint, sketch, depict or make any kind of image.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Detailed description of the image, in the user's language. Include any "
                                       "text that must appear on the image verbatim, and the orientation, style "
                                       "or level of detail if the user asked for them."
                    }
                },
                "required": ["prompt"],
            },
        }]

    def get_progress_message_key(self, function_name) -> str:
        return 'image_generation_progress'

    async def execute(self, function_name, helper, **kwargs) -> Dict:
        image, _ = await helper.generate_image(kwargs['prompt'])
        if isinstance(image, str):
            return {'direct_result': {'kind': 'photo', 'format': 'url', 'value': image}}

        os.makedirs('uploads/image_generation', exist_ok=True)
        image_file_path = os.path.join('uploads/image_generation', f'{uuid.uuid4().hex}.png')
        with open(image_file_path, 'wb') as f:
            f.write(image)
        return {'direct_result': {'kind': 'photo', 'format': 'path', 'value': image_file_path}}
