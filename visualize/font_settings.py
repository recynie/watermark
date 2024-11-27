# Copyright 2024 THU-BPM MarkLLM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ============================================
# font_settings.py
# Description: Font settings for visualization
# ============================================

from PIL import ImageFont

class FontSettings:
    """Font settings for visualization."""

    def __init__(self, font_path: str = "font/Courier_New_Bold.ttf", font_size: int = 20) -> None:
        self.font_path = font_path
        self.font_size = font_size
        self.font = ImageFont.truetype(self.font_path, self.font_size)
        self.font.getsize=self.get_text_dimensions

    def get_text_dimensions(self, text: str) -> tuple:
        """
        Get the dimensions of the text.
        This method works with both older and newer versions of Pillow.
        """
        if hasattr(self.font, "getbbox"):
            bbox = self.font.getbbox(text)
            return bbox[2] - bbox[0], bbox[3] - bbox[1]
        else:
            return self.font.getsize(text)
    