import json
import os
import re
from typing import Any, Tuple

from PIL import Image

from agentflow.engine.factory import create_llm_engine
from agentflow.models.formatters import MemoryVerification

_prompt_tpt = """
    Task: Evaluate if the current memory is complete and accurate enough to answer the query, or if more tools are needed.

    Context:
    {context}

    Instructions:
    1.  Review the query, initial analysis, and memory.
    2.  Assess the completeness of the memory: Does it fully address all parts of the query?
    3.  Check for potential issues:
        -   Are there any inconsistencies or contradictions?
        -   Is any information ambiguous or in need of verification?
    4.  Determine if any unused tools could provide missing information.

    Final Determination:
    -   If the memory is sufficient, explain why and conclude with "STOP".
    -   If more information is needed, explain what's missing, which tools could help, and conclude with "CONTINUE".

    IMPORTANT: The response must end with either "Conclusion: STOP" or "Conclusion: CONTINUE".
"""


class Verifier:
    def __init__(self,
                 llm_engine_name: str,
                 llm_engine_fixed_name: str = "dashscope",
                 toolbox_metadata: dict = None,
                 available_tools: list = None,
                 verbose: bool = False,
                 base_url: str = None,
                 is_multimodal: bool = False,
                 temperature: float = .0):
        self.llm_engine_name = llm_engine_name
        self.llm_engine_fixed_name = llm_engine_fixed_name
        self.is_multimodal = is_multimodal
        self._llm_engine = create_llm_engine(
            model_string=llm_engine_fixed_name,
            is_multimodal=False,
            base_url=base_url,
            temperature=temperature)
        self.toolbox_metadata = toolbox_metadata if toolbox_metadata is not None else {}
        self.available_tools = available_tools if available_tools is not None else []
        self.verbose = verbose

    def get_image_info(self, image_path: str) -> dict:
        image_info = {}
        if image_path and os.path.isfile(image_path):
            image_info["image_path"] = image_path
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                image_info.update({"width": width, "height": height})
            except Exception as e:
                print(f"Error processing image file: {str(e)}")
        return image_info

    def verify(self, context):
        prompt = _prompt_tpt.format(context=context)

        stop_verification = self._llm_engine(
            prompt, response_format=MemoryVerification)
        return self._extract_conclusion(stop_verification)

    def _extract_conclusion(self, response: Any) -> Tuple[str, str]:
        if isinstance(response, str):
            # Attempt to parse the response as JSON
            try:
                response_dict = json.loads(response)
                response = MemoryVerification(**response_dict)
            except Exception as e:
                print(f"Failed to parse response as JSON: {str(e)}")
        if isinstance(response, MemoryVerification):
            analysis = response.analysis
            stop_signal = response.stop_signal
            if stop_signal:
                return analysis, 'STOP'
            else:
                return analysis, 'CONTINUE'
        else:
            analysis = response
            pattern = r'conclusion\**:?\s*\**\s*(\w+)'
            matches = list(
                re.finditer(pattern, response, re.IGNORECASE | re.DOTALL))
            if matches:
                conclusion = matches[-1].group(1).upper()
                if conclusion in ['STOP', 'CONTINUE']:
                    return analysis, conclusion

            # If no valid conclusion found, search for STOP or CONTINUE anywhere in the text
            if 'stop' in response.lower():
                return analysis, 'STOP'
            elif 'continue' in response.lower():
                return analysis, 'CONTINUE'
            else:
                print(
                    "No valid conclusion (STOP or CONTINUE) found in the response. Continuing..."
                )
                return analysis, 'CONTINUE'
