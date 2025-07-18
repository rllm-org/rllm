import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
from transformers import AutoTokenizer
from rllm.parser.chat_template.parser import ChatTemplateParser

class TestInternLMModel(unittest.TestCase):
    def test_internlm_chat_template(self):
        model_name = "internlm/internlm2_5-1_8b-chat"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        parser = ChatTemplateParser.get_parser(tokenizer)

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi, how can I help you?"}
        ]

        parsed_output = parser.parse(messages, is_first_msg=True)
        expected_output = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nHi, how can I help you?<|eot_id|>"

        self.assertEqual(parsed_output, expected_output)

if __name__ == '__main__':
    unittest.main()
