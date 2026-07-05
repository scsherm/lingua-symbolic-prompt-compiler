import unittest

from prompt_compiler.prompt.chunk import ChunkType
from prompt_compiler.prompt.chunkers import ParagraphChunker, SchemaAwareChunker, SentenceChunker
from prompt_compiler.prompt.template import PromptTemplate


class TemplateAndChunkingTests(unittest.TestCase):
    def test_template_renders_input_without_changing_template(self):
        template = PromptTemplate("Rules:\nReturn JSON only.\nInput:\n{{input}}")

        rendered = template.render({"input": "hello"})

        self.assertEqual(rendered, "Rules:\nReturn JSON only.\nInput:\nhello")
        self.assertEqual(template.text, "Rules:\nReturn JSON only.\nInput:\n{{input}}")

    def test_instruction_text_removes_variable_payload_for_token_measurement(self):
        template = PromptTemplate("Task: triage\nInput:\n{{input}}\nReturn JSON.")

        self.assertNotIn("{{input}}", template.instruction_text())
        self.assertIn("Task: triage", template.instruction_text())
        self.assertIn("Return JSON.", template.instruction_text())

    def test_chunkers_mark_input_placeholders_as_protected(self):
        prompt = "Task: triage alert.\n\nInput:\n{{input}}\n\nReturn JSON only."

        paragraph_chunks = ParagraphChunker().chunk(prompt)
        sentence_chunks = SentenceChunker().chunk(prompt)

        self.assertTrue(any(chunk.protected for chunk in paragraph_chunks))
        self.assertTrue(any(chunk.protected for chunk in sentence_chunks))
        self.assertTrue(any(chunk.chunk_type == ChunkType.INPUT_SLOT for chunk in paragraph_chunks))

    def test_schema_chunker_does_not_split_template_placeholder_as_schema(self):
        prompt = "Return JSON like {\"status\":\"OPEN\"}.\nInput:\n{{input}}"

        chunks = SchemaAwareChunker().chunk(prompt)

        self.assertTrue(any("{{input}}" in chunk.text and chunk.protected for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
