"""Focused encoder checks; optional real Gemma test needs tokenizer assets only.

P005_GEMMA_TOKENIZER_DIR=/path/to/tokenizer python3 -m unittest training.tests.test_encoding
Real tokenizer assets are not fetched by this test suite.
"""
import copy
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest

MODULE = Path(__file__).resolve().parents[1] / "encoding" / "gemma_tool_chat.py"
spec = importlib.util.spec_from_file_location("p005_encoding", MODULE)
enc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(enc)

TOOLS = [{"type": "function", "function": {"name": "read_file",
          "description": "Read source", "parameters": {"type": "object",
          "properties": {"filepath": {"type": "string"}}, "required": ["filepath"]}}}]


def transaction():
    return {"transaction_id": "real-record-1", "protocol_id": "p005-harness-tools-v1",
            "target_assistant_steps": [0, 1],
            "messages": [
                {"role": "system", "content": "SYSTEM PRIVATE CONTEXT"},
                {"role": "user", "content": "USER ISSUE 中国"},
                {"role": "assistant", "content": "", "reasoning_content": "Actual recorded thought",
                 "tool_calls": [{"id": "call-1", "type": "function",
                                 "function": {"name": "read_file",
                                              "arguments": '{"filepath":"a.py"}'}}]},
                {"role": "tool", "tool_call_id": "call-1", "content": 'TOOL OBSERVATION {"x":1}'},
                {"role": "assistant", "content": "Final answer 完成"}]}


def supervised_bytes(record):
    return bytes(t - 1 for t in record["labels"] if t != -100).decode("utf-8")


class StrictWholeTokenizer(enc.SyntheticTokenizer):
    """Cannot accidentally pass an implementation that renders one message."""
    def apply_chat_template(self, messages, **kwargs):
        if len(messages) != 5 or messages[1]["role"] != "user":
            raise AssertionError("isolated or incomplete prefix render")
        return super().apply_chat_template(messages, **kwargs)


class EncoderTests(unittest.TestCase):
    def test_whole_context_and_selected_assistant_masks(self):
        tx = transaction()
        tx["target_assistant_steps"] = [1]
        record = enc.encode_transaction(tx, StrictWholeTokenizer(), TOOLS)
        self.assertEqual(supervised_bytes(record), "Final answer 完成")
        self.assertEqual(record["meta"]["supervised_tokens"],
                         sum(x != -100 for x in record["labels"][1:]))
        self.assertTrue(record["meta"]["synthetic"])
        self.assertFalse(record["meta"]["training_eligible"])

    def test_actions_preserved_without_observation_thought_or_headers(self):
        tx = transaction()
        tx["target_assistant_steps"] = [0]
        record = enc.encode_transaction(tx, StrictWholeTokenizer(), TOOLS)
        target = supervised_bytes(record)
        self.assertEqual(target, '<|tool_call>call:read_file{"filepath":"a.py"}<tool_call|>')
        self.assertNotIn("OBSERVATION", target)
        self.assertNotIn("thought", target)
        self.assertNotIn("ASSISTANT", target)

    def test_thought_supervision_explicit_and_no_channel_prefix(self):
        tx = transaction()
        tx["target_assistant_steps"] = [0]
        record = enc.encode_transaction(tx, StrictWholeTokenizer(), TOOLS, thought_loss="include")
        target = supervised_bytes(record)
        self.assertIn("Actual recorded thought", target)
        self.assertNotIn("<|channel>", target)
        self.assertNotIn("TOOL OBSERVATION", target)

    def test_flat_and_standard_formats_have_identical_tokens(self):
        tx = transaction()
        flat = copy.deepcopy(tx)
        flat["messages"][2] = {"role": "assistant", "content": "",
            "reasoning_content": "Actual recorded thought", "tool_name": "read_file",
            "tool_call_id": "call-1", "arguments": {"filepath": "a.py"}}
        flat["messages"][3] = {"role": "tool", "content": "", "tool_name": "read_file",
            "tool_call_id": "call-1", "tool_response": 'TOOL OBSERVATION {"x":1}'}
        a = enc.encode_transaction(tx, enc.SyntheticTokenizer(), TOOLS)
        b = enc.encode_transaction(flat, enc.SyntheticTokenizer(), TOOLS)
        self.assertEqual(a["input_ids"], b["input_ids"])
        self.assertEqual(a["labels"], b["labels"])

    def test_parallel_tools_match_ids_and_do_not_leak_results(self):
        tx = transaction()
        tx["messages"][2]["tool_calls"].append({"id": "call-2", "type": "function",
             "function": {"name": "read_file", "arguments": {"filepath": "b.py"}}})
        tx["messages"].insert(4, {"role": "tool", "tool_call_id": "call-2", "content": "SECOND RESULT"})
        tx["target_assistant_steps"] = [0]
        record = enc.encode_transaction(tx, enc.SyntheticTokenizer(), TOOLS)
        target = supervised_bytes(record)
        self.assertIn("a.py", target)
        self.assertIn("b.py", target)
        self.assertNotIn("RESULT", target)

    def test_orphan_response_incomplete_call_and_invalid_json_fail(self):
        cases = []
        a = transaction(); a["messages"][3]["tool_call_id"] = "wrong"; cases.append(a)
        a = transaction(); del a["messages"][3]; cases.append(a)
        a = transaction(); a["messages"][2]["tool_calls"][0]["function"]["arguments"] = '{"bad"'; cases.append(a)
        for tx in cases:
            with self.subTest(tx=tx), self.assertRaises(enc.EncodingError):
                enc.encode_transaction(tx, enc.SyntheticTokenizer(), TOOLS)

    def test_reasoning_only_is_not_silently_reinterpreted_as_final_answer(self):
        tx = transaction()
        del tx["messages"][2]["tool_calls"]
        with self.assertRaisesRegex(enc.EncodingError, "reasoning-only"):
            enc.encode_transaction(tx, enc.SyntheticTokenizer(), TOOLS)

    def test_template_that_loses_thinking_fails_even_when_thought_loss_masked(self):
        class DroppingThoughts(enc.SyntheticTokenizer):
            def apply_chat_template(self, messages, **kwargs):
                ms = copy.deepcopy(messages)
                for m in ms:
                    m.pop("reasoning_content", None)
                return super().apply_chat_template(ms, **kwargs)
        with self.assertRaisesRegex(enc.EncodingError, "dropped"):
            enc.encode_transaction(transaction(), DroppingThoughts(), TOOLS)

    def test_rendered_token_disagreement_fails(self):
        class DifferentIDs(enc.SyntheticTokenizer):
            def apply_chat_template(self, messages, **kwargs):
                result = super().apply_chat_template(messages, **kwargs)
                return [999] + result if kwargs.get("tokenize") else result
        with self.assertRaisesRegex(enc.EncodingError, "tokens differ"):
            enc.encode_transaction(transaction(), DifferentIDs(), TOOLS)

    def test_zero_supervision_and_oversize_are_errors(self):
        tx = transaction(); tx["target_assistant_steps"] = []
        with self.assertRaises(enc.EncodingError):
            enc.encode_transaction(tx, enc.SyntheticTokenizer(), TOOLS)
        tx = transaction(); tx["messages"][-1]["content"] = ""; tx["target_assistant_steps"] = [1]
        with self.assertRaisesRegex(enc.EncodingError, "no supervised"):
            enc.encode_transaction(tx, enc.SyntheticTokenizer(), TOOLS)
        with self.assertRaises(enc.OversizeTransaction):
            enc.encode_transaction(transaction(), enc.SyntheticTokenizer(), TOOLS, max_seq_len=30)

    def test_context_changing_template_rejected(self):
        class BadContext(enc.SyntheticTokenizer):
            def apply_chat_template(self, messages, **kwargs):
                result = super().apply_chat_template(messages, **kwargs)
                if not kwargs.get("tokenize") and "P005_SENTINEL_" in result:
                    return "DIFFERENT CONTEXT" + result
                return result
        with self.assertRaisesRegex(enc.EncodingError, "changed context"):
            enc.encode_transaction(transaction(), BadContext(), TOOLS)

    def test_protocol_hash_and_synthetic_source_provenance(self):
        tx = transaction(); tx["source"] = {"uri": "synthetic://fixture"}
        record = enc.encode_transaction(tx, enc.SyntheticTokenizer(), TOOLS)
        for field in ("protocol_hash", "tokenizer_hash", "tools_hash", "chat_template_hash"):
            self.assertEqual(len(record["meta"][field]), 64)
        with self.assertRaisesRegex(enc.EncodingError, "protocol_id differ"):
            enc.encode_transaction(tx, enc.SyntheticTokenizer(), TOOLS, tool_protocol_id="other")

    def test_trace_provenance_survives_without_labels_or_code_leakage(self):
        tx = transaction()
        before = enc.encode_transaction(tx, enc.SyntheticTokenizer(), TOOLS)
        tx.update(sample_id="original-trace-7", origin_kind="student",
            group={"issue": "issue-7", "repo": "org/repo", "commit": "abc123", "split": "select",
                   "offline_labels": {"answer": "SECRET GOLD"}},
            source={"uri": "local://traces/7", "sha256": "a" * 64, "license": "Apache-2.0",
                    "trace_sha256": "b" * 64, "code": "SECRET WHOLE FILE",
                    "offline_labels": {"answer": "SECRET GOLD"}},
            offline_labels={"answer": "SECRET GOLD"})
        after = enc.encode_transaction(tx, enc.SyntheticTokenizer(), TOOLS)
        self.assertEqual(after["meta"]["sample_id"], "original-trace-7")
        self.assertEqual(after["meta"]["origin_kind"], "student")
        self.assertEqual(after["meta"]["group"]["split"], "select")
        self.assertEqual(after["meta"]["source"]["trace_sha256"], "b" * 64)
        self.assertEqual(after["meta"]["source"]["license"], "Apache-2.0")
        self.assertNotIn("SECRET", json.dumps(after))
        self.assertNotIn("offline_labels", json.dumps(after))
        for field in ("input_ids", "attention_mask", "labels"):
            self.assertEqual(before[field], after[field])
        for field in ("tokenizer_hash", "chat_template_hash", "tools_hash", "protocol_hash"):
            self.assertEqual(before["meta"][field], after["meta"][field])

    def test_cli_explicit_synthetic_atomic_failure_and_skip_report(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "tools.json").write_text(json.dumps(TOOLS))
            tx = transaction()
            (root / "input.jsonl").write_text(json.dumps(tx) + "\n")
            args = ["encode", "--input", str(root / "input.jsonl"), "--output", str(root / "out.jsonl"),
                    "--tools", str(root / "tools.json")]
            with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                self.assertNotEqual(enc.main(args), 0)
                self.assertEqual(enc.main(args + ["--synthetic"]), 0)
                before = (root / "out.jsonl").read_text()
                self.assertNotEqual(enc.main(args + ["--synthetic", "--max-seq-len", "20"]), 0)
                self.assertEqual((root / "out.jsonl").read_text(), before)
                large = transaction(); large["messages"][1]["content"] *= 1000
                (root / "input.jsonl").write_text(json.dumps(large) + "\n" + json.dumps(tx) + "\n")
                self.assertEqual(enc.main(args + ["--synthetic", "--on-oversize", "skip"]), 0)
            report = json.loads((root / "out.jsonl.report.json").read_text())
            self.assertEqual(report["written"], 1)
            self.assertEqual(len(report["skipped_oversize"]), 1)


@unittest.skipUnless(os.environ.get("P005_GEMMA_TOKENIZER_DIR"), "optional real tokenizer assets absent")
class RealGemmaTokenizerTests(unittest.TestCase):
    def test_official_tokenizer_template_no_weights(self):
        from transformers import AutoTokenizer
        # Old transformers accepts only a dict here; the explicit opt-in below
        # affects a video-token Python alias, not tokenizer.json token IDs.
        kwargs = {}
        if os.environ.get("P005_OLD_TRANSFORMERS_ALIAS_COMPAT") == "1":
            kwargs["extra_special_tokens"] = {}
        tok = AutoTokenizer.from_pretrained(os.environ["P005_GEMMA_TOKENIZER_DIR"],
                                           local_files_only=True, **kwargs)
        before = enc.tokenizer_fingerprint(tok)
        tx = transaction()
        tx["target_assistant_steps"] = [0]
        for thinking in (True, False):
            record = enc.encode_transaction(tx, tok, TOOLS, enable_thinking=thinking)
            target = tok.decode([x for x in record["labels"] if x != -100])
            self.assertIn("read_file", target)
            self.assertIn("a.py", target)
            self.assertNotIn("TOOL OBSERVATION", target)
            self.assertNotIn("Actual recorded thought", target)
            self.assertNotIn("<|turn>", target)
            self.assertEqual(before, enc.tokenizer_fingerprint(tok))
        record = enc.encode_transaction(tx, tok, TOOLS, thought_loss="include")
        target = tok.decode([x for x in record["labels"] if x != -100])
        self.assertIn("Actual recorded thought", target)
        self.assertNotIn("<|channel>", target)
        self.assertEqual(record["meta"]["synthetic"], False)
        tx["target_assistant_steps"] = [1]
        record = enc.encode_transaction(tx, tok, TOOLS, content_format="text-parts")
        target = tok.decode([x for x in record["labels"] if x != -100])
        self.assertEqual(target, "Final answer 完成")
        self.assertNotIn("<|turn>", target)


if __name__ == "__main__":
    unittest.main()
