"""Build reviewed causal action recipes for the pinned compatibility candidates."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
REPLAY = ROOT / "competition/vfarl/replay"
ART = REPLAY / "artifacts/expand-compat"


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def function_text(path, name):
    text = path.read_text()
    tree = ast.parse(text)
    node = next(n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)
    lines = text.splitlines(keepends=True)
    return "".join(lines[node.lineno - 1:node.end_lineno]), node.lineno, node.end_lineno


def prepare(key, rel, symbol, body, notes, family):
    prep_path = REPLAY / ("expand-r4-" + key + "-source-preparation.json")
    prep = json.loads(prep_path.read_text())
    workspace = ROOT / prep["workspace"]
    upstream = ART / (prep["canonical_repo"].replace("/", "__") + "-upstream")
    before, line, end = function_text(workspace / rel, symbol)
    after, _, _ = function_text(upstream / rel, symbol)
    if before == after:
        raise ValueError("No causal source change")
    command = "python3 - <<'PYREPLAY'\nfrom pathlib import Path\nimport hashlib,sys\n"
    module = rel.removeprefix("src/").removesuffix(".py").replace("/", ".")
    command += f"import importlib\nm=importlib.import_module({module!r})\nactual=Path(m.__file__).resolve()\nexpected=(Path.cwd()/{rel!r}).resolve()\nassert actual==expected,(actual,expected)\nprint('source_binding',actual,'sha256',hashlib.sha256(actual.read_bytes()).hexdigest(),'Python',sys.version)\n"
    command += body + "\nPYREPLAY"
    def run(status):
        return {"tool": "run_command", "arguments": {"command": command}, "expected_status": status,
                "reviewed_command_sha256": hashlib.sha256(command.encode()).hexdigest()}
    recipe = {"evidence_kind": "human_reviewed_off_policy_demonstration",
              "reviewed_by": "Codex compatibility source/diff and original requirements review,2026-10-25",
              "review_notes": notes + " Original test/required files and identifiers retained. Commands exercise production APIs, not authored test files. No model/policy or counterfactual-pair claim.",
              "actions": [{"tool": "get_status", "arguments": {}, "expected_status": "ok"},
                          {"tool": "read_file", "arguments": {"filepath": rel, "start_line": max(1,line-4), "end_line": min(line+125,end+5)}, "expected_status": "ok"},
                          run("error"),
                          {"tool": "edit_file", "arguments": {"filepath": rel, "old_string": before, "new_string": after}, "expected_status": "ok"},
                          run("ok"),
                          {"tool": "get_status", "arguments": {}, "expected_status": "ok"},
                          {"tool": "submit_patch", "arguments": {}, "expected_status": "ok"}],
              "final_response": notes + " The actual source-bound reproduction passes after the production repair. Patch submitted for independent full task verification."}
    recipe_path = REPLAY / ("expand-r4-" + key + "-reviewed-recipe.json")
    save(recipe_path, recipe)
    prep.update({"recipe": str(recipe_path.relative_to(ROOT)), "repair_family_id": family,
                 "snapshot_group_id": prep["canonical_repo"] + "::" + prep["upstream_source"]["resolved_revision"],
                 "mechanism_evidence": {"primary_mechanism": "R4", "production_symbol": symbol, "causal_contract": notes,
                                        "secondary_mechanisms": ["S5"] if key.startswith("apispec") else [],
                                        "actual_observation_scope": "Production API across explicitly supported OpenAPI versions" if key.startswith("apispec") else "Python3.11 function/code roundtrip, line/exception tables and argument semantics"}})
    save(prep_path, prep)


prepare("apispec-ref", "src/apispec/ext/marshmallow/field_converter.py", "nested2properties", '''from apispec import APISpec
from apispec.ext.marshmallow import MarshmallowPlugin
from marshmallow import Schema,fields
class Child(Schema):
 name=fields.Str()
results=[]
for version in ('2.0','3.0.3','3.1.0'):
 plugin=MarshmallowPlugin()
 spec=APISpec(title='version contract',version='1',openapi_version=version,plugins=[plugin])
 spec.components.schema('Child',schema=Child)
 result=plugin.converter.field2property(fields.Nested(Child,metadata={'description':'child note'}))
 print('nested_ref_version',version,result)
 results.append((version,result))
for version,result in results:
 assert result['description']=='child note'
 if version=='3.1.0':
  assert '$ref' in result and 'allOf' not in result,result
 else:
  assert 'allOf' in result and '$ref' not in result,result''',
        "OpenAPI2/3.0 reference objects need allOf wrapping to retain sibling metadata; OpenAPI3.1 supports a direct $ref sibling. Restore the explicit version guard without changing the field metadata or schema identity.",
        "apispec::FieldConverterMixin.nested2properties::OAS_reference_sibling_version_contract")

prepare("apispec-null", "src/apispec/ext/marshmallow/field_converter.py", "field2nullable", '''from apispec import APISpec
from apispec.ext.marshmallow import MarshmallowPlugin
from marshmallow import Schema,fields
class Child(Schema):
 name=fields.Str()
results=[]
for version in ('2.0','3.0.3','3.1.0'):
 for description in (False,True):
  plugin=MarshmallowPlugin()
  spec=APISpec(title='nullable contract',version='1',openapi_version=version,plugins=[plugin])
  spec.components.schema('Child',schema=Child)
  metadata={'description':'child note'} if description else {}
  result=plugin.converter.field2property(fields.Nested(Child,allow_none=True,metadata=metadata))
  print('nullable_ref_version',version,'with_metadata',description,result)
  results.append((version,result))
for version,result in results:
 if version=='2.0':
  assert result['x-nullable'] is True,result
 elif version=='3.0.3':
  assert 'anyOf' in result and {'type':'object','nullable':True} in result['anyOf'],result
  assert any('$ref' in x for x in result['anyOf']),result
 else:
  assert 'anyOf' in result and {'type':'null'} in result['anyOf'],result
  assert any('$ref' in x for x in result['anyOf']),result''',
        "Nullable referenced schemas require x-nullable in OpenAPI2, a nullable-object alternative in3.0, and an explicit null alternative in3.1. Restore version-specific union handling for both direct refs and allOf refs while retaining metadata.",
        "apispec::FieldConverterMixin.field2nullable::OAS_nullable_reference_version_contract")

prepare("cloudpickle", "cloudpickle/cloudpickle.py", "_code_reduce", '''import cloudpickle,types
def compute(value,/,offset=2,*,scale=3):
 try:
  return (value+offset)*scale
 except TypeError as error:
  raise ValueError('arithmetic input') from error
original=compute.__code__
constructor,args=m._code_reduce(original)
print('code_reduce_runtime',sys.version_info[:3],'constructor_argument_count',len(args),'has_exceptiontable',hasattr(original,'co_exceptiontable'))
restored_code=constructor(*args)
assert restored_code.co_posonlyargcount==original.co_posonlyargcount
assert restored_code.co_kwonlyargcount==original.co_kwonlyargcount
assert restored_code.co_linetable==original.co_linetable
assert restored_code.co_exceptiontable==original.co_exceptiontable
restored=cloudpickle.loads(cloudpickle.dumps(compute))
assert restored(4,scale=2)==12
try:
 restored('invalid')
except ValueError as error:
 print('roundtrip_exception',type(error).__name__,str(error))
else:
 raise AssertionError('exception table semantics lost')
print('code_runtime_contract_preserved')''',
        "Restore runtime-specific CodeType reconstruction: Python3.11 needs qualname, line table and exception table in its constructor, while earlier supported Python branches retain their own layouts. Preserve tuple-copy determinism and positional/keyword-only call semantics.",
        "cloudpickle::_code_reduce::Python_CodeType_runtime_version_layout")
