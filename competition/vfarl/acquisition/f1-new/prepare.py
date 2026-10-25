"""F1-new batch for F600: prepare + replay + normalize-count 40 fresh F1 tasks.

Stages: prepare | run | count
Follows S-batch conventions (pinned pages, ls-remote refs, acquire, adapted
task with verbatim required nodes, reviewed recipe, dual-phase official replay)
without touching S600 files. Outputs live in acquisition/f1-new/ and
replay/artifacts/f1-new/ (+ replay/f1-new-*-reviewed-recipe.json and
replay/f1-new-*-replay-result.json).
"""
import argparse, ast, difflib, hashlib, importlib.util, json, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
REPLAY = ROOT / "competition/vfarl/replay"
ART = REPLAY / "artifacts/f1-new"
REV = "77cab9055d42ab4a5c25c89a8f937096db13558e"
DATASET_README = "competition/vfarl/acquisition/recover-semantics/sources/SWE-smith-py-README.txt"

PLANS = [    dict(
        iid="oauthlib__oauthlib.1fd52536.func_basic__n1tdsvkt",
        key="f1-new-oauthlib-legacy-client-id",
        canonical_repo="oauthlib/oauthlib",
        primary_adapter="F", submechanism="F1",
        family=("oauthlib.oauth2.legacy_application.LegacyApplicationClient.__init__"
                "::client_id_super_forwarding"),
        prod_path="oauthlib/oauth2/rfc6749/clients/legacy_application.py",
        prod_module="oauthlib.oauth2.rfc6749.clients.legacy_application",
        profile=".venv-f1-oauthlib",
        old="        super().__init__(**kwargs)\n",
        new="        super().__init__(client_id, **kwargs)\n",
        read_start=34, read_end=44,
        mechanism={
            "trigger": "Constructing LegacyApplicationClient with a client_id and preparing a password-grant body",
            "base_wrong_behavior": "client_id never reaches the parent Client constructor, so construction raises TypeError and no client_id attribute exists",
            "production_symbols": ["oauthlib.oauth2.rfc6749.clients.legacy_application.LegacyApplicationClient.__init__"],
            "causal_change": "Restore client_id as the first positional argument of super().__init__; kwargs forwarding unchanged",
            "discriminating_observation": {
                "kind": "executed_before_after_source_observation",
                "description": "LegacyApplicationClient('my_client_id').client_id must equal 'my_client_id' and prepare_request_body(username, password) must return the password-grant body. Before repair construction raises TypeError; after repair both checks pass.",
            },
            "bug_patch_direction": "introduces_defect",
        },
        observation=(
            "from oauthlib.oauth2 import LegacyApplicationClient\n"
            "c = LegacyApplicationClient('my_client_id')\n"
            "print('client_id', c.client_id)\n"
            "assert c.client_id == 'my_client_id'\n"
            "body = c.prepare_request_body(username='u', password='p')\n"
            "print('body', body)\n"
            "assert 'grant_type=password' in body and 'username=u' in body\n"
        ),
        review_notes=("F1 config-propagation: entry constructor client_id -> middle "
                      "LegacyApplicationClient.__init__ -> consumer parent Client storage and "
                      "prepare_request_body. The bug drops the client_id edge at the super() call. "
                      "Single production line restored; original F2P/P2P and full target files unchanged."),
        final_response=("The client_id is forwarded to the parent Client again; "
                        "the actual source-bound observation completed and the patch was submitted."),
    ),
    dict(
        iid="oauthlib__oauthlib.1fd52536.func_basic__q553vi4s",
        key="f1-new-oauthlib-dispatcher-grants",
        canonical_repo="oauthlib/oauthlib",
        primary_adapter="F", submechanism="F1",
        family=("oauthlib.openid.dispatchers.AuthorizationCodeGrantDispatcher.__init__"
                "::grant_handler_assignment"),
        prod_path="oauthlib/openid/connect/core/grant_types/dispatchers.py",
        prod_module="oauthlib.openid.connect.core.grant_types.dispatchers",
        profile=".venv-f1-oauthlib",
        old=("    def __init__(self, default_grant=None, oidc_grant=None):\n"
             "        self.default_grant = oidc_grant\n"
             "        self.oidc_grant = default_grant\n"),
        new=("    def __init__(self, default_grant=None, oidc_grant=None):\n"
             "        self.default_grant = default_grant\n"
             "        self.oidc_grant = oidc_grant\n"),
        read_start=11, read_end=26,
        mechanism={
            "trigger": "Constructing AuthorizationCodeGrantDispatcher with distinct default and OIDC grant handlers",
            "base_wrong_behavior": "The two handlers are stored into each other's attributes, so later dispatch selects the wrong grant",
            "production_symbols": ["oauthlib.openid.connect.core.grant_types.dispatchers.AuthorizationCodeGrantDispatcher.__init__"],
            "causal_change": "Restore each constructor argument to its own attribute; dispatch selection logic untouched",
            "discriminating_observation": {
                "kind": "executed_before_after_source_observation",
                "description": "AuthorizationCodeGrantDispatcher(default_grant='D', oidc_grant='O') must expose .default_grant=='D' and .oidc_grant=='O'. Before repair both asserts fail (swapped); after repair they pass.",
            },
            "bug_patch_direction": "introduces_defect",
        },
        observation=(
            "from oauthlib.openid.connect.core.grant_types.dispatchers import AuthorizationCodeGrantDispatcher\n"
            "d = AuthorizationCodeGrantDispatcher(default_grant='D', oidc_grant='O')\n"
            "print('grants', d.default_grant, d.oidc_grant)\n"
            "assert d.default_grant == 'D' and d.oidc_grant == 'O'\n"
        ),
        review_notes=("F1 config-propagation: entry constructor handler pair -> middle dispatcher "
                      "__init__ storage -> consumer _handler_for_request selection. The bug cross-wires "
                      "the two handler edges. Distinct production class from the oauth2 client "
                      "constructors. Single hunk restored; original requirements unchanged."),
        final_response=("Each grant handler is stored under its own attribute again; "
                        "the actual source-bound observation completed and the patch was submitted."),
    ),
    dict(
        iid="oauthlib__oauthlib.1fd52536.func_basic__qpiwixs0",
        key="f1-new-oauthlib-device-secret",
        canonical_repo="oauthlib/oauthlib",
        primary_adapter="F", submechanism="F1",
        family=("oauthlib.oauth2.device.DeviceClient.__init__"
                "::client_secret_publication"),
        prod_path="oauthlib/oauth2/rfc8628/clients/device.py",
        prod_module="oauthlib.oauth2.rfc8628.clients.device",
        profile=".venv-f1-oauthlib",
        old="        client_secret = kwargs.get('client_secret', 'default_secret')\n",
        new="        self.client_secret = kwargs.get('client_secret')\n",
        read_start=28, read_end=38,
        mechanism={
            "trigger": "Constructing DeviceClient with a client_secret keyword",
            "base_wrong_behavior": "The secret is kept in a discarded local with a wrong default instead of being published as an instance attribute, so later reads fail",
            "production_symbols": ["oauthlib.oauth2.rfc8628.clients.device.DeviceClient.__init__"],
            "causal_change": "Restore publication of kwargs client_secret to self.client_secret with a None default; request-uri logic untouched",
            "discriminating_observation": {
                "kind": "executed_before_after_source_observation",
                "description": "DeviceClient('cid', client_secret='s3cr3t').client_secret must equal 's3cr3t'. Before repair attribute access raises AttributeError; after repair it passes.",
            },
            "bug_patch_direction": "introduces_defect",
        },
        observation=(
            "from oauthlib.oauth2.rfc8628.clients.device import DeviceClient\n"
            "c = DeviceClient('cid', client_secret='s3cr3t')\n"
            "print('secret', c.client_secret)\n"
            "assert c.client_secret == 's3cr3t'\n"
        ),
        review_notes=("F1 config-propagation: entry kwargs client_secret -> middle DeviceClient.__init__ "
                      "-> consumer prepare_request_uri secret lookup. The bug drops the self-publication "
                      "edge. Single production line restored; original requirements unchanged."),
        final_response=("The client_secret is published on the instance again; "
                        "the actual source-bound observation completed and the patch was submitted."),
    ),
    dict(
        iid="oauthlib__oauthlib.1fd52536.func_basic__maxi5yta",
        key="f1-new-oauthlib-implicit-handler",
        canonical_repo="oauthlib/oauthlib",
        primary_adapter="F", submechanism="F1",
        family=("oauthlib.oauth2.implicit.ImplicitGrant.create_authorization_response"
                "::token_handler_forwarding"),
        prod_path="oauthlib/oauth2/rfc6749/grant_types/implicit.py",
        prod_module="oauthlib.oauth2.rfc6749.grant_types.implicit",
        profile=".venv-f1-oauthlib",
        old=("        new_handler = None  # Assigning None could subtly override token_handler\n"
             "        return self.create_token_response(request, new_handler)\n"),
        new="        return self.create_token_response(request, token_handler)\n",
        read_start=158, read_end=168,
        mechanism={
            "trigger": "Creating an implicit-grant token response with an explicit BearerToken handler",
            "base_wrong_behavior": "The provided token_handler is replaced by None before the downstream call, so token creation raises AttributeError",
            "production_symbols": ["oauthlib.oauth2.rfc6749.grant_types.implicit.ImplicitGrant.create_authorization_response"],
            "causal_change": "Forward the received token_handler unchanged to create_token_response; token creation logic untouched",
            "discriminating_observation": {
                "kind": "executed_before_after_source_observation",
                "description": "create_authorization_response on a valid implicit request with a BearerToken handler must return 302 with access_token in the fragment Location. Before repair it raises AttributeError on None.create_token; after repair it passes.",
            },
            "bug_patch_direction": "introduces_defect",
        },
        observation=(
            "from unittest import mock\n"
            "from oauthlib.common import Request\n"
            "from oauthlib.oauth2.rfc6749.grant_types import ImplicitGrant\n"
            "from oauthlib.oauth2.rfc6749.tokens import BearerToken\n"
            "v = mock.MagicMock()\n"
            "g = ImplicitGrant(request_validator=v)\n"
            "req = Request('http://a.b/path')\n"
            "req.scopes = ('hello', 'world')\n"
            "req.client = mock.MagicMock()\n"
            "req.client_id = 'abcdef'\n"
            "req.response_type = 'token'\n"
            "req.state = 'xyz'\n"
            "req.redirect_uri = 'https://b.c/p'\n"
            "bearer = BearerToken(v, expires_in=1800)\n"
            "h, b, s = g.create_authorization_response(req, bearer)\n"
            "print('status', s, 'location', h['Location'][:80])\n"
            "assert s == 302 and 'access_token=' in h['Location']\n"
        ),
        review_notes=("F1 config-propagation: entry token_handler argument -> middle "
                      "ImplicitGrant.create_authorization_response -> consumer create_token_response "
                      "token creation. The bug severs the handler edge with None. Distinct production "
                      "function from the client constructors. Hunk restored; original requirements unchanged."),
        final_response=("The token handler is forwarded to token creation again; "
                        "the actual source-bound observation completed and the patch was submitted."),
    ),
    dict(
        iid="oauthlib__oauthlib.1fd52536.func_basic__zrajljoi",
        key="f1-new-oauthlib-webapp-wiring",
        canonical_repo="oauthlib/oauthlib",
        primary_adapter="F", submechanism="F1",
        family=("oauthlib.oauth2.web_application.WebApplicationClient.__init__"
                "::client_id_code_wiring"),
        prod_path="oauthlib/oauth2/rfc6749/clients/web_application.py",
        prod_module="oauthlib.oauth2.rfc6749.clients.web_application",
        profile=".venv-f1-oauthlib",
        old=("        super().__init__(code, **kwargs)\n"
             "        self.code = client_id\n"),
        new=("        super().__init__(client_id, **kwargs)\n"
             "        self.code = code\n"),
        read_start=35, read_end=46,
        mechanism={
            "trigger": "Constructing WebApplicationClient with a client_id and an authorization code",
            "base_wrong_behavior": "client_id and code are cross-wired: the code reaches the parent constructor while client_id lands in self.code",
            "production_symbols": ["oauthlib.oauth2.rfc6749.clients.web_application.WebApplicationClient.__init__"],
            "causal_change": "Restore client_id to super().__init__ and code to self.code; request builders untouched",
            "discriminating_observation": {
                "kind": "executed_before_after_source_observation",
                "description": "WebApplicationClient('my_id', code='authcode') must expose .client_id=='my_id' and .code=='authcode'. Before repair both asserts fail (cross-wired); after repair they pass.",
            },
            "bug_patch_direction": "introduces_defect",
        },
        observation=(
            "from oauthlib.oauth2 import WebApplicationClient\n"
            "c = WebApplicationClient('my_id', code='authcode')\n"
            "print('wired', c.client_id, c.code)\n"
            "assert c.client_id == 'my_id' and c.code == 'authcode'\n"
        ),
        review_notes=("F1 config-propagation: entry (client_id, code) pair -> middle "
                      "WebApplicationClient.__init__ -> consumers parent Client storage and self.code. "
                      "The bug cross-wires both edges (drop-plus-misroute). Distinct production class "
                      "from LegacyApplicationClient (drop-only). Hunk restored; requirements unchanged."),
        final_response=("The client_id and code are wired to their own slots again; "
                        "the actual source-bound observation completed and the patch was submitted."),
    ),
    dict(
        iid="oauthlib__oauthlib.1fd52536.func_basic__ty24mp1h",
        key="f1-new-oauthlib-bearer-order",
        canonical_repo="oauthlib/oauthlib",
        primary_adapter="F", submechanism="F1",
        family=("oauthlib.oauth2.tokens.BearerToken.validate_request"
                "::validator_argument_order"),
        prod_path="oauthlib/oauth2/rfc6749/tokens.py",
        prod_module="oauthlib.oauth2.rfc6749.tokens",
        profile=".venv-f1-oauthlib",
        old="            token, request, request.scopes)\n",
        new="            token, request.scopes, request)\n",
        read_start=333, read_end=343,
        mechanism={
            "trigger": "Validating a bearer token from an Authorization header with request scopes",
            "base_wrong_behavior": "The validator receives (token, request, scopes) against its (token, scopes, request) contract, so scopes and request are swapped at the call edge",
            "production_symbols": ["oauthlib.oauth2.rfc6749.tokens.BearerToken.validate_request"],
            "causal_change": "Restore the (token, scopes, request) argument order at the validator call; header extraction untouched",
            "discriminating_observation": {
                "kind": "executed_before_after_source_observation",
                "description": "A capturing validator with the real (token, scopes, request) signature must observe token='tok123', scopes=['s'] and the identical request object. Before repair scopes/request are swapped; after repair all three asserts pass.",
            },
            "bug_patch_direction": "introduces_defect",
        },
        observation=(
            "from oauthlib.common import Request\n"
            "from oauthlib.oauth2.rfc6749.tokens import BearerToken\n"
            "seen = {}\n"
            "class V:\n"
            "    def validate_bearer_token(self, token, scopes, request):\n"
            "        seen.update(token=token, scopes=scopes, request=request)\n"
            "        return True\n"
            "req = Request('https://example.com/res', headers={'Authorization': 'Bearer tok123'})\n"
            "req.scopes = ['s']\n"
            "t = BearerToken(request_validator=V())\n"
            "assert t.validate_request(req) is True\n"
            "print('observed', seen['token'], seen['scopes'], seen['request'] is req)\n"
            "assert seen['token'] == 'tok123' and seen['scopes'] == ['s'] and seen['request'] is req\n"
        ),
        review_notes=("F1 config-propagation: entry (token, scopes, request) triple -> middle "
                      "BearerToken.validate_request extraction -> consumer request_validator call. "
                      "The bug swaps two positional channels at the call edge. Single line restored; "
                      "requirements unchanged."),
        final_response=("The validator arguments are ordered (token, scopes, request) again; "
                        "the actual source-bound observation completed and the patch was submitted."),
    ),
    # ---- batch 2: arrow (4) ----
    dict(
        iid="arrow-py__arrow.1d70d009.func_basic__6c49cv3v",
        key="f1-new-arrow-now-tz",
        canonical_repo="arrow-py/arrow",
        primary_adapter="F", submechanism="F1",
        family="arrow.api.now::tz_forwarding",
        prod_path="arrow/api.py", prod_module="arrow.api",
        profile=".venv-f1-arrow",
        old="    return _factory.now() if tz is None else _factory.now(tz.tzname(None))\n",
        new="    return _factory.now(tz)\n",
        read_start=100, read_end=112,
        mechanism={
            "trigger": "Calling arrow.now with a timezone string such as 'UTC'",
            "base_wrong_behavior": "The tz argument is replaced by a tzname() call on it, so string timezones raise AttributeError instead of reaching the factory",
            "production_symbols": ["arrow.api.now"],
            "causal_change": "Forward tz unchanged to ArrowFactory.now; factory parsing untouched",
            "discriminating_observation": {
                "kind": "executed_before_after_source_observation",
                "description": "arrow.now('UTC') must return an Arrow whose tzinfo is tzutc(). Before repair it raises AttributeError on str.tzname; after repair it passes.",
            },
            "bug_patch_direction": "introduces_defect",
        },
        observation=(
            "import arrow\n"
            "from dateutil import tz\n"
            "a = arrow.now('UTC')\n"
            "print('now', a, a.tzinfo)\n"
            "assert a.tzinfo == tz.tzutc()\n"
        ),
        review_notes=("F1 config-propagation: entry tz argument -> middle arrow.api.now -> consumer "
                      "ArrowFactory.now parsing. The bug corrupts the tz edge with a tzname() call. "
                      "Single line restored; requirements unchanged."),
        final_response=("The timezone argument is forwarded to the factory unchanged; "
                        "the actual source-bound observation completed and the patch was submitted."),
    ),
    dict(
        iid="arrow-py__arrow.1d70d009.func_basic__luco0gci",
        key="f1-new-arrow-factory-type",
        canonical_repo="arrow-py/arrow",
        primary_adapter="F", submechanism="F1",
        family="arrow.api.factory::type_forwarding",
        prod_path="arrow/api.py", prod_module="arrow.api",
        profile=".venv-f1-arrow",
        old="    return ArrowFactory()  # misplaced type argument intentionally omitted\n",
        new="    return ArrowFactory(type)\n",
        read_start=115, read_end=124,
        mechanism={
            "trigger": "Building a factory with a custom Arrow subclass via arrow.api.factory",
            "base_wrong_behavior": "The type argument never reaches ArrowFactory, so the factory builds base Arrow instances instead of the custom class",
            "production_symbols": ["arrow.api.factory"],
            "causal_change": "Forward type to the ArrowFactory constructor; factory internals untouched",
            "discriminating_observation": {
                "kind": "executed_before_after_source_observation",
                "description": "api.factory(CustomArrow).type must be CustomArrow and .get() must return a CustomArrow instance. Before repair the type edge is dropped; after repair both pass.",
            },
            "bug_patch_direction": "introduces_defect",
        },
        observation=(
            "import arrow\n"
            "from arrow import api\n"
            "class CustomArrow(arrow.Arrow):\n"
            "    pass\n"
            "f = api.factory(CustomArrow)\n"
            "print('factory type', f.type)\n"
            "assert f.type is CustomArrow\n"
            "r = f.get('2020-01-01')\n"
            "assert type(r) is CustomArrow\n"
        ),
        review_notes=("F1 config-propagation: entry custom Arrow type -> middle arrow.api.factory -> "
                      "consumer ArrowFactory construction. The bug drops the type edge. Single line "
                      "restored; requirements unchanged."),
        final_response=("The custom Arrow type reaches the factory again; "
                        "the actual source-bound observation completed and the patch was submitted."),
    ),
    dict(
        iid="arrow-py__arrow.1d70d009.func_basic__p9zbp6c3",
        key="f1-new-arrow-utcnow-dispatch",
        canonical_repo="arrow-py/arrow",
        primary_adapter="F", submechanism="F1",
        family="arrow.api.utcnow::factory_dispatch",
        prod_path="arrow/api.py", prod_module="arrow.api",
        profile=".venv-f1-arrow",
        old="    return _factory.now()\n",
        new="    return _factory.utcnow()\n",
        read_start=95, read_end=104,
        mechanism={
            "trigger": "Calling arrow.utcnow() for a UTC timestamp",
            "base_wrong_behavior": "The call is dispatched to the local-time factory method, so the UTC edge never reaches the consumer",
            "production_symbols": ["arrow.api.utcnow"],
            "causal_change": "Restore dispatch to ArrowFactory.utcnow; factory methods untouched",
            "discriminating_observation": {
                "kind": "executed_before_after_source_observation",
                "description": "arrow.utcnow().tzinfo must equal dateutil tzutc(). Before repair it is the local tz (never identical to tzutc); after repair it passes.",
            },
            "bug_patch_direction": "introduces_defect",
        },
        observation=(
            "import arrow\n"
            "from dateutil import tz\n"
            "u = arrow.utcnow()\n"
            "print('utcnow', u, u.tzinfo)\n"
            "assert u.tzinfo == tz.tzutc()\n"
        ),
        review_notes=("F1 config-propagation: entry utcnow() -> middle arrow.api.utcnow -> consumer "
                      "ArrowFactory.utcnow. The bug misroutes the call to the local-time edge. Single "
                      "line restored; requirements unchanged."),
        final_response=("utcnow dispatches to the UTC factory method again; "
                        "the actual source-bound observation completed and the patch was submitted."),
    ),
    dict(
        iid="arrow-py__arrow.1d70d009.func_basic__72evk1eu",
        key="f1-new-arrow-format-wiring",
        canonical_repo="arrow-py/arrow",
        primary_adapter="F", submechanism="F1",
        family="arrow.Arrow.__format__::formatter_locale_wiring",
        prod_path="arrow/arrow.py", prod_module="arrow.arrow",
        profile=".venv-f1-arrow",
        old="        return formatter.DateTimeFormatter(fmt).format(self._datetime, locale)\n",
        new="        return formatter.DateTimeFormatter(locale).format(self._datetime, fmt)\n",
        read_start=1115, read_end=1125,
        mechanism={
            "trigger": "Formatting an Arrow with an explicit format string and locale",
            "base_wrong_behavior": "The format string and locale are cross-wired at both the formatter construction and the format call edges",
            "production_symbols": ["arrow.arrow.Arrow.__format__"],
            "causal_change": "Restore locale to the DateTimeFormatter constructor and fmt to the format call; formatter internals untouched",
            "discriminating_observation": {
                "kind": "executed_before_after_source_observation",
                "description": "Arrow(2020,1,2,3,4,5).format('YYYY-MM-DD') must equal '2020-01-02'. Before repair the cross-wired channels break formatting; after repair it passes.",
            },
            "bug_patch_direction": "introduces_defect",
        },
        observation=(
            "import arrow\n"
            "a = arrow.Arrow(2020, 1, 2, 3, 4, 5)\n"
            "s = a.format('YYYY-MM-DD')\n"
            "print('formatted', s)\n"
            "assert s == '2020-01-02'\n"
        ),
        review_notes=("F1 config-propagation: entry (fmt, locale) pair -> middle Arrow.__format__ -> "
                      "consumers DateTimeFormatter construction and format call. The bug cross-wires "
                      "both edges. Distinct production function from the api.py forwarders. Hunk "
                      "restored; requirements unchanged."),
        final_response=("The formatter and locale channels are wired correctly again; "
                        "the actual source-bound observation completed and the patch was submitted."),
    ),
    # ---- batch 2: markupsafe (4) ----
    dict(
        iid="pallets__markupsafe.620c06c9.func_basic__qyy4vtv6",
        key="f1-new-markupsafe-format-map",
        canonical_repo="pallets/markupsafe",
        primary_adapter="F", submechanism="F1",
        family="markupsafe.Markup.format_map::vformat_argument_order",
        prod_path="src/markupsafe/__init__.py", prod_module="markupsafe",
        profile=".venv-f1-markupsafe", source_layout="src", build_ext=True,
        license_file="LICENSE.txt",
        old="        return self.__class__(formatter.vformat(self, mapping, ()))\n",
        new="        return self.__class__(formatter.vformat(self, (), mapping))\n",
        read_start=317, read_end=327,
        mechanism={
            "trigger": "Markup.format_map with a value mapping needing escaping",
            "base_wrong_behavior": "The mapping and empty-args channels are swapped at the vformat call edge, so keyword interpolation breaks",
            "production_symbols": ["markupsafe.Markup.format_map"],
            "causal_change": "Restore the (args=(), kwargs=mapping) order at the vformat call; formatter untouched",
            "discriminating_observation": {
                "kind": "executed_before_after_source_observation",
                "description": "Markup('<em>{x}</em>').format_map({'x': '<bad>'}) must equal the escaped Markup. Before repair the swapped channels break interpolation; after repair it passes.",
            },
            "bug_patch_direction": "introduces_defect",
        },
        observation=(
            "from markupsafe import Markup\n"
            "out = Markup('<em>{x}</em>').format_map({'x': '<bad>'})\n"
            "print('mapped', out)\n"
            "assert out == '<em>&lt;bad&gt;</em>'\n"
        ),
        review_notes=("F1 config-propagation: entry value mapping -> middle Markup.format_map -> "
                      "consumer string.Formatter.vformat. The bug swaps the positional channels at the "
                      "call edge. Single line restored; requirements unchanged. C extension prebuilt "
                      "from pristine sources so both native/speedups variants execute."),
        final_response=("The vformat argument order is restored; "
                        "the actual source-bound observation completed and the patch was submitted."),
    ),
    dict(
        iid="pallets__markupsafe.620c06c9.func_basic__s2pysxd9",
        key="f1-new-markupsafe-split",
        canonical_repo="pallets/markupsafe",
        primary_adapter="F", submechanism="F1",
        family="markupsafe.Markup.split::maxsplit_forwarding",
        prod_path="src/markupsafe/__init__.py", prod_module="markupsafe",
        profile=".venv-f1-markupsafe", source_layout="src", build_ext=True,
        license_file="LICENSE.txt",
        old="        return [self.__class__(v) for v in super().split(sep, maxsplit + 1)]\n",
        new="        return [self.__class__(v) for v in super().split(sep, maxsplit)]\n",
        read_start=173, read_end=183,
        mechanism={
            "trigger": "Markup.split with an explicit maxsplit",
            "base_wrong_behavior": "maxsplit is incremented before reaching str.split, so one extra split happens downstream",
            "production_symbols": ["markupsafe.Markup.split"],
            "causal_change": "Forward maxsplit unchanged to str.split; Markup wrapping untouched",
            "discriminating_observation": {
                "kind": "executed_before_after_source_observation",
                "description": "Markup('a,b,c').split(',', 1) must equal ['a', 'b,c']. Before repair the off-by-one edge yields three parts; after repair it passes.",
            },
            "bug_patch_direction": "introduces_defect",
        },
        observation=(
            "from markupsafe import Markup\n"
            "parts = Markup('a,b,c').split(',', 1)\n"
            "print('parts', parts)\n"
            "assert parts == ['a', 'b,c']\n"
        ),
        review_notes=("F1 config-propagation: entry maxsplit -> middle Markup.split -> consumer "
                      "str.split. The bug alters the value at the forwarding edge. Single line "
                      "restored; requirements unchanged."),
        final_response=("maxsplit is forwarded unchanged again; "
                        "the actual source-bound observation completed and the patch was submitted."),
    ),
    dict(
        iid="pallets__markupsafe.620c06c9.func_basic__vdbfno8m",
        key="f1-new-markupsafe-escape-formatter",
        canonical_repo="pallets/markupsafe",
        primary_adapter="F", submechanism="F1",
        family="markupsafe.EscapeFormatter.__init__::escape_publication",
        prod_path="src/markupsafe/__init__.py", prod_module="markupsafe",
        profile=".venv-f1-markupsafe", source_layout="src", build_ext=True,
        license_file="LICENSE.txt",
        old="        self.escape: _TPEscape = None\n",
        new="        self.escape: _TPEscape = escape\n",
        read_start=332, read_end=342,
        mechanism={
            "trigger": "Formatting a Markup template with an EscapeFormatter carrying Markup.escape",
            "base_wrong_behavior": "The escape callable never reaches the instance attribute, so field formatting raises TypeError on None instead of escaping",
            "production_symbols": ["markupsafe.EscapeFormatter.__init__"],
            "causal_change": "Restore publication of the escape argument to self.escape; format dispatch untouched",
            "discriminating_observation": {
                "kind": "executed_before_after_source_observation",
                "description": "Markup('<em>{x}</em>').format(x='<bad>') must equal the escaped Markup. Before repair the missing escape edge raises; after repair it passes.",
            },
            "bug_patch_direction": "introduces_defect",
        },
        observation=(
            "from markupsafe import Markup\n"
            "out = Markup('<em>{x}</em>').format(x='<bad>')\n"
            "print('formatted', out)\n"
            "assert out == '<em>&lt;bad&gt;</em>'\n"
        ),
        review_notes=("F1 config-propagation: entry escape callable -> middle EscapeFormatter.__init__ "
                      "-> consumer format_field. The bug drops the self-publication edge. Single line "
                      "restored; requirements unchanged."),
        final_response=("The escape callable is published on the formatter again; "
                        "the actual source-bound observation completed and the patch was submitted."),
    ),
    dict(
        iid="pallets__markupsafe.620c06c9.func_basic__w4wiqdd5",
        key="f1-new-markupsafe-helper-wiring",
        canonical_repo="pallets/markupsafe",
        primary_adapter="F", submechanism="F1",
        family="markupsafe._MarkupEscapeHelper.__init__::obj_escape_wiring",
        prod_path="src/markupsafe/__init__.py", prod_module="markupsafe",
        profile=".venv-f1-markupsafe", source_layout="src", build_ext=True,
        license_file="LICENSE.txt",
        old=("        self.obj: _TPEscape = escape\n"
             "        self.escape: t.Any = obj\n"),
        new=("        self.obj: t.Any = obj\n"
             "        self.escape: _TPEscape = escape\n"),
        read_start=357, read_end=368,
        mechanism={
            "trigger": "Percent-formatting a Markup template with a value mapping",
            "base_wrong_behavior": "obj and escape are cross-wired in the helper constructor, so item lookup and escaping operate on each other's channels",
            "production_symbols": ["markupsafe._MarkupEscapeHelper.__init__"],
            "causal_change": "Restore obj to self.obj and escape to self.escape; helper accessors untouched",
            "discriminating_observation": {
                "kind": "executed_before_after_source_observation",
                "description": "Markup('<em>%(name)s</em>') % {'name': '<bad>'} must equal the escaped Markup. Before repair the cross-wired channels break interpolation; after repair it passes.",
            },
            "bug_patch_direction": "introduces_defect",
        },
        observation=(
            "from markupsafe import Markup\n"
            "out = Markup('<em>%(name)s</em>') % {'name': '<bad>'}\n"
            "print('interpolated', out)\n"
            "assert out == '<em>&lt;bad&gt;</em>'\n"
        ),
        review_notes=("F1 config-propagation: entry (obj, escape) pair -> middle "
                      "_MarkupEscapeHelper.__init__ -> consumer __getitem__/__str__ escaping. The bug "
                      "cross-wires both edges (drop-plus-misroute). Distinct production class from "
                      "EscapeFormatter (drop-only). Hunk restored; requirements unchanged."),
        final_response=("The helper channels are wired to their own slots again; "
                        "the actual source-bound observation completed and the patch was submitted."),
    ),
    # ---- batch 2: faker (1) ----
    # ---- batch 2: schema (2) ----
    dict(
        iid="keleshev__schema.24a30457.func_basic__o5zoj22w",
        key="f1-new-schema-kwargs-forward",
        canonical_repo="keleshev/schema",
        primary_adapter="F", submechanism="F1",
        family="schema._invoke_with_optional_kwargs::kwargs_forwarding",
        prod_path="schema/__init__.py", prod_module="schema",
        profile=".venv-f1-schema", license_file="LICENSE-MIT",
        old="    return f(*kwargs)\n",
        new="    return f(**kwargs)\n",
        read_start=340, read_end=350,
        mechanism={
            "trigger": "Validating with keyword arguments that reach a callable Optional default",
            "base_wrong_behavior": "The kwargs mapping is exploded into positional keys at the default-callable edge, so the consumer receives garbage positionals instead of keywords",
            "production_symbols": ["schema._invoke_with_optional_kwargs"],
            "causal_change": "Restore **kwargs unpacking at the default-callable call; signature probing untouched",
            "discriminating_observation": {
                "kind": "executed_before_after_source_observation",
                "description": "_invoke_with_optional_kwargs(f, name='World') must return 'World' and zero-arg callables still work. Before repair the exploded positionals raise TypeError; after repair both pass.",
            },
            "bug_patch_direction": "introduces_defect",
        },
        observation=(
            "from schema import _invoke_with_optional_kwargs\n"
            "def f(**kw):\n"
            "    return kw.get('name', 'Unknown')\n"
            "assert _invoke_with_optional_kwargs(f, name='World') == 'World'\n"
            "def g():\n"
            "    return 'zero'\n"
            "assert _invoke_with_optional_kwargs(g) == 'zero'\n"
            "print('kwargs forwarded')\n"
        ),
        review_notes=("F1 config-propagation: entry validate kwargs -> middle "
                      "_invoke_with_optional_kwargs -> consumer default callable. The bug destroys the "
                      "kwargs channel (* instead of **). Single line restored; requirements unchanged."),
        final_response=("Keyword arguments reach the default callable again; "
                        "the actual source-bound observation completed and the patch was submitted."),
    ),
    dict(
        iid="keleshev__schema.24a30457.func_basic__y6pl4w6n",
        key="f1-new-schema-and-wiring",
        canonical_repo="keleshev/schema",
        primary_adapter="F", submechanism="F1",
        family="schema.And._build_schema::error_flag_wiring",
        prod_path="schema/__init__.py", prod_module="schema",
        profile=".venv-f1-schema", license_file="LICENSE-MIT",
        old=("    def _build_schema(self, arg: Any) -> TSchema:\n"
             "        return self._schema_class(\n"
             "            arg, error=self._ignore_extra_keys, ignore_extra_keys=self._error\n"
             "        )\n"),
        new=("    def _build_schema(self, arg: Any) -> TSchema:\n"
             "        # Assume self._schema_class(arg, ...) returns an instance of TSchema\n"
             "        return self._schema_class(\n"
             "            arg, error=self._error, ignore_extra_keys=self._ignore_extra_keys\n"
             "        )\n"),
        read_start=163, read_end=175,
        mechanism={
            "trigger": "Validating with And carrying both a custom error and ignore_extra_keys",
            "base_wrong_behavior": "error and ignore_extra_keys are cross-wired at the subschema-construction edge, so a bool reaches .format and the flag is lost",
            "production_symbols": ["schema.And._build_schema"],
            "causal_change": "Restore error and ignore_extra_keys to their own keyword channels; subschema construction untouched",
            "discriminating_observation": {
                "kind": "executed_before_after_source_observation",
                "description": "Schema(And({'key': str}, ignore_extra_keys=True, error='Custom error')).validate({'key': 'value', 'extra': 'ignored'}) must return the mapping. Before repair the cross-wired channels raise; after repair it passes.",
            },
            "bug_patch_direction": "introduces_defect",
        },
        observation=(
            "from schema import Schema, And\n"
            "s = Schema(And({'key': str}, ignore_extra_keys=True, error='Custom error'))\n"
            "out = s.validate({'key': 'value', 'extra': 'ignored'})\n"
            "print('validated', out)\n"
            "assert out == {'key': 'value'}\n"
        ),
        review_notes=("F1 config-propagation: entry (error, ignore_extra_keys) pair -> middle "
                      "And._build_schema -> consumer subschema constructor. The bug cross-wires both "
                      "edges. Distinct production function from _invoke_with_optional_kwargs (drop at "
                      "call edge). Line restored; requirements unchanged."),
        final_response=("The error and flag channels reach the subschema again; "
                        "the actual source-bound observation completed and the patch was submitted."),
    ),
    # ---- batch 2: parsimonious (2) ----
    # ---- batch 2: soupsieve (1) ----
    dict(
        iid="facelessuser__soupsieve.a8080d97.func_basic__pzcgtq1e",
        key="f1-new-soupsieve-match-tag",
        canonical_repo="facelessuser/soupsieve",
        primary_adapter="F", submechanism="F1",
        family="soupsieve.SoupSieve.match::tag_forwarding",
        prod_path="soupsieve/css_match.py", prod_module="soupsieve.css_match",
        profile=".venv-f1-soup39", task_python="/usr/bin/python3", task_python39=True,
        license_file="LICENSE.md",
        old="        return CSSMatch(self.selectors, tag, self.namespaces, self.flags).match(self.flags)\n",
        new="        return CSSMatch(self.selectors, tag, self.namespaces, self.flags).match(tag)\n",
        read_start=1525, read_end=1535,
        mechanism={
            "trigger": "Testing whether a tag matches a compiled selector via SoupSieve.match",
            "base_wrong_behavior": "The tag edge is replaced by the flags int at the final matcher call, so matching operates on the wrong channel",
            "production_symbols": ["soupsieve.css_match.SoupSieve.match"],
            "causal_change": "Restore tag as the argument of the CSSMatch.match call; selector compilation untouched",
            "discriminating_observation": {
                "kind": "executed_before_after_source_observation",
                "description": "A compiled 'span.a' pattern must match the span tag and reject the p tag. Before repair the flags-valued call breaks matching; after repair both pass.",
            },
            "bug_patch_direction": "introduces_defect",
        },
        observation=(
            "import soupsieve as sv\n"
            "from bs4 import BeautifulSoup\n"
            "soup = BeautifulSoup('<p id=\"1\"><span id=\"2\" class=\"a\"></span></p>', 'html.parser')\n"
            "pat = sv.compile('span.a')\n"
            "assert len(pat.select(soup)) == 1\n"
            "assert pat.match(soup.span) is True\n"
            "assert pat.match(soup.p) is False\n"
            "print('match channels ok')\n"
        ),
        review_notes=("F1 config-propagation: entry tag -> middle SoupSieve.match -> consumer "
                      "CSSMatch.match. The bug swaps the tag channel for flags at the final call edge. "
                      "Single line restored; requirements unchanged. Verified under the pinned task "
                      "interpreter CPython 3.9.6 because later html.parser releases parse iframe "
                      "content as text while the smith-era suite expects tags; profile supplies "
                      "pytest/beautifulsoup4/lxml/html5lib for 3.9."),
        final_response=("The tag reaches the matcher again; "
                        "the actual source-bound observation completed and the patch was submitted."),
    ),
    # ---- batch 3: thefuzz (3) ----
    dict(
        iid="seatgeek__thefuzz.8a05a3ee.func_basic__3yvg2tul",
        key="f1-new-thefuzz-processor-order",
        canonical_repo="seatgeek/thefuzz",
        primary_adapter="F", submechanism="F1",
        family="thefuzz.process._get_processor::processor_composition_order",
        prod_path="thefuzz/process.py", prod_module="thefuzz.process",
        profile=".venv-f1-thefuzz", license_file="LICENSE.txt",
        old="        return processor(pre_processor(s))\n",
        new="        return pre_processor(processor(s))\n",
        read_start=28, read_end=40,
        mechanism={
            "trigger": "Wrapping a custom processor together with the default preprocessor for WRatio scoring",
            "base_wrong_behavior": "The custom processor and the default preprocessor are composed in the wrong order, so user preprocessing runs on already-normalized text instead of raw input",
            "production_symbols": ["thefuzz.process._get_processor"],
            "causal_change": "Restore pre_processor(processor(s)) composition inside the wrapper; scorer gating untouched",
            "discriminating_observation": {
                "kind": "executed_before_after_source_observation",
                "description": "A recording custom processor must observe the raw string and the wrapper output must equal full_process of the uppercased input. Before repair the order swap breaks both asserts; after repair they pass.",
            },
            "bug_patch_direction": "introduces_defect",
        },
        observation=(
            "from thefuzz import process, fuzz\n"
            "from thefuzz import utils\n"
            "seen = []\n"
            "def my_proc(s):\n"
            "    seen.append(s)\n"
            "    return s.upper()\n"
            "w = process._get_processor(my_proc, fuzz.WRatio)\n"
            "out = w('  hello world  ')\n"
            "print('wrapped', repr(out), seen)\n"
            "assert seen == ['  hello world  ']\n"
            "assert out == utils.full_process('  HELLO WORLD  ', force_ascii=True)\n"
        ),
        review_notes=("F1 config-propagation: entry (processor, scorer) pair -> middle "
                      "_get_processor wrapper composition -> consumer scoring pipeline. The bug swaps "
                      "the composition order at the single wrapper edge. Single line restored; "
                      "requirements unchanged."),
        final_response=("The processor composition order is restored; "
                        "the actual source-bound observation completed and the patch was submitted."),
    ),
    dict(
        iid="seatgeek__thefuzz.8a05a3ee.func_basic__1rnfhg0x",
        key="f1-new-thefuzz-ratio-forward",
        canonical_repo="seatgeek/thefuzz",
        primary_adapter="F", submechanism="F1",
        family="thefuzz.fuzz.partial_token_sort_ratio::forward_argument_order",
        prod_path="thefuzz/fuzz.py", prod_module="thefuzz.fuzz",
        profile=".venv-f1-thefuzz", license_file="LICENSE.txt",
        old="        _partial_token_sort_ratio, s2, s1, not force_ascii, full_process\n",
        new="        _partial_token_sort_ratio, s1, s2, force_ascii, full_process\n",
        read_start=62, read_end=72,
        mechanism={
            "trigger": "Scoring partial token-sort ratio with force_ascii control",
            "base_wrong_behavior": "s1/s2 are swapped and force_ascii inverted at the rapidfuzz forward edge, so asymmetric partial scoring and ASCII folding both break",
            "production_symbols": ["thefuzz.fuzz.partial_token_sort_ratio"],
            "causal_change": "Restore (s1, s2, force_ascii) order at the scorer call; ratio internals untouched",
            "discriminating_observation": {
                "kind": "executed_before_after_source_observation",
                "description": "partial_token_sort_ratio on accented vs plain strings must be 100 with force_ascii and below 100 without. Before repair the swapped/inverted edge breaks the first assert; after repair both pass.",
            },
            "bug_patch_direction": "introduces_defect",
        },
        observation=(
            "from thefuzz import fuzz\n"
            "s1 = 'ABCDÁ HELPÁ'\n"
            "s2 = 'ABCD HELP'\n"
            "a = fuzz.partial_token_sort_ratio(s1, s2, force_ascii=True)\n"
            "b = fuzz.partial_token_sort_ratio(s1, s2, force_ascii=False)\n"
            "print('scores', a, b)\n"
            "assert a == 100\n"
            "assert b < 100\n"
        ),
        review_notes=("F1 config-propagation: entry (s1, s2, force_ascii) triple -> middle public "
                      "ratio function -> consumer rapidfuzz scorer. The bug scrambles all three "
                      "channels at the single forward edge as one forwarding contract. Line restored; "
                      "requirements unchanged."),
        final_response=("The scorer forward channels are ordered again; "
                        "the actual source-bound observation completed and the patch was submitted."),
    ),
    dict(
        iid="seatgeek__thefuzz.8a05a3ee.func_basic__pbmk1lv5",
        key="f1-new-thefuzz-extract-forward",
        canonical_repo="seatgeek/thefuzz",
        primary_adapter="F", submechanism="F1",
        family="thefuzz.process.extract::extractBests_forwarding",
        prod_path="thefuzz/process.py", prod_module="thefuzz.process",
        profile=".venv-f1-thefuzz", license_file="LICENSE.txt",
        old="    return extractBests(choices, query, processor=scorer, scorer=processor, limit=limit + 1)\n",
        new="    return extractBests(query, choices, processor=processor, scorer=scorer, limit=limit)\n",
        read_start=185, read_end=195,
        mechanism={
            "trigger": "Extracting best fuzzy matches with a custom processor over event records",
            "base_wrong_behavior": "query/choices and processor/scorer are cross-wired plus limit bumped at the extractBests forward, so matching runs on scrambled channels",
            "production_symbols": ["thefuzz.process.extract"],
            "causal_change": "Restore all five channels to the extractBests call; ranking internals untouched",
            "discriminating_observation": {
                "kind": "executed_before_after_source_observation",
                "description": "extract with a field-selecting processor must rank the matching event first. Before repair the scrambled forward breaks ranking; after repair it passes.",
            },
            "bug_patch_direction": "introduces_defect",
        },
        observation=(
            "from thefuzz import process\n"
            "events = [['chicago cubs vs new york mets', 'CitiField'], ['new york yankees vs boston red sox', 'Fenway']]\n"
            "query = ['new york mets vs chicago cubs', 'CitiField']\n"
            "best = process.extract(query, events, processor=lambda event: event[0], limit=2)\n"
            "print('best', best[0])\n"
            "assert best[0][0] == events[0]\n"
        ),
        review_notes=("F1 config-propagation: entry (query, choices, processor, scorer, limit) tuple -> "
                      "middle extract -> consumer extractBests. The bug cross-wires the channels at the "
                      "single forward edge as one forwarding contract. Distinct production function "
                      "from _get_processor (composition) and the fuzz ratio forwarders. Line restored; "
                      "requirements unchanged."),
        final_response=("The extract forward channels are ordered again; "
                        "the actual source-bound observation completed and the patch was submitted."),
    ),
    # ---- batch 3: cookiecutter (2) ----
    # ---- batch 3: colorlog (2) ----
    # ---- batch 3: radon (1) ----
    dict(
        iid="rubik__radon.54b88e58.func_pm_ctrl_shuffle__r052edqc",
        key="f1-new-radon-config-defaults",
        canonical_repo="rubik/radon",
        primary_adapter="F", submechanism="F1",
        family="radon.cli.Config.from_function::defaults_assembly_order",
        prod_path="radon/cli/__init__.py", prod_module="radon.cli",
        profile=".venv-f1-radon",
        old=("        return cls(**values)\n"
             "        values.update(kwonlydefaults)\n"
             "        args, _, _, defaults = argspec[:4]\n"
             "        try:\n"
             "            argspec = inspect.getfullargspec(func)\n"
             "            kwonlydefaults = argspec.kwonlydefaults or {}\n"
             "        except AttributeError:  # pragma: no cover\n"
             "            argspec = inspect.getargspec(func)\n"
             "        '''Construct a Config object from a function's defaults.'''\n"
             "        values = dict(zip(reversed(args), reversed(defaults or [])))\n"
             "        kwonlydefaults = {}\n"),
        new=("        '''Construct a Config object from a function's defaults.'''\n"
             "        kwonlydefaults = {}\n"
             "        try:\n"
             "            argspec = inspect.getfullargspec(func)\n"
             "            kwonlydefaults = argspec.kwonlydefaults or {}\n"
             "        except AttributeError:  # pragma: no cover\n"
             "            argspec = inspect.getargspec(func)\n"
             "        args, _, _, defaults = argspec[:4]\n"
             "        values = dict(zip(reversed(args), reversed(defaults or [])))\n"
             "        values.update(kwonlydefaults)\n"
             "        return cls(**values)\n"),
        read_start=344, read_end=362,
        mechanism={
            "trigger": "Constructing a Config from a function's default arguments",
            "base_wrong_behavior": "The assembly is shuffled so the constructor returns before values exists (UnboundLocalError) and kwonly defaults never merge",
            "production_symbols": ["radon.cli.Config.from_function"],
            "causal_change": "Restore inspect-assemble-merge-return ordering; Config construction untouched",
            "discriminating_observation": {
                "kind": "executed_before_after_source_observation",
                "description": "Config.from_function on (a, b=2, c=[], d=None) must expose b/c/d defaults. Before repair the premature return raises UnboundLocalError; after repair all pass.",
            },
            "bug_patch_direction": "introduces_defect",
        },
        observation=(
            "from radon.cli import Config\n"
            "def example_func(a, b=2, c=[], d=None):\n"
            "    pass\n"
            "cfg = Config.from_function(example_func)\n"
            "print('config', cfg.b, cfg.c, cfg.d)\n"
            "assert (cfg.b, cfg.c, cfg.d) == (2, [], None)\n"
        ),
        review_notes=("F1 config-propagation: entry function defaults -> middle from_function "
                      "assembly -> consumer Config(**values). The shuffle severs the assembly edge "
                      "(premature return) as one ordering contract. Block restored; requirements "
                      "unchanged."),
        final_response=("The defaults assembly edge materializes again; "
                        "the actual source-bound observation completed and the patch was submitted."),
    ),
    # ---- batch 3: schedule (2) ----
    dict(
        iid="dbader__schedule.82a43db1.func_basic__hw6jthis",
        key="f1-new-schedule-repeat-forward",
        canonical_repo="dbader/schedule",
        primary_adapter="F", submechanism="F1",
        family="schedule.repeat._schedule_decorator::call_spec_forwarding",
        prod_path="schedule/__init__.py", prod_module="schedule",
        profile=".venv-f1-schedule", license_file="LICENSE.txt",
        old=("        job.do(decorated_function, *args, args)\n"
             "        return lambda *a, **k: decorated_function(*a, **k)\n"),
        new=("        job.do(decorated_function, *args, **kwargs)\n"
             "        return decorated_function\n"),
        read_start=903, read_end=915,
        mechanism={
            "trigger": "Scheduling a decorated function with positional and keyword call arguments",
            "base_wrong_behavior": "The kwargs channel is replaced by a duplicate args channel at the job.do edge and the decorator returns a lambda instead of the function",
            "production_symbols": ["schedule.repeat._schedule_decorator"],
            "causal_change": "Restore (*args, **kwargs) forwarding to job.do and return the decorated function; job execution untouched",
            "discriminating_observation": {
                "kind": "executed_before_after_source_observation",
                "description": "A repeat-decorated job must keep its __name__ and run once with the exact call-spec. Before repair both the identity and the recorded call break; after repair they pass.",
            },
            "bug_patch_direction": "introduces_defect",
        },
        observation=(
            "from unittest import mock\n"
            "import schedule\n"
            "from schedule import repeat, every\n"
            "mock_job = mock.Mock()\n"
            "@repeat(every().minute, 1, 2, 'three', foo=23, bar={})\n"
            "def job(*args, **kwargs):\n"
            "    mock_job(*args, **kwargs)\n"
            "assert job.__name__ == 'job', job.__name__\n"
            "schedule.run_all()\n"
            "mock_job.assert_called_once_with(1, 2, 'three', foo=23, bar={})\n"
            "print('repeat forwards')\n"
        ),
        review_notes=("F1 config-propagation: entry decorator (*args, **kwargs) call-spec -> middle "
                      "_schedule_decorator -> consumers job.do storage and the returned function. The "
                      "bug corrupts the kwargs channel and severs the function-identity edge as one "
                      "decorator-forwarding contract. Hunk restored; requirements unchanged."),
        final_response=("The decorator forwards the call-spec and function again; "
                        "the actual source-bound observation completed and the patch was submitted."),
    ),
    dict(
        iid="dbader__schedule.82a43db1.func_basic__eatzeuij",
        key="f1-new-schedule-tag-selection",
        canonical_repo="dbader/schedule",
        primary_adapter="F", submechanism="F1",
        family="schedule.Scheduler.get_next_run::tag_filtered_selection",
        prod_path="schedule/__init__.py", prod_module="schedule",
        profile=".venv-f1-schedule", license_file="LICENSE.txt",
        old=("        # Introduced a bug by skipping the filtered jobs step, which affects tag-based filtering.\n"
             "        if not self.jobs:\n"
             "            return None\n"
             "        return max(self.jobs).next_run\n"),
        new=("        jobs_filtered = self.get_jobs(tag)\n"
             "        if not jobs_filtered:\n"
             "            return None\n"
             "        return min(jobs_filtered).next_run\n"),
        read_start=183, read_end=196,
        mechanism={
            "trigger": "Querying the next run filtered by tag across jobs with different schedules",
            "base_wrong_behavior": "The tag-filter edge is skipped (unfiltered jobs used) and min is flipped to max, so tag selection returns the wrong job",
            "production_symbols": ["schedule.Scheduler.get_next_run"],
            "causal_change": "Restore tag-filtered selection with min; job ordering untouched",
            "discriminating_observation": {
                "kind": "executed_before_after_source_observation",
                "description": "get_next_run('tag1') over a 5s-tag1 job and a 2h-tag1/tag2 job must equal the 5s job's next_run. Before repair the unfiltered max returns the 2h job; after repair it passes.",
            },
            "bug_patch_direction": "introduces_defect",
        },
        observation=(
            "import schedule\n"
            "sch = schedule.Scheduler()\n"
            "def j1(): pass\n"
            "def j2(): pass\n"
            "jb1 = sch.every(5).seconds.do(j1).tag('tag1')\n"
            "jb2 = sch.every(2).hours.do(j2).tag('tag1', 'tag2')\n"
            "nxt = sch.get_next_run('tag1')\n"
            "print('next', nxt)\n"
            "assert nxt == jb1.next_run\n"
        ),
        review_notes=("F1 config-propagation: entry tag -> middle get_next_run filtering -> consumer "
                      "min selection. The bug drops the filter edge and flips the selector as one "
                      "tag-selection contract. Distinct production function from the repeat decorator. "
                      "Hunk restored; requirements unchanged."),
        final_response=("Tag-filtered selection is restored; "
                        "the actual source-bound observation completed and the patch was submitted."),
    ),
    # ---- batch 3: schema extra (3) ----
    dict(
        iid="keleshev__schema.24a30457.func_basic__vev5jdcq",
        key="f1-new-schema-const-passthrough",
        canonical_repo="keleshev/schema",
        primary_adapter="F", submechanism="F1",
        family="schema.Const.validate::data_passthrough",
        prod_path="schema/__init__.py", prod_module="schema",
        profile=".venv-f1-schema", license_file="LICENSE-MIT",
        old=("        super(Const, self).validate(kwargs)  # Changed data to kwargs\n"
             "        return None  # Changed return value from data to None\n"),
        new=("        super(Const, self).validate(data, **kwargs)\n"
             "        return data\n"),
        read_start=906, read_end=916,
        mechanism={
            "trigger": "Validating data through a Const schema wrapping a composite validator",
            "base_wrong_behavior": "The data edge is replaced by kwargs at the super-validation call and the return channel is nulled, so validated data never flows through",
            "production_symbols": ["schema.Const.validate"],
            "causal_change": "Restore data to the super call and to the return; composite validation untouched",
            "discriminating_observation": {
                "kind": "executed_before_after_source_observation",
                "description": "Schema(Const(...)).validate(data) must return data unchanged. Before repair the severed data edge raises/returns None; after repair it passes.",
            },
            "bug_patch_direction": "introduces_defect",
        },
        observation=(
            "from schema import Schema, Const, And, Use\n"
            "from functools import partial\n"
            "def unique_list(_list):\n"
            "    return len(_list) == len(set(_list))\n"
            "def dict_keys(key, _list):\n"
            "    return list(map(lambda d: d[key], _list))\n"
            "schema = Schema(Const(And(Use(partial(dict_keys, 'index')), unique_list)))\n"
            "data = [{'index': 1, 'value': 'foo'}, {'index': 2, 'value': 'bar'}]\n"
            "assert schema.validate(data) == data\n"
            "print('const passthrough')\n"
        ),
        review_notes=("F1 config-propagation: entry data -> middle Const.validate -> consumers super "
                      "validation and return channel. The bug severs the data edge at both edges as "
                      "one passthrough contract. Distinct production class from And/Literal helpers. "
                      "Hunk restored; requirements unchanged."),
        final_response=("Validated data flows through Const again; "
                        "the actual source-bound observation completed and the patch was submitted."),
    ),
    dict(
        iid="keleshev__schema.24a30457.func_basic__5hkx4ybr",
        key="f1-new-schema-forbidden-init",
        canonical_repo="keleshev/schema",
        primary_adapter="F", submechanism="F1",
        family="schema.Forbidden.__init__::handler_installation",
        prod_path="schema/__init__.py", prod_module="schema",
        profile=".venv-f1-schema", license_file="LICENSE-MIT",
        old=("        kwargs[\"handler\"] = self._alternate_function\n"
             "        super(Forbidden, self).__init__(args, kwargs)\n"),
        new=("        kwargs[\"handler\"] = self._default_function\n"
             "        super(Forbidden, self).__init__(*args, **kwargs)\n"),
        read_start=876, read_end=886,
        mechanism={
            "trigger": "Constructing a Forbidden key validator and validating against it",
            "base_wrong_behavior": "The default handler edge points at a missing attribute and the arg channels are packed instead of forwarded, so construction raises AttributeError",
            "production_symbols": ["schema.Forbidden.__init__"],
            "causal_change": "Restore the default handler and *args/**kwargs forwarding to Hook; hook dispatch untouched",
            "discriminating_observation": {
                "kind": "executed_before_after_source_observation",
                "description": "Forbidden('b') schemas must raise SchemaForbiddenKeyError on banned keys and pass sibling Optional keys. Before repair construction raises AttributeError; after repair both pass.",
            },
            "bug_patch_direction": "introduces_defect",
        },
        observation=(
            "from schema import Schema, Forbidden, Optional, SchemaForbiddenKeyError\n"
            "try:\n"
            "    Schema({Forbidden('b'): object}).validate({'b': 'bye'})\n"
            "    raise SystemExit('no error')\n"
            "except SchemaForbiddenKeyError:\n"
            "    pass\n"
            "assert Schema({Forbidden('b'): int, Optional('b'): object}).validate({'b': 'bye'}) == {'b': 'bye'}\n"
            "print('forbidden installed')\n"
        ),
        review_notes=("F1 config-propagation: entry (*args, **kwargs) plus default handler -> middle "
                      "Forbidden.__init__ -> consumer Hook storage. The bug corrupts both the handler "
                      "edge and the channel packing as one constructor-forwarding contract. Distinct "
                      "production class from Const/And. Hunk restored; requirements unchanged."),
        final_response=("The forbidden handler installs and forwards again; "
                        "the actual source-bound observation completed and the patch was submitted."),
    ),
    dict(
        iid="keleshev__schema.24a30457.func_basic__t668k5ky",
        key="f1-new-schema-key-priority",
        canonical_repo="keleshev/schema",
        primary_adapter="F", submechanism="F1",
        family="schema.Schema._dict_key_priority::hook_optional_precedence",
        prod_path="schema/__init__.py", prod_module="schema",
        profile=".venv-f1-schema", license_file="LICENSE-MIT",
        old=("    def _dict_key_priority(s) -> float:\n"
             "        \"\"\"Return priority for a given key object.\"\"\"\n"
             "        if isinstance(s, Hook):\n"
             "            return _priority(s._schema) + 0.5\n"
             "        if isinstance(s, Optional):\n"
             "            return _priority(s._schema) - 0.5\n"
             "        return _priority(s) + 1\n"),
        new=("    def _dict_key_priority(s) -> float:\n"
             "        \"\"\"Return priority for a given key object.\"\"\"\n"
             "        if isinstance(s, Hook):\n"
             "            return _priority(s._schema) - 0.5\n"
             "        if isinstance(s, Optional):\n"
             "            return _priority(s._schema) + 0.5\n"
             "        return _priority(s)\n"),
        read_start=393, read_end=405,
        mechanism={
            "trigger": "Validating a dict where a Hook key and an Optional key compete for ordering",
            "base_wrong_behavior": "Hook/Optional precedence is inverted and the base shifted, so validation visits keys in the wrong order and hook dispatch breaks",
            "production_symbols": ["schema.Schema._dict_key_priority"],
            "causal_change": "Restore Hook-early/Optional-late precedence and the base priority; key validation untouched",
            "discriminating_observation": {
                "kind": "executed_before_after_source_observation",
                "description": "A Hook+Optional schema must validate and fire the hook exactly once. Before repair the inverted precedence breaks dispatch; after repair it passes.",
            },
            "bug_patch_direction": "introduces_defect",
        },
        observation=(
            "from unittest.mock import Mock\n"
            "from schema import Schema, Optional, Hook\n"
            "m = Mock(return_value=None)\n"
            "hook = Hook('b', handler=m)\n"
            "assert Schema({hook: str, Optional('b'): object}).validate({'b': 'bye'}) == {'b': 'bye'}\n"
            "assert m.call_count == 1\n"
            "print('precedence ok')\n"
        ),
        review_notes=("F1 precedence: entry key objects -> middle _dict_key_priority ordering -> "
                      "consumer dict validation sequence. The bug inverts Hook/Optional precedence as "
                      "one ordering contract. Distinct production function from the validators. Hunk "
                      "restored; requirements unchanged."),
        final_response=("Key precedence is ordered Hook-before-Optional again; "
                        "the actual source-bound observation completed and the patch was submitted."),
    ),
    # ---- batch 3: oauthlib extra (1) ----
    dict(
        iid="oauthlib__oauthlib.1fd52536.func_basic__52w4ldd8",
        key="f1-new-oauthlib-client-fields",
        canonical_repo="oauthlib/oauthlib",
        primary_adapter="F", submechanism="F1",
        family="oauthlib.oauth2.clients.base.Client.__init__::constructor_field_identity",
        prod_path="oauthlib/oauth2/rfc6749/clients/base.py",
        prod_module="oauthlib.oauth2.rfc6749.clients.base",
        profile=".venv-f1-oauthlib",
        old=("        self.access_token = refresh_token  # Bug: Assigned refresh_token instead of access_token\n"
             "        self.refresh_token = access_token  # Bug: Assigned access_token instead of refresh_token\n"
             "        self.mac_key = mac_key\n"
             "        self.mac_algorithm = mac_algorithm\n"
             "        self.token = token if isinstance(token, list) else {}  # Bug: Changed token to list check\n"
             "        self.scope = scope\n"
             "        self.state_generator = state_generator\n"
             "        self.state = self.token.get('state', state)  # Bug: Incorrectly prioritize state from token dict\n"
             "        self.redirect_url = redirect_url\n"
             "        self.code_verifier = code_challenge  # Bug: Assigned code_challenge instead of code_verifier\n"
             "        self.code_challenge = code_verifier  # Bug: Assigned code_verifier instead of code_challenge\n"),
        new=("        self.access_token = access_token\n"
             "        self.refresh_token = refresh_token\n"
             "        self.mac_key = mac_key\n"
             "        self.mac_algorithm = mac_algorithm\n"
             "        self.token = token or {}\n"
             "        self.scope = scope\n"
             "        self.state_generator = state_generator\n"
             "        self.state = state\n"
             "        self.redirect_url = redirect_url\n"
             "        self.code_verifier = code_verifier\n"
             "        self.code_challenge = code_challenge\n"),
        read_start=119, read_end=136,
        mechanism={
            "trigger": "Constructing an OAuth2 Client with access/refresh tokens, PKCE pair, state and token mapping",
            "base_wrong_behavior": "Each constructor argument must land in its own field; the bug cross-wires token pairs, replaces token storage with a list check, and sources state from the token dict",
            "production_symbols": ["oauthlib.oauth2.rfc6749.clients.base.Client.__init__"],
            "causal_change": "Restore field-identity storage for every constructor argument; token placement consumers untouched",
            "discriminating_observation": {
                "kind": "executed_before_after_source_observation",
                "description": "Client(access_token='A', refresh_token='R', code_verifier='V', code_challenge='C', state='S', token={...}) must expose each value in its own field. Before repair the cross-wired fields fail the asserts; after repair all pass.",
            },
            "bug_patch_direction": "introduces_defect",
        },
        observation=(
            "from oauthlib.oauth2 import Client\n"
            "c = Client('cid', access_token='A', refresh_token='R', code_verifier='V', code_challenge='C', state='S', token={'k': 'v'})\n"
            "print('fields', c.access_token, c.refresh_token, c.code_verifier, c.code_challenge, c.state, c.token)\n"
            "assert (c.access_token, c.refresh_token) == ('A', 'R')\n"
            "assert (c.code_verifier, c.code_challenge) == ('V', 'C')\n"
            "assert c.state == 'S' and c.token == {'k': 'v'}\n"
        ),
        review_notes=("F1 config-propagation: entry constructor argument tuple -> middle Client.__init__ "
                      "field storage -> consumers token placement and PKCE flows. The bug violates "
                      "field identity across four bindings as one storage contract. Distinct production "
                      "function from the per-client subclass constructors. Block restored; requirements "
                      "unchanged."),
        final_response=("Each constructor argument lands in its own field again; "
                        "the actual source-bound observation completed and the patch was submitted."),
    ),
    # ---- batch 3: langdetect (1) ----
    dict(
        iid="Mimino666__langdetect.a1598f1a.func_basic__311rpkbk",
        key="f1-new-langdetect-storage",
        canonical_repo="Mimino666/langdetect",
        primary_adapter="F", submechanism="F1",
        family="langdetect.DetectorFactory.clear::storage_shape",
        prod_path="langdetect/detector_factory.py", prod_module="langdetect.detector_factory",
        profile=".venv-f1-langdetect",
        old=("        self.langlist = {}\n"
             "        self.word_lang_prob_map = []\n"),
        new=("        self.langlist = []\n"
             "        self.word_lang_prob_map = {}\n"),
        read_start=88, read_end=100,
        mechanism={
            "trigger": "Resetting a DetectorFactory with loaded state via clear()",
            "base_wrong_behavior": "The list/dict storage shapes are swapped on reset, so subsequent profile loading appends into a dict and maps into a list",
            "production_symbols": ["langdetect.detector_factory.DetectorFactory.clear"],
            "causal_change": "Restore list reset for langlist and dict reset for the probability map; loading logic untouched",
            "discriminating_observation": {
                "kind": "executed_before_after_source_observation",
                "description": "After seeding state, clear() must restore langlist==[] and word_lang_prob_map=={}. Before repair the swapped shapes fail both asserts; after repair they pass.",
            },
            "bug_patch_direction": "introduces_defect",
        },
        observation=(
            "from langdetect.detector_factory import DetectorFactory\n"
            "f = DetectorFactory()\n"
            "f.langlist.append('en')\n"
            "f.word_lang_prob_map['x'] = 1\n"
            "f.clear()\n"
            "print('storage', f.langlist, f.word_lang_prob_map)\n"
            "assert f.langlist == [] and f.word_lang_prob_map == {}\n"
        ),
        review_notes=("F1 config-propagation: entry loaded profile state -> middle "
                      "DetectorFactory.clear reset -> consumers profile loading and detection. "
                      "The bug cross-wires the two storage shapes on reset as one storage contract. "
                      "Hunk restored; requirements unchanged."),
        final_response=("Profile storage shapes reset correctly again; "
                        "the actual source-bound observation completed and the patch was submitted."),
    ),
    # ---- batch 3: arrow extra (1) ----
    dict(
        iid="arrow-py__arrow.1d70d009.func_basic__jfpi7hld",
        key="f1-new-arrow-factory-now",
        canonical_repo="arrow-py/arrow",
        primary_adapter="F", submechanism="F1",
        family="arrow.ArrowFactory.now::local_default",
        prod_path="arrow/factory.py", prod_module="arrow.factory",
        profile=".venv-f1-arrow",
        old=("        if tz is None:\n"
             "            tz = dateutil_tz.tzutc()  # Changed from tzlocal to tzutc\n"
             "        elif not isinstance(tz, dt_tzinfo):\n"
             "            tz = parser.TzinfoParser.parse(tz)\n"
             "\n"
             "        return self.type.utcnow()  # Changed now to utcnow\n"),
        new=("        if tz is None:\n"
             "            tz = dateutil_tz.tzlocal()\n"
             "        elif not isinstance(tz, dt_tzinfo):\n"
             "            tz = parser.TzinfoParser.parse(tz)\n"
             "\n"
             "        return self.type.now(tz)\n"),
        read_start=335, read_end=347,
        mechanism={
            "trigger": "Calling ArrowFactory.now() without a timezone for local time",
            "base_wrong_behavior": "The local-tz default is replaced by UTC and construction dispatches to utcnow, so the local-time edge never reaches the consumer",
            "production_symbols": ["arrow.factory.ArrowFactory.now"],
            "causal_change": "Restore tzlocal default and now(tz) dispatch; tz parsing untouched",
            "discriminating_observation": {
                "kind": "executed_before_after_source_observation",
                "description": "ArrowFactory().now().tzinfo must be exactly tzlocal. Before repair it is tzutc; after repair it passes.",
            },
            "bug_patch_direction": "introduces_defect",
        },
        observation=(
            "from arrow.factory import ArrowFactory\n"
            "from dateutil import tz\n"
            "a = ArrowFactory().now()\n"
            "print('now tz', a.tzinfo)\n"
            "assert type(a.tzinfo) is tz.tzlocal\n"
        ),
        review_notes=("F1 config-propagation: entry absent tz -> middle ArrowFactory.now defaulting -> "
                      "consumer Arrow construction. The bug corrupts both the default source and the "
                      "dispatch target as one local-now contract. Distinct production function from "
                      "the api.py forwarders. Hunk restored; requirements unchanged."),
        final_response=("Local-now defaulting and dispatch are restored; "
                        "the actual source-bound observation completed and the patch was submitted."),
    ),
    # ---- batch 3: icecream (1) ----
    dict(
        iid="gruns__icecream.f76fef56.func_pm_ctrl_invert_if__7iujb6a9",
        key="f1-new-icecream-arity-passthrough",
        canonical_repo="gruns/icecream",
        primary_adapter="F", submechanism="F1",
        family="icecream.IceCreamDebugger.__call__::arity_passthrough",
        prod_path="icecream/icecream.py", prod_module="icecream.icecream",
        profile=".venv-f1-icecream", license_file="LICENSE.txt",
        old=("        elif len(args) == 1:  # E.g. ic(1, 2, 3).\n"
             "            passthrough = args\n"
             "        else:  # E.g. ic(1).\n"
             "            passthrough = args[0]\n"
             "\n"
             "        return passthrough\n"
             "    def format(self, *args):\n"),
        new=("        elif len(args) == 1:  # E.g. ic(1).\n"
             "            passthrough = args[0]\n"
             "        else:  # E.g. ic(1, 2, 3).\n"
             "            passthrough = args\n"
             "\n"
             "        return passthrough\n"
             "\n"
             "    def format(self, *args):\n"),
        read_start=198, read_end=214,
        mechanism={
            "trigger": "Calling ic() with a single value versus multiple values",
            "base_wrong_behavior": "The single/multi arity passthrough values are cross-wired, so ic(5) returns a tuple and ic(1, 2) returns a scalar",
            "production_symbols": ["icecream.icecream.IceCreamDebugger.__call__"],
            "causal_change": "Restore scalar passthrough for arity one and tuple for arity many; output formatting untouched",
            "discriminating_observation": {
                "kind": "executed_before_after_source_observation",
                "description": "ic(5) must return 5 and ic(1, 2) must return (1, 2) with output suppressed. Before repair the cross-wired arity branches fail both asserts; after repair they pass.",
            },
            "bug_patch_direction": "introduces_defect",
        },
        observation=(
            "from icecream import ic\n"
            "ic.configureOutput(includeContext=False, outputFunction=lambda s: None)\n"
            "assert ic(5) == 5\n"
            "assert ic(1, 2) == (1, 2)\n"
            "print('arity passthrough')\n"
        ),
        review_notes=("F1 config-propagation: entry call arity -> middle IceCreamDebugger.__call__ "
                      "passthrough selection -> consumer return value. The bug cross-wires the two "
                      "arity branches as one dispatch contract. Hunk restored; requirements unchanged."),
        final_response=("Arity passthrough returns scalars and tuples correctly again; "
                        "the actual source-bound observation completed and the patch was submitted."),
    ),
    # ---- batch 3: schedule extra (1) ----
    dict(
        iid="dbader__schedule.82a43db1.func_basic__avciypb3",
        key="f1-new-schedule-run-pending",
        canonical_repo="dbader/schedule",
        primary_adapter="F", submechanism="F1",
        family="schedule.run_pending::scheduler_dispatch",
        prod_path="schedule/__init__.py", prod_module="schedule",
        profile=".venv-f1-schedule", license_file="LICENSE.txt",
        old="    default_scheduler.run()\n",
        new="    default_scheduler.run_pending()\n",
        read_start=848, read_end=856,
        mechanism={
            "trigger": "Running due jobs through the module-level run_pending entry",
            "base_wrong_behavior": "The call is dispatched to a nonexistent run method instead of the scheduler's run_pending, so due-job execution never happens",
            "production_symbols": ["schedule.run_pending"],
            "causal_change": "Restore dispatch to default_scheduler.run_pending; scheduler internals untouched",
            "discriminating_observation": {
                "kind": "executed_before_after_source_observation",
                "description": "With a mocked default scheduler, schedule.run_pending() must call run_pending exactly once. Before repair the misdispatched call breaks the assert; after repair it passes.",
            },
            "bug_patch_direction": "introduces_defect",
        },
        observation=(
            "from unittest import mock\n"
            "import schedule\n"
            "with mock.patch.object(schedule, 'default_scheduler') as ds:\n"
            "    schedule.run_pending()\n"
            "    ds.run_pending.assert_called_once_with()\n"
            "print('dispatch ok')\n"
        ),
        review_notes=("F1 config-propagation: entry run_pending() call -> middle module-level "
                      "run_pending -> consumer Scheduler.run_pending. The bug misroutes the dispatch "
                      "edge. Distinct production function from the decorator and selector. Single "
                      "line restored; requirements unchanged."),
        final_response=("run_pending dispatches to the scheduler again; "
                        "the actual source-bound observation completed and the patch was submitted."),
    ),
    # ---- batch 3: schema extra (1) ----
    dict(
        iid="keleshev__schema.24a30457.func_basic__lgfgdj4p",
        key="f1-new-schema-hook-key",
        canonical_repo="keleshev/schema",
        primary_adapter="F", submechanism="F1",
        family="schema.Hook.__init__::key_storage",
        prod_path="schema/__init__.py", prod_module="schema",
        profile=".venv-f1-schema", license_file="LICENSE-MIT",
        old="        self.key = self._schema.copy()\n",
        new="        self.key = self._schema\n",
        read_start=866, read_end=878,
        mechanism={
            "trigger": "Constructing a Hook key validator and validating a banned key through it",
            "base_wrong_behavior": "The key channel is stored as a .copy() of the schema instead of the schema itself, so construction itself raises AttributeError for plain schema values",
            "production_symbols": ["schema.Hook.__init__"],
            "causal_change": "Restore direct key storage; hook dispatch untouched",
            "discriminating_observation": {
                "kind": "executed_before_after_source_observation",
                "description": "Forbidden('b') validation must raise SchemaForbiddenKeyError on banned keys. Before repair construction raises AttributeError on .copy(); after repair it passes.",
            },
            "bug_patch_direction": "introduces_defect",
        },
        observation=(
            "from schema import Schema, Forbidden, SchemaForbiddenKeyError\n"
            "try:\n"
            "    Schema({Forbidden('b'): object}).validate({'b': 'bye'})\n"
            "    raise SystemExit('no error')\n"
            "except SchemaForbiddenKeyError:\n"
            "    pass\n"
            "print('hook key ok')\n"
        ),
        review_notes=("F1 config-propagation: entry key schema -> middle Hook.__init__ storage -> "
                      "consumer key matching. The bug corrupts the storage channel with .copy(). "
                      "Distinct production class from Forbidden/Const/And. Single line restored; "
                      "requirements unchanged."),
        final_response=("The hook key stores the schema itself again; "
                        "the actual source-bound observation completed and the patch was submitted."),
    ),
    # ---- batch 3: oauthlib extra (1) ----
    dict(
        iid="oauthlib__oauthlib.1fd52536.func_basic__jxvhnz3r",
        key="f1-new-oauthlib-bearer-uri",
        canonical_repo="oauthlib/oauthlib",
        primary_adapter="F", submechanism="F1",
        family="oauthlib.oauth2.tokens.prepare_bearer_uri::token_value_forwarding",
        prod_path="oauthlib/oauth2/rfc6749/tokens.py",
        prod_module="oauthlib.oauth2.rfc6749.tokens",
        profile=".venv-f1-oauthlib",
        old="    return add_params_to_uri(uri, [('access_token', '')])\n",
        new="    return add_params_to_uri(uri, [(('access_token', token))])\n",
        read_start=171, read_end=183,
        mechanism={
            "trigger": "Placing a bearer token into a request URI",
            "base_wrong_behavior": "The token value is replaced by an empty string at the URI-building call edge, so the credential never reaches the URI",
            "production_symbols": ["oauthlib.oauth2.rfc6749.tokens.prepare_bearer_uri"],
            "causal_change": "Restore the token value in the access_token parameter; URI building untouched",
            "discriminating_observation": {
                "kind": "executed_before_after_source_observation",
                "description": "prepare_bearer_uri('tok123', uri) must embed access_token=tok123. Before repair the emptied value breaks the assert; after repair it passes.",
            },
            "bug_patch_direction": "introduces_defect",
        },
        observation=(
            "from oauthlib.oauth2.rfc6749.tokens import prepare_bearer_uri\n"
            "out = prepare_bearer_uri('tok123', 'https://example.com/res')\n"
            "print('uri', out)\n"
            "assert 'access_token=tok123' in out\n"
        ),
        review_notes=("F1 config-propagation: entry (token, uri) pair -> middle prepare_bearer_uri -> "
                      "consumer add_params_to_uri. The bug drops the token value at the call edge. "
                      "Distinct production function from the validator and client constructors. "
                      "Single line restored; requirements unchanged."),
        final_response=("The token value reaches the URI again; "
                        "the actual source-bound observation completed and the patch was submitted."),
    ),
    # ---- batch 3: radon extra (1) ----
    dict(
        iid="rubik__radon.54b88e58.func_pm_ctrl_shuffle__md7nx925",
        key="f1-new-radon-codeclimate-edge",
        canonical_repo="rubik/radon",
        primary_adapter="F", submechanism="F1",
        family="radon.cli.tools.dict_to_codeclimate_issues::early_return_edge",
        prod_path="radon/cli/tools.py", prod_module="radon.cli.tools",
        profile=".venv-f1-radon",
        edits=[
            ("def dict_to_codeclimate_issues(results, threshold='B'):\n"
             "    codeclimate_issues = []\n"
             "    '''Convert a dictionary holding CC analysis results into Code Climate\n"
             "     issue json.'''\n"
             "    return codeclimate_issues\n",
             "def dict_to_codeclimate_issues(results, threshold='B'):\n"
             "    '''Convert a dictionary holding CC analysis results into Code Climate\n"
             "     issue json.'''\n"
             "    codeclimate_issues = []\n"
             "    content = get_content()\n"
             "    error_content = 'We encountered an error attempting to analyze this line.'\n"),
            ("                    )\n"
             "    error_content = 'We encountered an error attempting to analyze this line.'\n"
             "    content = get_content()\n"
             "\n"
             "def cc_to_terminal(results, show_complexity, min, max, total_average):\n",
             "                    )\n"
             "    return codeclimate_issues\n"
             "\n"
             "\n"
             "def cc_to_terminal(results, show_complexity, min, max, total_average):\n"),
        ],
        read_start=376, read_end=390,
        mechanism={
            "trigger": "Converting cyclomatic complexity results into Code Climate issues",
            "base_wrong_behavior": "A premature return emits the empty list before the per-result computation, so the issue-building edge never runs",
            "production_symbols": ["radon.cli.tools.dict_to_codeclimate_issues"],
            "causal_change": "Restore computation-before-return ordering; issue formatting untouched",
            "discriminating_observation": {
                "kind": "executed_before_after_source_observation",
                "description": "Converting a two-block CC result must yield two Cyclomatic-complexity issues. Before repair the premature return yields []; after repair both pass.",
            },
            "bug_patch_direction": "introduces_defect",
        },
        observation=(
            "from radon.cli.tools import dict_to_codeclimate_issues\n"
            "case = [{'closures': [], 'endline': 16, 'complexity': 6, 'lineno': 12, 'type': 'function', 'name': 'foo', 'col_offset': 0, 'rank': 'B'}, {'complexity': 8, 'endline': 29, 'rank': 'B', 'lineno': 17, 'type': 'class', 'name': 'Classname', 'col_offset': 0}]\n"
            "out = dict_to_codeclimate_issues({'filename': case})\n"
            "print('issues', len(out))\n"
            "assert len(out) == 2 and 'Cyclomatic complexity' in out[0]\n"
        ),
        review_notes=("F1 config-propagation: entry analysis results -> middle "
                      "dict_to_codeclimate_issues loop -> consumer issues list. The shuffle inserts a "
                      "premature return that severs the computation edge as one ordering contract. "
                      "Distinct production function from Config.from_function. Hunk restored; "
                      "requirements unchanged."),
        final_response=("The issue-building edge runs to completion again; "
                        "the actual source-bound observation completed and the patch was submitted."),
    ),
    # ---- batch 3: channels (1) ----
    # ---- batch 4: pyopenssl (1) ----
    # ---- batch 4: grafanalib (1) ----
    dict(
        iid="weaveworks__grafanalib.5c3b17ed.func_basic__0r7hck5b",
        key="f1-new-grafanalib-print-stream",
        canonical_repo="weaveworks/grafanalib",
        primary_adapter="F", submechanism="F1",
        family="grafanalib._gen.print_dashboard::output_stream",
        prod_path="grafanalib/_gen.py", prod_module="grafanalib._gen",
        profile=".venv-f1-grafanalib",
        old="    write_dashboard(dashboard, stream=sys.stderr)\n",
        new="    write_dashboard(dashboard, stream=sys.stdout)\n",
        read_start=163, read_end=172,
        mechanism={
            "trigger": "Printing a dashboard to standard output for piping",
            "base_wrong_behavior": "The output stream edge is rerouted to stderr, so stdout consumers and pipes receive nothing",
            "production_symbols": ["grafanalib._gen.print_dashboard"],
            "causal_change": "Restore sys.stdout as the write_dashboard stream; rendering untouched",
            "discriminating_observation": {
                "kind": "executed_before_after_source_observation",
                "description": "print_dashboard under redirected stdout must emit non-empty JSON. Before repair stdout stays empty; after repair it passes.",
            },
            "bug_patch_direction": "introduces_defect",
        },
        observation=(
            "import io\n"
            "from contextlib import redirect_stdout\n"
            "from grafanalib.core import Dashboard\n"
            "from grafanalib._gen import print_dashboard\n"
            "buf = io.StringIO()\n"
            "with redirect_stdout(buf):\n"
            "    print_dashboard(Dashboard(title='T', rows=[]))\n"
            "print('bytes', len(buf.getvalue()))\n"
            "assert buf.getvalue() != ''\n"
        ),
        review_notes=("F1 config-propagation: entry dashboard -> middle print_dashboard -> consumer "
                      "write_dashboard stream sink. The bug misroutes the sink edge. Single line "
                      "restored; requirements unchanged."),
        final_response=("Dashboards print to stdout again; "
                        "the actual source-bound observation completed and the patch was submitted."),
    ),
    # ---- batch 4: soupsieve extra (2) ----
    dict(
        iid="facelessuser__soupsieve.a8080d97.func_basic__bvfjkr3d",
        key="f1-new-soupsieve-iselect-forward",
        canonical_repo="facelessuser/soupsieve",
        primary_adapter="F", submechanism="F1",
        family="soupsieve.iselect::compile_option_forwarding",
        prod_path="soupsieve/__init__.py", prod_module="soupsieve",
        profile=".venv-f1-soup39", task_python="/usr/bin/python3", task_python39=True,
        license_file="LICENSE.md",
        old="    yield from compile(select, flags, namespaces, **kwargs).iselect(tag, limit + 1)\n",
        new="    yield from compile(select, namespaces, flags, **kwargs).iselect(tag, limit)\n",
        read_start=155, read_end=165,
        mechanism={
            "trigger": "Iterating matches with namespaces, flags and a result limit",
            "base_wrong_behavior": "namespaces/flags are cross-wired at the compile edge and the limit is bumped, so selection runs on scrambled options",
            "production_symbols": ["soupsieve.iselect"],
            "causal_change": "Restore (select, namespaces, flags) order and the verbatim limit; matching untouched",
            "discriminating_observation": {
                "kind": "executed_before_after_source_observation",
                "description": "iselect('p.a', soup, limit=2) over three matches must yield exactly two. Before repair the scrambled forward breaks selection/count; after repair it passes.",
            },
            "bug_patch_direction": "introduces_defect",
        },
        observation=(
            "import soupsieve as sv\n"
            "from bs4 import BeautifulSoup\n"
            "soup = BeautifulSoup('<div><p class=\"a\">x</p><p class=\"a\">y</p><p class=\"a\">z</p></div>', 'html.parser')\n"
            "out = list(sv.iselect('p.a', soup, limit=2))\n"
            "print('selected', len(out))\n"
            "assert len(out) == 2\n"
        ),
        review_notes=("F1 config-propagation: entry (select, namespaces, flags, limit) tuple -> middle "
                      "iselect -> consumer compile plus iselect. The bug scrambles the option channels "
                      "at the single forward edge as one forwarding contract. Distinct production "
                      "function from SoupSieve.match (method) and filter. Line restored; requirements "
                      "unchanged. Same pinned 3.9 interpreter as the sibling task."),
        final_response=("The iselect option channels are ordered again; "
                        "the actual source-bound observation completed and the patch was submitted."),
    ),
    dict(
        iid="facelessuser__soupsieve.a8080d97.func_basic__s6wv59yh",
        key="f1-new-soupsieve-filter-forward",
        canonical_repo="facelessuser/soupsieve",
        primary_adapter="F", submechanism="F1",
        family="soupsieve.filter::select_iterable_forwarding",
        prod_path="soupsieve/__init__.py", prod_module="soupsieve",
        profile=".venv-f1-soup39", task_python="/usr/bin/python3", task_python39=True,
        license_file="LICENSE.md",
        old=("    if custom is not None:\n"
             "        flags |= len(custom)\n"
             "    \n"
             "    return compile(iterable, namespaces, flags, **kwargs).filter(select)\n"),
        new="    return compile(select, namespaces, flags, **kwargs).filter(iterable)\n",
        read_start=107, read_end=122,
        mechanism={
            "trigger": "Filtering a tag list by selector",
            "base_wrong_behavior": "select and iterable are cross-wired at both the compile and filter edges (plus flags pollution), so filtering runs on swapped inputs",
            "production_symbols": ["soupsieve.filter"],
            "causal_change": "Restore select to compile and iterable to filter, dropping the flags pollution; matching untouched",
            "discriminating_observation": {
                "kind": "executed_before_after_source_observation",
                "description": "filter('p.a', tags) must return the single p tag. Before repair the swapped inputs break filtering; after repair it passes.",
            },
            "bug_patch_direction": "introduces_defect",
        },
        observation=(
            "import soupsieve as sv\n"
            "from bs4 import BeautifulSoup\n"
            "soup = BeautifulSoup('<div><p class=\"a\">x</p><span>y</span></div>', 'html.parser')\n"
            "out = list(sv.filter('p.a', soup.find_all(True)))\n"
            "print('filtered', [(t.name) for t in out])\n"
            "assert len(out) == 1 and out[0].name == 'p'\n"
        ),
        review_notes=("F1 config-propagation: entry (select, iterable) pair -> middle filter -> "
                      "consumers compile and filter calls. The bug cross-wires both edges as one "
                      "forwarding contract. Distinct production function from iselect (generator) and "
                      "match (method). Hunk restored; requirements unchanged. Same pinned 3.9 "
                      "interpreter as the sibling tasks."),
        final_response=("The filter input channels are ordered again; "
                        "the actual source-bound observation completed and the patch was submitted."),
    ),
    # ---- batch 4: oauthlib extra (1) ----
    dict(
        iid="oauthlib__oauthlib.1fd52536.func_basic__lpglyr7k",
        key="f1-new-oauthlib-refresh-body",
        canonical_repo="oauthlib/oauthlib",
        primary_adapter="F", submechanism="F1",
        family="oauthlib.oauth2.clients.base.Client.prepare_refresh_body::body_assembly",
        prod_path="oauthlib/oauth2/rfc6749/clients/base.py",
        prod_module="oauthlib.oauth2.rfc6749.clients.base",
        profile=".venv-f1-oauthlib",
        old=("        refresh_token = refresh_token if refresh_token is not None else self.scope\n"
             "        scope = scope or self.refresh_token\n"
             "        return prepare_token_request(self.refresh_token_key, body=scope, scope=body,\n"),
        new=("        refresh_token = refresh_token or self.refresh_token\n"
             "        scope = self.scope if scope is None else scope\n"
             "        return prepare_token_request(self.refresh_token_key, body=body, scope=scope,\n"),
        read_start=440, read_end=452,
        mechanism={
            "trigger": "Preparing a refresh-token request body with explicit body, scope and stored token",
            "base_wrong_behavior": "Fallback sources are cross-wired and body/scope swapped at the token-request edge, so the grant carries scrambled parameters",
            "production_symbols": ["oauthlib.oauth2.rfc6749.clients.base.Client.prepare_refresh_body"],
            "causal_change": "Restore or-fallbacks and (body, scope) order at the token-request call; grant encoding untouched",
            "discriminating_observation": {
                "kind": "executed_before_after_source_observation",
                "description": "prepare_refresh_body(body='b', scope=['s2']) on a client holding RT must embed refresh_token=RT and scope=s2. Before repair the scrambled assembly breaks both; after repair they pass.",
            },
            "bug_patch_direction": "introduces_defect",
        },
        observation=(
            "from oauthlib.oauth2 import WebApplicationClient\n"
            "c = WebApplicationClient('cid', refresh_token='RT', scope=['s'])\n"
            "body = c.prepare_refresh_body(body='b', scope=['s2'])\n"
            "print('body', body)\n"
            "assert 'refresh_token=RT' in body and 'scope=s2' in body\n"
        ),
        review_notes=("F1 config-propagation: entry (body, scope, stored token) triple -> middle "
                      "prepare_refresh_body assembly -> consumer prepare_token_request. The bug "
                      "cross-wires fallbacks and swaps body/scope as one assembly contract. Distinct "
                      "production function from the constructors and validators. Hunk restored; "
                      "requirements unchanged."),
        final_response=("The refresh-body assembly is ordered again; "
                        "the actual source-bound observation completed and the patch was submitted."),
    ),
    # ---- batch 4: arrow extra (1) ----
    dict(
        iid="arrow-py__arrow.1d70d009.func_basic__6hsm14v4",
        key="f1-new-arrow-fromordinal",
        canonical_repo="arrow-py/arrow",
        primary_adapter="F", submechanism="F1",
        family="arrow.Arrow.fromordinal::ordinal_tz_mapping",
        prod_path="arrow/arrow.py", prod_module="arrow.arrow",
        profile=".venv-f1-arrow",
        edits=[
            ("        dt = dt_datetime.fromordinal(ordinal + 1)\n",
             "        dt = dt_datetime.fromordinal(ordinal)\n"),
            ("            None,  # Change tzinfo to None\n",
             "            dt.tzinfo,\n"),
        ],
        read_start=396, read_end=412,
        mechanism={
            "trigger": "Constructing an Arrow from a proleptic ordinal date",
            "base_wrong_behavior": "The ordinal is shifted by one and the tzinfo channel is nulled at the construction edges, so the mapped date is wrong",
            "production_symbols": ["arrow.arrow.Arrow.fromordinal"],
            "causal_change": "Restore verbatim ordinal mapping and tzinfo forwarding; field extraction untouched",
            "discriminating_observation": {
                "kind": "executed_before_after_source_observation",
                "description": "Arrow.fromordinal(date(2020,1,2).toordinal()) must expose 2020-01-02. Before repair the shifted mapping breaks the date assert; after repair it passes.",
            },
            "bug_patch_direction": "introduces_defect",
        },
        observation=(
            "import arrow, datetime\n"
            "d = datetime.date(2020, 1, 2)\n"
            "a = arrow.Arrow.fromordinal(d.toordinal())\n"
            "print('mapped', a.year, a.month, a.day)\n"
            "assert (a.year, a.month, a.day) == (2020, 1, 2)\n"
        ),
        review_notes=("F1 config-propagation: entry ordinal -> middle fromordinal mapping -> "
                      "consumer Arrow field construction. The bug corrupts the ordinal value and "
                      "drops the tzinfo channel as one mapping contract. Distinct production function "
                      "from the factory/api forwarders. Hunks restored; requirements unchanged."),
        final_response=("Ordinal mapping reaches the fields unchanged; "
                        "the actual source-bound observation completed and the patch was submitted."),
    ),
    # ---- batch 4: radon harvester (1) ----
    dict(
        iid="rubik__radon.54b88e58.func_basic__1db71x85",
        key="f1-new-radon-xml-provider",
        canonical_repo="rubik/radon",
        primary_adapter="F", submechanism="F1",
        family="radon.cli.harvest.CCHarvester.as_xml::dict_provider_selection",
        prod_path="radon/cli/harvest.py", prod_module="radon.cli.harvest",
        profile=".venv-f1-radon",
        old="        return dict_to_xml(self._to_dict())\n",
        new="        return dict_to_xml(self._to_dicts())\n",
        read_start=203, read_end=212,
        mechanism={
            "trigger": "Serializing harvested complexity results as XML",
            "base_wrong_behavior": "as_xml feeds the singular _to_dict provider instead of the plural _to_dicts collection, so XML generation runs on the wrong data channel",
            "production_symbols": ["radon.cli.harvest.CCHarvester.as_xml"],
            "causal_change": "Restore _to_dicts() as the dict_to_xml input; serialization untouched",
            "discriminating_observation": {
                "kind": "executed_before_after_source_observation",
                "description": "as_xml with a mocked _to_dicts must call dict_to_xml exactly once with its return value. Before repair the singular provider breaks the assert; after repair it passes.",
            },
            "bug_patch_direction": "introduces_defect",
        },
        observation=(
            "from unittest import mock\n"
            "from radon.cli import harvest\n"
            "h = harvest.CCHarvester([], mock.MagicMock())\n"
            "h._to_dicts = mock.MagicMock(return_value={'a': {'rank': 'A'}})\n"
            "with mock.patch('radon.cli.harvest.dict_to_xml') as d2x:\n"
            "    h.as_xml()\n"
            "    d2x.assert_called_once_with({'a': {'rank': 'A'}})\n"
            "print('xml provider ok')\n"
        ),
        review_notes=("F1 config-propagation: entry harvested results -> middle CCHarvester.as_xml -> "
                      "consumer dict_to_xml. The bug selects the wrong provider method at the call "
                      "edge. Distinct production function from the codeclimate converter. Single line "
                      "restored; requirements unchanged."),
        final_response=("XML generation consumes the plural collection again; "
                        "the actual source-bound observation completed and the patch was submitted."),
    ),
    # ---- batch 4: soupsieve extra (1, margin) ----
    dict(
        iid="facelessuser__soupsieve.a8080d97.func_basic__wwzpejgy",
        key="f1-new-soupsieve-closest-miss",
        canonical_repo="facelessuser/soupsieve",
        primary_adapter="F", submechanism="F1",
        family="soupsieve.SoupSieve.closest::miss_fallback",
        prod_path="soupsieve/css_match.py", prod_module="soupsieve.css_match",
        profile=".venv-f1-soup39", task_python="/usr/bin/python3", task_python39=True,
        license_file="LICENSE.md",
        old="        return CSSMatch(self.selectors, tag, self.namespaces, self.flags).closest() or tag\n",
        new="        return CSSMatch(self.selectors, tag, self.namespaces, self.flags).closest()\n",
        read_start=1533, read_end=1543,
        mechanism={
            "trigger": "Querying the closest matching ancestor with no match in the tree",
            "base_wrong_behavior": "A miss substitutes the queried tag itself instead of propagating the empty result, so callers cannot distinguish miss from match",
            "production_symbols": ["soupsieve.css_match.SoupSieve.closest"],
            "causal_change": "Restore verbatim closest() propagation; matching untouched",
            "discriminating_observation": {
                "kind": "executed_before_after_source_observation",
                "description": "closest('span.a') on a tag with no such ancestor must return None. Before repair the fallback returns the tag; after repair it passes.",
            },
            "bug_patch_direction": "introduces_defect",
        },
        observation=(
            "import soupsieve as sv\n"
            "from bs4 import BeautifulSoup\n"
            "soup = BeautifulSoup('<div><p class=\"a\">x</p></div>', 'html.parser')\n"
            "pat = sv.compile('span.a')\n"
            "out = pat.closest(soup.p)\n"
            "print('closest miss', out)\n"
            "assert out is None\n"
        ),
        review_notes=("F1 config-propagation: entry tag -> middle SoupSieve.closest -> consumer miss "
                      "propagation. The bug substitutes the input tag on miss. Distinct production "
                      "function from match/filter/iselect. Single line restored; requirements "
                      "unchanged. Same pinned 3.9 interpreter as the sibling tasks. Margin task."),
        final_response=("Miss results propagate empty again; "
                        "the actual source-bound observation completed and the patch was submitted."),
    ),
]


# Structurally reviewed but rejected: never (re)prepare these. Reasons live in
# reviewed-candidates.json -> rejected (official-matcher skip-poison, untestable
# nodes, or behavior-preserving mutations that cannot discriminate).
SKIP_IDS = {
    "mozilla__bleach.73871d76.func_basic__2w3h76l9",
    "erikrose__parsimonious.0d3f5f93.func_basic__sy77fb9d",
    "erikrose__parsimonious.0d3f5f93.func_basic__7w2xfe4x",
    "erikrose__parsimonious.0d3f5f93.func_basic__byxkdlaj",
    "joke2k__faker.8b401a7d.func_pm_ctrl_shuffle__nreuzr4i",
    "borntyping__python-colorlog.dfa10f59.func_basic__okirh4rk",
    "borntyping__python-colorlog.dfa10f59.func_basic__hq7m58me",
    "cookiecutter__cookiecutter.b4451231.func_pm_remove_assign__6caklxrl",
    "cookiecutter__cookiecutter.b4451231.func_pm_remove_cond__lmftuhor",
    "django__channels.a144b4b8.func_basic__30xfp610",
    "django__channels.a144b4b8.func_basic__4835kgsg",
    "pyca__pyopenssl.04766a49.func_basic__kicupmxc",
    "pyca__pyopenssl.04766a49.func_basic__h0kbfkcb",
    "mozillazg__python-pinyin.e42dede5.func_basic__7uwmgo5z",
}


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def shafile(p: Path) -> str:
    return sha(p.read_bytes())


def save(p: Path, o) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(o, ensure_ascii=False, indent=2) + "\n")


def label(p: Path) -> str:
    return str(p.resolve().relative_to(ROOT))


def load_rows():
    rows = {}
    for page in sorted((HERE / "pages").glob("train-*.json")):
        if page.name.endswith(".receipt.json"):
            continue
        for e in json.loads(page.read_text()).get("rows", []):
            rows[e["row"]["instance_id"]] = (page, e["row_idx"], e["row"])
    return rows


def load_helper():
    spec = importlib.util.spec_from_file_location("sfr_acq", REPLAY / "prepare_sfr.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def prepare(limit=None):
    helper = load_helper()
    rows = load_rows()
    index = {json.loads(l)["instance_id"]: json.loads(l)
             for l in (ROOT / "competition/vfarl/acquisition/general/smith-task-index.jsonl").read_text().splitlines()}
    review_path = HERE / "reviewed-candidates.json"
    review = json.loads(review_path.read_text()) if review_path.exists() else {
        "source_dataset": "SWE-bench/SWE-smith-py", "source_revision": REV, "physical_split": "train",
        "scope": ("F1-new batch for F600: fresh parameter/config propagation tasks disjoint from "
                  "S600 common 400 families/snapshots and supplemental catalog identities. "
                  "Each entry carries entry->middle->consumer evidence with an executed "
                  "discriminating observation; bug_patch_direction=introduces_defect. "
                  "Single-function default changes are rejected as F1."),
        "tasks": [],
    }
    have = {t["instance_id"] for t in review["tasks"]}
    receipts_path = HERE / "source-receipts.json"
    receipts = json.loads(receipts_path.read_text()) if receipts_path.exists() else {"pages": [], "refs": [], "sources": []}
    for plan in (PLANS if limit is None else PLANS[:limit]):
        iid = plan["iid"]
        if iid in have:
            print("skip prepared", iid, flush=True)
            continue
        if iid in SKIP_IDS:
            print("skip rejected", iid, flush=True)
            continue
        try:
            _prepare_one(plan, rows, index, helper, review, have, receipts,
                         review_path, receipts_path)
        except Exception as e:  # noqa: BLE001 - record and continue the batch
            print(f"SKIP {iid}: {type(e).__name__}: {e}", flush=True)
            review.setdefault("skipped", []).append(
                {"instance_id": iid, "stage": "prepare",
                 "reason": f"{type(e).__name__}: {e}"})
            save(review_path, review)


def _prepare_one(plan, rows, index, helper, review, have, receipts,
                 review_path, receipts_path):
    iid = plan["iid"]
    page, row_idx, row = rows[iid]
    known = index[iid]
    assert known["row_idx"] == row_idx and known["repo"] == row["repo"], "identity mismatch"
    assert (row.get("problem_statement") or "").strip(), "empty statement needs curation; skipped here"
    f2p = row["FAIL_TO_PASS"]
    p2p = row["PASS_TO_PASS"]
    if isinstance(f2p, str):
        f2p = json.loads(f2p)
    if isinstance(p2p, str):
        p2p = json.loads(p2p)
    assert f2p and p2p and not (set(f2p) & set(p2p))
    files = [x.split("::")[0] for x in f2p] + [x.split("::")[0] for x in p2p]
    assert all(x.endswith(".py") for x in files), f"non-py required nodes: {files}"
    for _f in files:
        assert ".." not in Path(_f).parts, f"escaping required path: {_f}"
        _bn = Path(_f).name
        assert (_bn.startswith("test_") or _bn.endswith("_test.py")
                or _bn in {"conftest.py", "test.py"}
                or "tests" in Path(_f).parts or "test" in Path(_f).parts), f"non-test required file: {_f}"
    # original task blob
    original = HERE / (iid + ".task.json")
    save(original, {"row": row})
    # immutable bug ref
    repo_base = row["repo"]  # swesmith/<owner>__<name>.<prefix>
    out = subprocess.check_output(
        ["git", "ls-remote", f"https://github.com/{repo_base}.git", f"refs/heads/{iid}"],
        text=True)
    commit, ref = out.strip().split()
    assert ref == f"refs/heads/{iid}" and len(commit) == 40, out
    receipts["refs"].append({"url": f"https://github.com/{repo_base}.git", "ref": ref,
                             "commit": commit,
                             "response_sha256": sha(out.encode())})
    # sources
    source = ART / (plan["key"] + "-source")
    buggy = helper.acquire(row["repo"], commit, source.resolve())
    prefix = row["repo"].rsplit(".", 1)[-1]
    up_dest = ART / "upstream" / (plan["canonical_repo"].replace("/", "__") + "." + prefix)
    upstream = helper.acquire(plan["canonical_repo"], prefix, up_dest.resolve())
    receipts["sources"].append({"instance_id": iid, "buggy_source": buggy, "upstream_source": upstream})
    build_provenance = None
    if plan.get("build_ext"):
        prof_python = REPLAY / plan["profile"] / "bin/python"
        build = subprocess.run(
            [str(prof_python), "setup.py", "build_ext", "--inplace"],
            cwd=str(source.resolve()), capture_output=True, text=True)
        if build.returncode != 0:
            raise ValueError("in-place extension build failed: " + build.stderr[-2000:])
        import shutil as _shutil
        _shutil.rmtree(source.resolve() / "build", ignore_errors=True)
        for egg in source.resolve().glob("*.egg-info"):
            _shutil.rmtree(egg, ignore_errors=True)
        sos = sorted(str(p.relative_to(source.resolve()))
                     for p in source.resolve().rglob("*.so"))
        if not sos:
            raise ValueError("extension build produced no .so")
        build_provenance = {
            "command": "setup.py build_ext --inplace",
            "profile_python": plan["profile"] + "/bin/python",
            "artifacts": [
                {"path": p,
                 "sha256": shafile(source.resolve() / p)} for p in sos],
            "build_log_tail": build.stdout[-2000:],
            "note": ("Pristine upstream C sources compiled for the pinned task "
                     "interpreter so the suite's optional-extension variants execute; "
                     "no test or requirement files touched."),
        }
        receipts["sources"][-1]["build_provenance"] = build_provenance
    snapshot_group = plan["canonical_repo"] + "::" + upstream["resolved_revision"]
    # verification targets + assets
    targets = sorted({x.split("::")[0] for x in f2p + p2p})
    assets = []
    test_patch = []
    for asset in sorted(up_dest.resolve().rglob("*")):
        if not asset.is_file() or asset.is_symlink():
            continue
        if "__pycache__" in asset.parts or asset.suffix in {".pyc", ".pyo"}:
            continue
        rel = asset.relative_to(up_dest.resolve())
        _name = rel.name
        _intest = any(part in {"tests", "test", "testing"} for part in rel.parts)
        _istest = (_name.startswith("test_") and _name.endswith(".py")) or \
            _name.endswith("_test.py") or _name in {"conftest.py", "test.py"}
        if not (_intest or _istest):
            continue
        raw = asset.read_bytes()
        existing = source.resolve() / rel
        if existing.exists() and existing.read_bytes() != raw:
            raise ValueError("original verification asset differs " + str(rel))
        assets.append({"path": str(rel), "sha256": sha(raw), "bytes": len(raw),
                       "present_in_bug_source": existing.exists()})
        if not existing.exists():
            test_patch.extend(difflib.unified_diff([], raw.decode().splitlines(keepends=True),
                                                   fromfile="/dev/null", tofile="b/" + str(rel)))
    assert all(any(a["path"] == t for a in assets) for t in targets), "missing target file"
    task = {"instance_id": iid, "repo": row["repo"], "base_commit": commit,
            "problem_statement": row["problem_statement"], "patch": "",
            "test_patch": "".join(test_patch), "FAIL_TO_PASS": f2p, "PASS_TO_PASS": p2p,
            "verification_target_files": targets}
    task_path = ART / (plan["key"] + "-task.json")
    save(task_path, task)
    # production edit invariant (single or sequential multi-edit)
    old_text = (source.resolve() / plan["prod_path"]).read_text()
    new_text = (up_dest.resolve() / plan["prod_path"]).read_text()
    pairs = plan.get("edits") or [(plan["old"], plan["new"])]
    current = old_text
    for _old, _new in pairs:
        assert current.count(_old) == 1, "edit not unique: " + _old[:80]
        current = current.replace(_old, _new)
    assert current == new_text, "edits do not explain full file diff"
    # recipe
    binding = ("from pathlib import Path\nimport hashlib,importlib\n"
               f"m=importlib.import_module({plan['prod_module']!r})\n"
               "actual=Path(m.__file__).resolve();"
               f"expected=(Path.cwd()/{plan['prod_path']!r}).resolve()\n"
               "assert actual==expected,(actual,expected)\n"
               "print('source_binding',actual,'sha256',hashlib.sha256(actual.read_bytes()).hexdigest())\n")
    command = "python3 - <<'PYREPLAY'\n" + binding + plan["observation"] + "PYREPLAY"

    def action(tool, args, status="ok"):
        a = {"tool": tool, "arguments": args, "expected_status": status}
        if tool == "run_command":
            a["reviewed_command_sha256"] = sha(args["command"].encode())
        return a

    actions = [
        action("get_status", {}),
        action("read_file", {"filepath": plan["prod_path"], "start_line": plan["read_start"],
                             "end_line": plan["read_end"]}),
        action("run_command", {"command": command}, "error"),
    ]
    for _old, _new in pairs:
        actions.append(action("edit_file", {"filepath": plan["prod_path"],
                                            "old_string": _old, "new_string": _new}))
    actions.extend([
        action("run_command", {"command": command}),
        action("get_status", {}),
        action("submit_patch", {}),
    ])
    recipe = {
        "evidence_kind": "human_reviewed_off_policy_demonstration",
        "reviewed_by": "F1-new curator source-contract and explicit command review 2026-10-25",
        "review_notes": plan["review_notes"] + (" Original F2P/P2P/full files unchanged; "
                                                "Smith patch is bug-introducing. No sampled policy or preference pair."),
        "actions": actions,
        "final_response": plan["final_response"],
    }
    recipe_path = REPLAY / (plan["key"] + "-reviewed-recipe.json")
    save(recipe_path, recipe)
    try:
        node = next(n for n in ast.parse(new_text).body
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)))
        prod_symbol = node.name
    except StopIteration:
        prod_symbol = plan["prod_path"]
    entry = {
        "instance_id": iid, "key": plan["key"], "canonical_repo": plan["canonical_repo"],
        "primary_adapter": plan["primary_adapter"], "submechanism": plan["submechanism"],
        "repair_family_id": plan["family"], "snapshot_group_id": snapshot_group,
        "original_task": label(original), "original_task_sha256": shafile(original),
        "source_task_sha": shafile(original),
        "source_dataset": "SWE-bench/SWE-smith-py", "source_revision": REV,
        "physical_split": "train", "row_idx": row_idx,
        "source_page": label(page), "source_page_sha256": shafile(page),
        "buggy_source": buggy, "upstream_source": upstream, "execution_commit": commit,
        "workspace": label(source.resolve()), "adapted_task": label(task_path),
        "recipe": label(recipe_path),
        "production_symbol": prod_symbol, "production_path": plan["prod_path"],
        "mechanism_evidence": plan["mechanism"],
        "build_provenance": build_provenance,
        "original_patch_semantics": "bug_introducing",
        "original_FAIL_TO_PASS_count": len(f2p), "original_PASS_TO_PASS_count": len(p2p),
        "verification_target_files": targets, "verification_assets": assets,
        "original_requirements_verbatim": True,
        "license_evidence": {
            "source": {"path": label(up_dest.resolve() / plan.get("license_file", "LICENSE")),
                       "sha256": shafile(up_dest.resolve() / plan.get("license_file", "LICENSE")),
                       "revision": upstream["resolved_revision"]},
            "dataset": {"license": "MIT", "path": DATASET_README,
                        "sha256": shafile(ROOT / DATASET_README), "revision": REV},
        },
    }
    # disjointness check
    used_ids = set(__import__("pickle").load(
        open("/private/var/folders/9s/sdk1x32s58g8dycwh6c74bqc0000gp/T/opencode/used_ids.pkl", "rb")))
    used_fs = __import__("pickle").load(
        open("/private/var/folders/9s/sdk1x32s58g8dycwh6c74bqc0000gp/T/opencode/used_famsnaps.pkl", "rb"))
    assert iid not in used_ids, "instance reuse"
    assert entry["repair_family_id"] not in used_fs["fams"], "family reuse"
    assert entry["snapshot_group_id"] not in used_fs["snaps"], "snapshot reuse"
    review["tasks"].append(entry)
    have.add(iid)
    save(review_path, review)
    save(receipts_path, receipts)
    print("prepared", iid, plan["key"], flush=True)


def run(limit=None, attempt=1):
    spec = importlib.util.spec_from_file_location("norm", REPLAY / "normalize_results.py")
    n = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(n)
    review = json.loads((HERE / "reviewed-candidates.json").read_text())
    tasks = review["tasks"] if limit is None else review["tasks"][:limit]
    for t in tasks:
        key = t["key"]
        output = ART / (key + f"-run-{attempt:03d}")
        report_path = REPLAY / (key + "-replay-result.json")
        if report_path.exists():
            print("retained_existing_result", t["instance_id"], flush=True)
            continue
        profile = None
        source_layout = "."
        task_python = None
        pytest_plugins = None
        for plan in PLANS:
            if plan["key"] == key:
                profile = REPLAY / plan["profile"] / "lib/python3.12/site-packages"
                if plan.get("task_python39"):
                    profile = REPLAY / plan["profile"] / "lib/python3.9/site-packages"
                source_layout = plan.get("source_layout", ".")
                task_python = plan.get("task_python")
                pytest_plugins = plan.get("pytest_plugins")
                break
        args = [sys.executable, str(REPLAY / "run_official.py"),
                "--task", str(ROOT / t["adapted_task"]),
                "--workspace", str(ROOT / t["workspace"]),
                "--recipe", str(ROOT / t["recipe"]),
                "--output", str(output),
                "--verify", "--verify-baseline",
                "--source-layout", source_layout,
                "--task-site-packages", str(profile),
                "--task-site-packages-mode", "copy",
                "--time-minutes", "30", "--verification-timeout", "300"]
        if task_python:
            args.extend(["--task-python", task_python])
        import os as _os
        run_env = dict(_os.environ)
        if pytest_plugins:
            # Load the repository's own pytest plugin (fixture provider) without
            # installing anything or touching tests; recorded in the report.
            run_env["PYTEST_PLUGINS"] = pytest_plugins
        with (ART / (key + ".log")).open("w") as log:
            result = subprocess.run(args, stdout=log, stderr=subprocess.STDOUT,
                                    env=run_env)
        if not (output / "run-summary.json").exists():
            print("actual_run_failed_without_summary", t["instance_id"], result.returncode, flush=True)
            continue
        summary = json.loads((output / "run-summary.json").read_text())
        original = json.loads((ROOT / t["original_task"]).read_text())["row"]
        adapted = json.loads((ROOT / t["adapted_task"]).read_text())
        if summary["source_task_sha256"] != shafile(ROOT / t["adapted_task"]) or \
                shafile(output / "recipe.json") != shafile(ROOT / t["recipe"]):
            raise ValueError("run bytes differ from reviewed bytes")
        reports = {}
        for phase in ("baseline", "repaired"):
            found = list(output.glob(phase + "-*_junit_*.xml"))
            if len(found) == 1:
                reports[phase] = n.junit_report(found[0], original, adapted)
        summary.update({
            "original_task": t["original_task"], "adapted_task": t["adapted_task"],
            "recipe": t["recipe"], "source_review": label(HERE / "reviewed-candidates.json"),
            "source_review_sha256": shafile(HERE / "reviewed-candidates.json"),
            "pytest_plugin_env": pytest_plugins,
            "replay_output": label(output), "reports": reports,
            "original_verifier_requirements_relaxed": False,
            "required_node_identifiers_verbatim_preserved": True,
            "source_FAIL_TO_PASS_count": t["original_FAIL_TO_PASS_count"],
            "source_PASS_TO_PASS_count": t["original_PASS_TO_PASS_count"],
            "source_train_revision": REV, "bug_branch_revision": t["execution_commit"],
            "primary_adapter": t["primary_adapter"], "submechanism": t["submechanism"],
            "repair_family_id": t["repair_family_id"], "snapshot_group_id": t["snapshot_group_id"],
            "mechanism": t["mechanism_evidence"],
            "training_eligibility": {
                "human_off_policy_tool_demonstration": bool(
                    summary.get("resolved_by_local_official_verify_task")
                    and reports.get("baseline", {}).get("all_original_F2P_observed_failed")
                    and reports.get("repaired", {}).get("original_official_junit_validation", [False])[0]),
                "competition_Docker_equivalence": False, "GPU_training_started": False},
        })
        save(report_path, summary)
        print(json.dumps({"instance_id": t["instance_id"],
                          "resolved": summary.get("resolved_by_local_official_verify_task"),
                          "baseline_all_F2P_failed": reports.get("baseline", {}).get("all_original_F2P_observed_failed"),
                          "repaired": {k: reports.get("repaired", {}).get(k)
                                       for k in ("tests", "failures", "errors", "skipped")}}, ensure_ascii=False),
              flush=True)


def count():
    spec = importlib.util.spec_from_file_location("norm", REPLAY / "normalize_results.py")
    n = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(n)
    review = json.loads((HERE / "reviewed-candidates.json").read_text())
    catalog = {t["instance_id"]: {
        "instance_id": t["instance_id"], "canonical_repo": t["canonical_repo"],
        "primary_adapter": t["primary_adapter"], "submechanism": t["submechanism"],
        "repair_family_id": t["repair_family_id"], "snapshot_group_id": t["snapshot_group_id"],
        "execution_commit": t["execution_commit"], "source_dataset": t["source_dataset"],
        "source_revision": t["source_revision"], "physical_split": t["physical_split"],
        "original_FAIL_TO_PASS_count": t["original_FAIL_TO_PASS_count"],
        "original_PASS_TO_PASS_count": t["original_PASS_TO_PASS_count"],
        "mechanism_evidence": t["mechanism_evidence"],
        "original_patch_semantics": t["original_patch_semantics"],
        "source_task_path": t["original_task"], "source_task_sha256": t["original_task_sha256"],
        "source_review_path": "competition/vfarl/acquisition/f1-new/reviewed-candidates.json",
        "source_review_sha256": shafile(HERE / "reviewed-candidates.json"),
    } for t in review["tasks"]}
    eligible, rows = [], []
    for t in review["tasks"]:
        rp = REPLAY / (t["key"] + "-replay-result.json")
        if not rp.exists():
            rows.append({"instance_id": t["instance_id"], "status": "no_replay"})
            continue
        try:
            res = n.normalize(rp, catalog, {})
        except Exception as e:  # noqa: BLE001
            rows.append({"instance_id": t["instance_id"], "status": f"normalize_error: {e}"})
            continue
        ok = res["eligible_for_human_off_policy_SFT"]
        rows.append({"instance_id": t["instance_id"], "status": "eligible" if ok else "ineligible",
                     "reasons": res["ineligibility_reasons"],
                     "f2p": res["source_FAIL_TO_PASS_count"], "p2p": res["source_PASS_TO_PASS_count"],
                     "run": res.get("actual_run_directory")})
        if ok:
            eligible.append(t["instance_id"])
    print(json.dumps({"eligible": len(eligible), "total": len(review["tasks"]), "rows": rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("prepare", "run", "count"))
    ap.add_argument("--limit", type=int, default=None)
    o = ap.parse_args()
    {"prepare": prepare, "run": run, "count": count}[o.stage](**({} if o.stage == "count" else {"limit": o.limit}))
