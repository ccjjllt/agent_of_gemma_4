import re
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from moto.core.base_backend import BackendDict, BaseBackend
from moto.core.common_models import BaseModel
from moto.core.utils import (
    camelcase_to_underscores,
    iso_8601_datetime_with_milliseconds,
)
from moto.moto_api._internal import mock_random
from moto.utilities.paginator import paginate
from moto.utilities.tagging_service import TaggingService
from moto.utilities.utils import ARN_PARTITION_REGEX, PARTITION_NAMES

from .exceptions import WAFNonexistentItemException, WAFV2DuplicateItemException
from .utils import make_arn_for_wacl

if TYPE_CHECKING:
    from moto.apigateway.models import Stage


US_EAST_1_REGION = "us-east-1"
GLOBAL_REGION = "global"
APIGATEWAY_REGEX = (
    ARN_PARTITION_REGEX
    + r":apigateway:[a-zA-Z0-9-]+::/restapis/[a-zA-Z0-9]+/stages/[a-zA-Z0-9]+"
)

PAGINATION_MODEL = {
    "list_ip_sets": {
        "input_token": "next_marker",
        "limit_key": "limit",
        "limit_default": 100,
        "unique_attribute": "arn",
    },
    "list_logging_configurations": {
        "input_token": "next_marker",
        "limit_key": "limit",
        "limit_default": 100,
        "unique_attribute": "arn",
    },
    "list_rule_groups": {
        "input_token": "next_marker",
        "limit_key": "limit",
        "limit_default": 100,
        "unique_attribute": "arn",
    },
    "list_tags_for_resource": {
        "input_token": "next_marker",
        "limit_key": "limit",
        "limit_default": 100,
        "unique_attribute": "Key",
    },
    "list_web_acls": {
        "input_token": "next_marker",
        "limit_key": "limit",
        "limit_default": 100,
        "unique_attribute": "arn",
    },
}


class FakeRule(BaseModel):
    def __init__(
        self,
        name: str,
        priority: int,
        statement: Dict[str, Any],
        visibility_config: Dict[str, Union[str, bool]],
        action: Optional[Dict[str, Any]] = None,
        captcha_config: Optional[Dict[str, Dict[str, int]]] = None,
        challenge_config: Optional[Dict[str, Dict[str, int]]] = None,
        override_action: Optional[Dict[str, Any]] = None,
        rule_labels: Optional[List[Dict[str, str]]] = None,
    ):
        self.name = name
        self.priority = priority
        self.statement = statement
        self.visibility_config = visibility_config
        self.action = action
        self.captcha_config = captcha_config
        self.challenge_config = challenge_config
        self.override_action = override_action
        self.rule_labels = rule_labels

    def get_consumed_label(self) -> Optional[str]:
        return self.statement.get("LabelMatchStatement", {}).get("Key")

    def get_available_labels(self) -> Optional[List[str]]:
        return [r["Name"] for r in self.rule_labels] if self.rule_labels else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Name": self.name,
            "Action": self.action,
            "Priority": self.priority,
            "Statement": self.statement,
            "VisibilityConfig": self.visibility_config,
            "CaptchaConfig": self.captcha_config,
            "ChallengeConfig": self.challenge_config,
            "OverrideAction": self.override_action,
            "RuleLabels": self.rule_labels,
        }


class FakeWebACL(BaseModel):
    """
    https://docs.aws.amazon.com/waf/latest/APIReference/API_WebACL.html
    """

    def __init__(
        self,
        name: str,
        account: str,
        arn: str,
        wacl_id: str,
        visibility_config: Dict[str, Any],
        default_action: Dict[str, Any],
        description: Optional[str],
        rules: Optional[List[FakeRule]],
        association_config: Optional[Dict[str, Any]] = None,
        captcha_config: Optional[Dict[str, Any]] = None,
        challenge_config: Optional[Dict[str, Any]] = None,
        custom_response_bodies: Optional[Dict[str, Any]] = None,
        token_domains: Optional[List[str]] = None,
    ):
        self.name = name
        self.account = account
        self.created_time = iso_8601_datetime_with_milliseconds()
        self.id = wacl_id
        self.arn = arn
        self.description = description or ""
        self.capacity = 3
        self.rules = rules or []
        self.visibility_config = visibility_config
        self.default_action = default_action
        self.lock_token = self._generate_lock_token()
        self.associated_resources: List[str] = []
        self.association_config = association_config
        self.captcha_config = captcha_config
        self.challenge_config = challenge_config
        self.custom_response_bodies = custom_response_bodies
        self.token_domains = token_domains
        self.label_namespace = self._get_label_namespace()

    def _generate_lock_token(self) -> str:
        return str(mock_random.uuid4())

    def _get_label_namespace(self) -> str:
        return f"awswaf:{self.account}:webacl:{self.name}:"

    def update(
        self,
        default_action: Optional[Dict[str, Any]],
        rules: Optional[List[FakeRule]],
        description: Optional[str],
        visibility_config: Optional[Dict[str, Any]],
        custom_response_bodies: Optional[Dict[str, Any]],
        captcha_config: Optional[Dict[str, Any]],
        challenge_config: Optional[Dict[str, Any]],
        token_domains: Optional[List[str]],
        association_config: Optional[Dict[str, Any]],
    ) -> None:
        if default_action is not None:
            self.default_action = default_action
        if rules is not None:
            self.rules = rules
        if description is not None:
            self.description = description
        if visibility_config is not None:
            self.visibility_config = visibility_config
        if custom_response_bodies is not None:
            self.custom_response_bodies = custom_response_bodies
        if captcha_config is not None:
            self.captcha_config = captcha_config
        if challenge_config is not None:
            self.challenge_config = challenge_config
        if token_domains is not None:
            self.token_domains = token_domains
        if association_config is not None:
            self.association_config = association_config
        self.lock_token = self._generate_lock_token()

    def to_short_dict(self) -> Dict[str, Any]:
        # Format for summary https://docs.aws.amazon.com/waf/latest/APIReference/API_CreateWebACL.html (response syntax section)
        return {
            "ARN": self.arn,
            "Description": self.description,
            "Id": self.id,
            "Name": self.name,
            "LockToken": self.lock_token,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Name": self.name,
            "Id": self.id,
            "ARN": self.arn,
            "Description": self.description,
            "Rules": [r.to_dict() for r in self.rules],
            "DefaultAction": self.default_action,
            "VisibilityConfig": self.visibility_config,
            "Capacity": self.capacity,
            "LockToken": self.lock_token,
            "AssociationConfig": self.association_config,
            "CaptchaConfig": self.captcha_config,
            "ChallengeConfig": self.challenge_config,
            "CustomResponseBodies": self.custom_response_bodies,
            "TokenDomains": self.token_domains,
            "LabelNamespace": self.label_namespace,
        }


class FakeRuleGroup(BaseModel):
    def __init__(
        self,
        account: str,
        name: str,
        id: str,
        arn: str,
        scope: str,
        capacity: int,
        visibility_config: Dict[str, Any],
        description: Optional[str],
        rules: Optional[List[FakeRule]],
        custom_response_bodies: Optional[Dict[str, Any]] = None,
    ):
        self.account = account
        self.name = name
        self.id = id
        self.arn = arn
        self.lock_token = self._generate_lock_token()
        self.scope = scope
        self.capacity = capacity
        self.description = description or ""
        self.rules = rules or []
        self.visibility_config = visibility_config
        self.custom_response_bodies = custom_response_bodies
        self.label_namespace = self._get_label_namespace()
        self.available_labels = self._get_available_labels()
        self.consumed_labels = self._get_consumed_labels()

    def _generate_lock_token(self) -> str:
        return str(mock_random.uuid4())

    def _get_available_labels(self) -> Optional[List[Dict[str, str]]]:
        return (
            [
                {"Name": f"{self.label_namespace}{label}"}
                for rule in self.rules
                for label in (rule.get_available_labels() or [])
            ]
            if self.rules
            else None
        )

    def _get_consumed_labels(self) -> Optional[List[Dict[str, str]]]:
        return (
            [
                {"Name": f"{self.label_namespace}{label}"}
                for rule in self.rules
                if (label := rule.get_consumed_label())
            ]
            if self.rules
            else None
        )

    def _get_label_namespace(self) -> str:
        return f"awswaf:{self.account}:rulegroup:{self.name}:"

    def update(
        self,
        description: Optional[str],
        rules: Optional[List[FakeRule]],
        visibility_config: Optional[Dict[str, Any]],
        custom_response_bodies: Optional[Dict[str, Any]],
    ) -> str:
        if description is not None:
            self.description = description
        if rules is not None:
            self.rules = rules
        if visibility_config is not None:
            self.visibility_config = visibility_config
        if custom_response_bodies is not None:
            self.custom_response_bodies = custom_response_bodies

        self.lock_token = self._generate_lock_token()
        return self.lock_token

    def to_short_dict(self) -> Dict[str, Any]:
        return {
            "Name": self.name,
            "Id": self.id,
            "Description": self.description,
            "LockToken": self.lock_token,
            "ARN": self.arn,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Name": self.name,
            "Id": self.id,
            "Capacity": self.capacity,
            "ARN": self.arn,
            "Description": self.description,
            "Rules": [r.to_dict() for r in self.rules],
            "VisibilityConfig": self.visibility_config,
            "LabelNamespace": self.label_namespace,
            "AvailableLabels": self.available_labels,
            "ConsumedLabels": self.consumed_labels,
            "CustomResponseBodies": self.custom_response_bodies,
            "LockToken": self.lock_token,
        }


class WAFV2Backend(BaseBackend):
    """
    https://docs.aws.amazon.com/waf/latest/APIReference/API_Operations_AWS_WAFV2.html
    """

    def __init__(self, region_name: str, account_id: str):
        super().__init__(region_name, account_id)
        self.wacls: Dict[str, FakeWebACL] = OrderedDict()
        self.tagging_service = TaggingService()
        # TODO: self.load_balancers = OrderedDict()

    def associate_web_acl(self, web_acl_arn: str, resource_arn: str) -> None:
        web_acl = self.wacls.get(web_acl_arn)
        if not web_acl:
            raise WAFNonexistentItemException
        web_acl.associated_resources.append(resource_arn)
        # Special Case - APIGateway wants to know about the WebACL it's associated to
        stage = self._find_apigw_stage(resource_arn)
        if stage:
            stage.web_acl_arn = web_acl_arn

    def disassociate_web_acl(self, resource_arn: str) -> None:
        for web_acl in self.wacls.values():
            if resource_arn in web_acl.associated_resources:
                web_acl.associated_resources.remove(resource_arn)
                break
        stage = self._find_apigw_stage(resource_arn)
        if stage:
            stage.web_acl_arn = None

    def get_web_acl_for_resource(self, resource_arn: str) -> Optional[FakeWebACL]:
        for wacl in self.wacls.values():
            if resource_arn in wacl.associated_resources:
                return wacl
        return None

    def _find_apigw_stage(self, resource_arn: str) -> Optional["Stage"]:  # type: ignore
        try:
            if re.search(APIGATEWAY_REGEX, resource_arn):
                region = resource_arn.split(":")[3]
                rest_api_id = resource_arn.split("/")[-3]
                stage_name = resource_arn.split("/")[-1]

                from moto.apigateway import apigateway_backends

                apigw = apigateway_backends[self.account_id][region]
                return apigw.get_stage(rest_api_id, stage_name)
        except:  # noqa: E722 Do not use bare except
            return None

    def _generate_id(self) -> str:
        return str(mock_random.uuid4())

    def create_web_acl(
        self,
        name: str,
        visibility_config: Dict[str, Any],
        default_action: Dict[str, Any],
        scope: str,
        description: str,
        tags: List[Dict[str, str]],
        rules: List[Dict[str, Any]],
        association_config: Optional[Dict[str, Any]],
        captcha_config: Optional[Dict[str, Any]],
        challenge_config: Optional[Dict[str, Any]],
        custom_response_bodies: Optional[Dict[str, Any]],
        token_domains: Optional[List[str]],
    ) -> FakeWebACL:
        wacl_id = self._generate_id()
        arn = make_arn_for_wacl(
            name=name,
            account_id=self.account_id,
            region_name=self.region_name,
            _id=wacl_id,
            scope=scope,
        )
        if arn in self.wacls or self._is_duplicate_name(name):
            raise WAFV2DuplicateItemException()
        rule_objs = [
            FakeRule(**{camelcase_to_underscores(k): v for k, v in rule.items()})
            for rule in rules
        ]
        new_wacl = FakeWebACL(
            name=name,
            account=self.account_id,
            arn=arn,
            wacl_id=wacl_id,
            visibility_config=visibility_config,
            default_action=default_action,
            description=description,
            rules=rule_objs,
            association_config=association_config,
            captcha_config=captcha_config,
            challenge_config=challenge_config,
            custom_response_bodies=custom_response_bodies,
            token_domains=token_domains,
        )
        self.wacls[arn] = new_wacl
        self.tag_resource(arn, tags)
        return new_wacl

    def delete_web_acl(self, name: str, scope: str, _id: str, lock_token: str) -> None:
        arn = make_arn_for_wacl(
            name=name,
            account_id=self.account_id,
            region_name=self.region_name,
            _id=_id,
            scope=scope,
        )
        if arn not in self.wacls:
            raise WAFNonexistentItemException
        wacl = self.wacls[arn]
        if wacl.lock_token != lock_token:
            raise WAFV2DuplicateItemException()
        self.wacls.pop(arn)

    def get_web_acl(self, name: str, _id: str) -> FakeWebACL:
        for wacl in self.wacls.values():
            if wacl.name == name and wacl.id == _id:
                return wacl
        raise WAFNonexistentItemException

    @paginate(PAGINATION_MODEL)  # type: ignore
    def list_web_acls(self) -> List[FakeWebACL]:
        return [wacl for wacl in self.wacls.values()]

    def _is_duplicate_name(self, name: str) -> bool:
        all_wacl_names = set(wacl.name for wacl in self.wacls.values())
        return name in all_wacl_names

    @paginate(PAGINATION_MODEL)  # type: ignore
    def list_rule_groups(self, scope: str) -> List[FakeRuleGroup]:
        rule_groups = [
            group for group in [] if group.scope == scope
        ]
        return rule_groups

    @paginate(PAGINATION_MODEL)  # type: ignore
    def list_tags_for_resource(self, arn: str) -> List[Dict[str, str]]:
        return self.tagging_service.list_tags_for_resource(arn)["Tags"]

    def tag_resource(self, arn: str, tags: List[Dict[str, str]]) -> None:
        self.tagging_service.tag_resource(arn, tags)

    def untag_resource(self, arn: str, tag_keys: List[str]) -> None:
        self.tagging_service.untag_resource_using_names(arn, tag_keys)

    def update_web_acl(
        self,
        name: str,
        _id: str,
        default_action: Optional[Dict[str, Any]],
        rules: Optional[List[Dict[str, Any]]],
        description: Optional[str],
        visibility_config: Optional[Dict[str, Any]],
        lock_token: str,
        custom_response_bodies: Optional[Dict[str, Any]],
        captcha_config: Optional[Dict[str, Any]],
        challenge_config: Optional[Dict[str, Any]],
        token_domains: Optional[List[str]],
        association_config: Optional[Dict[str, Any]],
    ) -> str:
        acl = self.get_web_acl(name, _id)
        if acl.lock_token != lock_token:
            raise WAFV2DuplicateItemException()
        rule_objs = (
            [
                FakeRule(**{camelcase_to_underscores(k): v for k, v in rule.items()})
                for rule in rules
            ]
            if rules
            else None
        )
        acl.update(
            default_action,
            rule_objs,
            description,
            visibility_config,
            custom_response_bodies,
            captcha_config,
            challenge_config,
            token_domains,
            association_config,
        )
        return acl.lock_token


wafv2_backends = BackendDict(WAFV2Backend, "wafv2", additional_regions=PARTITION_NAMES)