import json

from moto.core.common_types import TYPE_RESPONSE
from moto.core.responses import BaseResponse

from .models import GLOBAL_REGION, WAFV2Backend, wafv2_backends


class WAFV2Response(BaseResponse):
    def __init__(self) -> None:
        super().__init__(service_name="wafv2")

    @property
    def wafv2_backend(self) -> WAFV2Backend:
        return wafv2_backends[self.current_account][self.region]

    def associate_web_acl(self) -> TYPE_RESPONSE:
        body = json.loads(self.body)
        web_acl_arn = body["WebACLArn"]
        resource_arn = body["ResourceArn"]
        self.wafv2_backend.associate_web_acl(web_acl_arn, resource_arn)
        return 200, {}, "{}"

    def disassociate_web_acl(self) -> TYPE_RESPONSE:
        body = json.loads(self.body)
        resource_arn = body["ResourceArn"]
        self.wafv2_backend.disassociate_web_acl(resource_arn)
        return 200, {}, "{}"

    def get_web_acl_for_resource(self) -> TYPE_RESPONSE:
        body = json.loads(self.body)
        resource_arn = body["ResourceArn"]
        web_acl = self.wafv2_backend.get_web_acl_for_resource(resource_arn)
        response = {"WebACL": web_acl.to_dict() if web_acl else None}
        response_headers = {"Content-Type": "application/json"}
        return 200, response_headers, json.dumps(response)

    def create_web_acl(self) -> TYPE_RESPONSE:
        """https://docs.aws.amazon.com/waf/latest/APIReference/API_CreateWebACL.html (response syntax section)"""

        scope = self._get_param("Scope")
        if scope == "CLOUDFRONT":
            self.region = GLOBAL_REGION
        name = self._get_param("Name")
        body = json.loads(self.body)
        description = body.get("Description")
        tags = body.get("Tags", [])
        rules = body.get("Rules", [])
        association_config = body.get("AssociationConfig")
        captcha_config = body.get("CaptchaConfig")
        challenge_config = body.get("ChallengeConfig")
        custom_response_bodies = body.get("CustomResponseBodies")
        token_domains = body.get("TokenDomains")
        web_acl = self.wafv2_backend.create_web_acl(
            name,
            body["VisibilityConfig"],
            body["DefaultAction"],
            scope,
            description,
            tags,
            rules,
            association_config,
            captcha_config,
            challenge_config,
            custom_response_bodies,
            token_domains,
        )
        response = {"Summary": web_acl.to_short_dict()}
        response_headers = {"Content-Type": "application/json"}
        return 200, response_headers, json.dumps(response)

    def delete_web_acl(self) -> TYPE_RESPONSE:
        scope = self._get_param("Scope")
        if scope == "CLOUDFRONT":
            self.region = GLOBAL_REGION
        name = self._get_param("Name")
        _id = self._get_param("Id")
        lock_token = self._get_param("LockToken")
        self.wafv2_backend.delete_web_acl(name, scope, _id, lock_token)
        response_headers = {"Content-Type": "application/json"}
        return 200, response_headers, "{}"

    def get_web_acl(self) -> TYPE_RESPONSE:
        scope = self._get_param("Scope")
        if scope == "CLOUDFRONT":
            self.region = GLOBAL_REGION
        name = self._get_param("Name")
        _id = self._get_param("Id")
        web_acl = self.wafv2_backend.get_web_acl(name, _id)
        response = {"WebACL": web_acl.to_dict(), "LockToken": web_acl.lock_token}
        response_headers = {"Content-Type": "application/json"}
        return 200, response_headers, json.dumps(response)

    def list_web_ac_ls(self) -> TYPE_RESPONSE:
        """https://docs.aws.amazon.com/waf/latest/APIReference/API_ListWebACLs.html (response syntax section)"""

        scope = self._get_param("Scope")
        if scope == "CLOUDFRONT":
            self.region = GLOBAL_REGION
        limit = self._get_int_param("Limit")
        next_marker = self._get_param("NextMarker")
        web_acls, next_marker = self.wafv2_backend.list_web_acls(
            limit=limit, next_marker=next_marker
        )
        response = {
            "NextMarker": next_marker,
            "WebACLs": [web.to_short_dict() for web in web_acls],
        }
        response_headers = {"Content-Type": "application/json"}
        return 200, response_headers, json.dumps(response)

    def list_rule_groups(self) -> TYPE_RESPONSE:
        scope = self._get_param("Scope")
        if scope == "CLOUDFRONT":
            self.region = GLOBAL_REGION
        limit = self._get_int_param("Limit")
        next_marker = self._get_param("NextMarker")
        rule_groups, next_marker = self.wafv2_backend.list_rule_groups(
            scope, limit=limit, next_marker=next_marker
        )
        response = {
            "RuleGroups": [rg.to_short_dict() for rg in rule_groups],
            "NextMarker": next_marker,
        }
        response_headers = {"Content-Type": "application/json"}
        return 200, response_headers, json.dumps(response)

    def list_tags_for_resource(self) -> TYPE_RESPONSE:
        arn = self._get_param("ResourceARN")
        self.region = arn.split(":")[3]
        limit = self._get_int_param("Limit")
        next_marker = self._get_param("NextMarker")
        tags, next_marker = self.wafv2_backend.list_tags_for_resource(
            arn, limit=limit, next_marker=next_marker
        )
        response = {
            "TagInfoForResource": {"ResourceARN": arn, "TagList": tags},
            "NextMarker": next_marker,
        }
        response_headers = {"Content-Type": "application/json"}
        return 200, response_headers, json.dumps(response)

    def tag_resource(self) -> TYPE_RESPONSE:
        body = json.loads(self.body)
        arn = body.get("ResourceARN")
        self.region = arn.split(":")[3]
        tags = body.get("Tags")
        self.wafv2_backend.tag_resource(arn, tags)
        return 200, {}, "{}"

    def untag_resource(self) -> TYPE_RESPONSE:
        body = json.loads(self.body)
        arn = body.get("ResourceARN")
        self.region = arn.split(":")[3]
        tag_keys = body.get("TagKeys")
        self.wafv2_backend.untag_resource(arn, tag_keys)
        return 200, {}, "{}"

    def update_web_acl(self) -> TYPE_RESPONSE:
        body = json.loads(self.body)
        name = body.get("Name")
        _id = body.get("Id")
        scope = body.get("Scope")
        if scope == "CLOUDFRONT":
            self.region = GLOBAL_REGION
        default_action = body.get("DefaultAction")
        rules = body.get("Rules")
        description = body.get("Description")
        visibility_config = body.get("VisibilityConfig")
        lock_token = body.get("LockToken")
        custom_response_bodies = body.get("CustomResponseBodies")
        captcha_config = body.get("CaptchaConfig")
        challenge_config = body.get("ChallengeConfig")
        token_domains = body.get("TokenDomains")
        association_config = body.get("AssociationConfig")
        new_lock_token = self.wafv2_backend.update_web_acl(
            name,
            _id,
            default_action,
            rules,
            description,
            visibility_config,
            lock_token,
            custom_response_bodies,
            captcha_config,
            challenge_config,
            token_domains,
            association_config,
        )
        return 200, {}, json.dumps({"NextLockToken": new_lock_token})

    def create_rule_group(self) -> TYPE_RESPONSE:
        name = self._get_param("Name")
        scope = self._get_param("Scope")
        capacity = self._get_param("Capacity")
        description = self._get_param("Description")
        rules = self._get_param("Rules", [])
        visibility_config = self._get_param("VisibilityConfig")
        tags = self._get_param("Tags")
        custom_response_bodies = self._get_param("CustomResponseBodies")
        if scope == "CLOUDFRONT":
            self.region = GLOBAL_REGION
        group = self.wafv2_backend.create_rule_group(
            name=name,
            scope=scope,
            capacity=capacity,
            description=description,
            rules=rules,
            visibility_config=visibility_config,
            tags=tags,
            custom_response_bodies=custom_response_bodies,
        )
        return 200, {}, json.dumps(dict(Summary=group.to_short_dict()))

    def update_rule_group(self) -> TYPE_RESPONSE:
        name = self._get_param("Name")
        scope = self._get_param("Scope")
        id = self._get_param("Id")
        description = self._get_param("Description")
        rules = self._get_param("Rules")
        visibility_config = self._get_param("VisibilityConfig")
        lock_token = self._get_param("LockToken")
        custom_response_bodies = self._get_param("CustomResponseBodies")
        if scope == "CLOUDFRONT":
            self.region = GLOBAL_REGION
        updated_group = self.wafv2_backend.update_rule_group(
            name=name,
            scope=scope,
            id=id,
            description=description,
            rules=rules,
            visibility_config=visibility_config,
            lock_token=lock_token,
            custom_response_bodies=custom_response_bodies,
        )
        return 200, {}, json.dumps(dict(NextLockToken=updated_group.lock_token))

    def delete_rule_group(self) -> TYPE_RESPONSE:
        name = self._get_param("Name")
        scope = self._get_param("Scope")
        id = self._get_param("Id")
        lock_token = self._get_param("LockToken")
        if scope == "CLOUDFRONT":
            self.region = GLOBAL_REGION
        self.wafv2_backend.delete_rule_group(
            name=name,
            scope=scope,
            id=id,
            lock_token=lock_token,
        )
        return 200, {}, json.dumps(dict())

    def get_rule_group(self) -> TYPE_RESPONSE:
        name = self._get_param("Name")
        scope = self._get_param("Scope")
        id = self._get_param("Id")
        arn = self._get_param("ARN")
        if scope == "CLOUDFRONT" or (
            isinstance(arn, str) and arn.split(":")[3] == GLOBAL_REGION
        ):
            self.region = GLOBAL_REGION
        rule_group = self.wafv2_backend.get_rule_group(
            name=name,
            scope=scope,
            id=id,
            arn=arn,
        )
        group_dict = rule_group.to_dict()
        lock_token = group_dict.pop("LockToken")
        return 200, {}, json.dumps(dict(RuleGroup=group_dict, LockToken=lock_token))


# notes about region and scope
# --scope = CLOUDFRONT is ALWAYS us-east-1 (but we use "global" instead to differentiate between REGIONAL us-east-1)
# --scope = REGIONAL defaults to us-east-1, but could be anything if specified with --region=<anyRegion>
# region is grabbed from the auth header, NOT from the body - even with --region flag
# The CLOUDFRONT wacls in aws console are located in us-east-1 but the us-east-1 REGIONAL wacls are not included