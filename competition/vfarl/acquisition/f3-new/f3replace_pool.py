"""F3-replace 40-task pool definition (DVC 9 + Conan 19 + Cloudpickle 8 + Typeguard 1 + Dask 3)."""
from pathlib import Path

# key, instance_id, swesmith_repo, canonical_repo, upstream_short, symbol, contract
TASKS = [
    # ---- DVC stage/cache 9 ----
    ("f3r-dvc-01", "iterative__dvc.1d6ea681.func_pm_class_rm_funcs__f24e89yi", "swesmith/iterative__dvc.1d6ea681", "iterative/dvc", "1d6ea681", "StageCache.save", "cache_write_key_scope_and_hit_equivalence"),
    ("f3r-dvc-02", "iterative__dvc.1d6ea681.func_pm_ctrl_shuffle__ypn4tc8h", "swesmith/iterative__dvc.1d6ea681", "iterative/dvc", "1d6ea681", "StageCache.create", "cache_entry_construction_and_key_isolation"),
    ("f3r-dvc-03", "iterative__dvc.1d6ea681.func_pm_remove_cond__eg99hxp1", "swesmith/iterative__dvc.1d6ea681", "iterative/dvc", "1d6ea681", "StageCache.is_valid", "cache_validity_scope_and_stale_invalidation"),
    ("f3r-dvc-04", "iterative__dvc.1d6ea681.lm_rewrite__h3lq5aia", "swesmith/iterative__dvc.1d6ea681", "iterative/dvc", "1d6ea681", "StageCache.restore", "cache_restore_hit_and_recompute_after_clear"),
    ("f3r-dvc-05", "iterative__dvc.1d6ea681.lm_rewrite__vu3wpb3g", "swesmith/iterative__dvc.1d6ea681", "iterative/dvc", "1d6ea681", "StageCache.save", "cache_overwrite_equivalence_and_isolation"),
    ("f3r-dvc-06", "iterative__dvc.1d6ea681.pr_7680", "swesmith/iterative__dvc.1d6ea681", "iterative/dvc", "1d6ea681", "StageCache", "stage_cache_roundtrip_and_invalidation"),
    ("f3r-dvc-07", "iterative__dvc.1d6ea681.pr_8405", "swesmith/iterative__dvc.1d6ea681", "iterative/dvc", "1d6ea681", "StageCache.transfer", "cache_transfer_scope_and_key_equivalence"),
    ("f3r-dvc-08", "iterative__dvc.1d6ea681.combine_module__y0nr2n7s", "swesmith/iterative__dvc.1d6ea681", "iterative/dvc", "1d6ea681", "StageCache", "cache_serialization_scope_and_reload_hit"),
    ("f3r-dvc-09", "iterative__dvc.1d6ea681.combine_file__viwsocw1", "swesmith/iterative__dvc.1d6ea681", "iterative/dvc", "1d6ea681", "IndexView.data_tree", "index_cache_build_and_gc_invalidation"),
    # ---- Conan cache 19 ----
    ("f3r-conan-01", "conan-io__conan.86f29e13.pr_14164", "swesmith/conan-io__conan.86f29e13", "conan-io/conan", "86f29e13", "ClientCache", "cache_path_scope_and_layout_equivalence"),
    ("f3r-conan-02", "conan-io__conan.86f29e13.pr_12930", "swesmith/conan-io__conan.86f29e13", "conan-io/conan", "86f29e13", "CacheDatabase", "cache_db_store_load_and_clear"),
    ("f3r-conan-03", "conan-io__conan.86f29e13.pr_14716", "swesmith/conan-io__conan.86f29e13", "conan-io/conan", "86f29e13", "CachingFileDownloader", "download_cache_hit_and_invalidation"),
    ("f3r-conan-04", "conan-io__conan.86f29e13.pr_13050", "swesmith/conan-io__conan.86f29e13", "conan-io/conan", "86f29e13", "CacheAPI", "cache_api_list_and_path_isolation"),
    ("f3r-conan-05", "conan-io__conan.86f29e13.pr_12554", "swesmith/conan-io__conan.86f29e13", "conan-io/conan", "86f29e13", "CacheAPI.clean", "cache_clean_scope_and_recompute"),
    ("f3r-conan-06", "conan-io__conan.86f29e13.pr_14938", "swesmith/conan-io__conan.86f29e13", "conan-io/conan", "86f29e13", "ClientCache", "cache_recipe_store_and_key_isolation"),
    ("f3r-conan-07", "conan-io__conan.86f29e13.pr_13561", "swesmith/conan-io__conan.86f29e13", "conan-io/conan", "86f29e13", "Proxy", "proxy_binary_cache_lookup_and_refresh"),
    ("f3r-conan-08", "conan-io__conan.86f29e13.pr_14231", "swesmith/conan-io__conan.86f29e13", "conan-io/conan", "86f29e13", "CacheDatabase.table", "cache_table_upsert_and_evict"),
    ("f3r-conan-09", "conan-io__conan.86f29e13.pr_15029", "swesmith/conan-io__conan.86f29e13", "conan-io/conan", "86f29e13", "CacheTable", "cache_table_row_scope_and_reload"),
    ("f3r-conan-10", "conan-io__conan.86f29e13.pr_14555", "swesmith/conan-io__conan.86f29e13", "conan-io/conan", "86f29e13", "CacheAPI", "cache_save_load_and_remove_invalidation"),
    ("f3r-conan-11", "conan-io__conan.86f29e13.pr_13581", "swesmith/conan-io__conan.86f29e13", "conan-io/conan", "86f29e13", "CacheAPI", "cache_path_profile_scope_and_equivalence"),
    ("f3r-conan-12", "conan-io__conan.86f29e13.pr_13626", "swesmith/conan-io__conan.86f29e13", "conan-io/conan", "86f29e13", "ClientCache", "cache_remote_state_and_clear"),
    ("f3r-conan-13", "conan-io__conan.86f29e13.pr_14923", "swesmith/conan-io__conan.86f29e13", "conan-io/conan", "86f29e13", "CacheAPI", "cache_query_hit_and_miss_isolation"),
    ("f3r-conan-14", "conan-io__conan.86f29e13.pr_13811", "swesmith/conan-io__conan.86f29e13", "conan-io/conan", "86f29e13", "CachingFileDownloader", "downloader_cache_reuse_and_expire"),
    ("f3r-conan-15", "conan-io__conan.86f29e13.pr_14342", "swesmith/conan-io__conan.86f29e13", "conan-io/conan", "86f29e13", "CachingFileDownloader", "downloader_cached_copy_and_recompute"),
    ("f3r-conan-16", "conan-io__conan.86f29e13.pr_14020", "swesmith/conan-io__conan.86f29e13", "conan-io/conan", "86f29e13", "DownloadCache", "download_cache_store_load_and_invalidate"),
    ("f3r-conan-17", "conan-io__conan.86f29e13.pr_14185", "swesmith/conan-io__conan.86f29e13", "conan-io/conan", "86f29e13", "DownloadCache", "download_cache_lock_scope_and_clear"),
    ("f3r-conan-18", "conan-io__conan.86f29e13.pr_14082", "swesmith/conan-io__conan.86f29e13", "conan-io/conan", "86f29e13", "CacheRemove", "cache_remove_scope_and_post_remove_miss"),
    ("f3r-conan-19", "conan-io__conan.86f29e13.func_pm_class_rm_funcs__dc4gm8g0", "swesmith/conan-io__conan.86f29e13", "conan-io/conan", "86f29e13", "ClientCache", "cache_method_presence_for_store_and_load"),
    # ---- Cloudpickle memo 8 ----
    ("f3r-cp-01", "cloudpipe__cloudpickle.6220b0ce.func_basic__w5ot5zcx", "swesmith/cloudpipe__cloudpickle.6220b0ce", "cloudpipe/cloudpickle", "6220b0ce", "PickleByValueRegistry", "registry_register_lookup_and_unregister_invalidation"),
    ("f3r-cp-02", "cloudpipe__cloudpickle.6220b0ce.func_basic__3brtui7n", "swesmith/cloudpipe__cloudpickle.6220b0ce", "cloudpipe/cloudpickle", "6220b0ce", "_class_getstate", "abc_cache_exclusion_and_registry_roundtrip"),
    ("f3r-cp-03", "cloudpipe__cloudpickle.6220b0ce.lm_rewrite__89r7c7sh", "swesmith/cloudpipe__cloudpickle.6220b0ce", "cloudpipe/cloudpickle", "6220b0ce", "_class_setstate", "memo_identity_scope_and_state_restore"),
    ("f3r-cp-04", "cloudpipe__cloudpickle.6220b0ce.lm_rewrite__6cyqq4th", "swesmith/cloudpipe__cloudpickle.6220b0ce", "cloudpipe/cloudpickle", "6220b0ce", "_code_reduce", "deterministic_memo_key_scope_and_recompute"),
    ("f3r-cp-05", "cloudpipe__cloudpickle.6220b0ce.func_pm_ctrl_invert_if__02r4vlsa", "swesmith/cloudpipe__cloudpickle.6220b0ce", "cloudpipe/cloudpickle", "6220b0ce", "_class_getstate", "abc_registry_branch_scope_and_cache_isolation"),
    ("f3r-cp-06", "cloudpipe__cloudpickle.6220b0ce.lm_rewrite__nycdsg8e", "swesmith/cloudpipe__cloudpickle.6220b0ce", "cloudpipe/cloudpickle", "6220b0ce", "_function_getstate", "function_state_capture_and_slot_restore"),
    ("f3r-cp-07", "cloudpipe__cloudpickle.6220b0ce.func_basic__2oqo0xtt", "swesmith/cloudpipe__cloudpickle.6220b0ce", "cloudpipe/cloudpickle", "6220b0ce", "_extract_class_dict", "class_dict_memo_scope_and_inherited_isolation"),
    ("f3r-cp-08", "cloudpipe__cloudpickle.6220b0ce.lm_rewrite__vcy7tnla", "swesmith/cloudpipe__cloudpickle.6220b0ce", "cloudpipe/cloudpickle", "6220b0ce", "_class_getstate", "class_state_extraction_and_cache_exclusion"),
    # ---- Typeguard _memo 1 ----
    ("f3r-tg-01", "agronholm__typeguard.b6a7e438.func_basic__4zwo7ul8", "swesmith/agronholm__typeguard.b6a7e438", "agronholm/typeguard", "b6a7e438", "TransformMemo", "memo_alias_namespace_binding_and_scope_isolation"),
    # ---- Dask 3 ----
    ("f3r-dask-01", "dask__dask.5f61e423.lm_rewrite__0jf8a01k", "swesmith/dask__dask.5f61e423", "dask/dask", "5f61e423", "get_async-cache", "scheduler_cache_set_get_and_recompute"),
    ("f3r-dask-02", "dask__dask.5f61e423.pr_9739", "swesmith/dask__dask.5f61e423", "dask/dask", "5f61e423", "CacheProfiler.clear", "profiler_result_cache_and_clear_invalidation"),
    ("f3r-dask-03", "dask__dask.5f61e423.lm_rewrite__kex0qbbb", "swesmith/dask__dask.5f61e423", "dask/dask", "5f61e423", "inline_functions-keysubs", "substitution_cache_single_use_and_output_protection"),
]

# Environmentally ineligible in repo-nested harnesses: these rows require
# test_cache_clean_all to pass, but that test asserts `create pkg1`
# (revision_mode='scm') FAILS while any git HEAD is discoverable above the
# test tmp dir. Demonstrated by f3r-conan-19 (26/27 pass, only clean_all
# fails). Kept as evidence, excluded from replay/admission.
SUPERSEDED = {'f3r-conan-04', 'f3r-conan-06', 'f3r-conan-07', 'f3r-conan-11', 'f3r-conan-19',
              'f3r-cp-01', 'f3r-cp-02', 'f3r-cp-03', 'f3r-cp-04', 'f3r-cp-05', 'f3r-cp-06',
              'f3r-cp-07', 'f3r-cp-08',
              'f3r-tg-01', 'f3r-tg-02', 'f3r-tg-03', 'f3r-tg-04',
              'f3r-dvc-17', 'f3r-conan-25'}
# f3r-dvc-17: repaired 104/105; the single failure is test_stage_run_ignore_sigint,
# order-dependent signal-state pollution (passes standalone and in-file, fails only
# in the full 8-file sandbox run across 3 attempts).
# f3r-conan-25: deterministic fixture-as-test artifact (install_test.py `client`
# fixture collected as a testcase, fails on both phases independent of repair).
# f3r-tg-*: repaired JUnit has 0 failures, but P2P contains
# test_raises_for_non_member[generator] whose xfail-marked parametrized siblings
# skip; the strict matcher prefix-matches them (same rule as deepdiff
# test_list_difference) and rejects. Demonstrated by all four runs.
# f3r-cp-*: the hermetic pytest.ini written by the harness forces fd-capture
# (EncodedFile) stdio, which cloudpickle cannot pickle. The 3 file-handle tests
# in every cloudpickle P2P therefore fail on ANY tree in-sandbox (reproduced:
# ini present -> EncodedFile -> TypeError; no ini -> real stdio -> pass).
# Demonstrated by f3r-cp-01 (repaired 269/272, only the 3 stdio tests fail).
# f3r-conan-07: repaired 13/14; the single failure is the duplicate copy of
# test_multi_remote_update_resolution (harness double-collects the file and the
# 'core:update_policy' deprecation warning is warn-once-per-process).

# Replacement spares (same mirrors/upstreams, no fragile nodes).
SPARE_TASKS = [
    ("f3r-conan-20", "conan-io__conan.86f29e13.pr_14846", "swesmith/conan-io__conan.86f29e13", "conan-io/conan", "86f29e13", "PyRequireLoader.cache", "pyrequire_resolution_cache_scope"),
    ("f3r-conan-21", "conan-io__conan.86f29e13.pr_14478", "swesmith/conan-io__conan.86f29e13", "conan-io/conan", "86f29e13", "CacheRemove.norev", "cache_remove_without_revision_scope"),
    ("f3r-conan-22", "conan-io__conan.86f29e13.pr_14261", "swesmith/conan-io__conan.86f29e13", "conan-io/conan", "86f29e13", "DownloadAPI.sources", "download_sources_scope_and_cache_store"),
    ("f3r-conan-23", "conan-io__conan.86f29e13.pr_13432", "swesmith/conan-io__conan.86f29e13", "conan-io/conan", "86f29e13", "RemoteManager.integrity", "remote_payload_integrity_and_reject"),
    ("f3r-conan-24", "conan-io__conan.86f29e13.pr_14526", "swesmith/conan-io__conan.86f29e13", "conan-io/conan", "86f29e13", "DownloadAPI.timestamps", "download_timestamp_adoption_and_cache_update"),
    ("f3r-dvc-10", "iterative__dvc.1d6ea681.combine_file__n3kqhhy3", "swesmith/iterative__dvc.1d6ea681", "iterative/dvc", "1d6ea681", "StageCache._can_hash+_load_cache+save", "combined_gate_order_and_deterministic_reuse"),
    ("f3r-dvc-11", "iterative__dvc.1d6ea681.combine_file__ela813xu", "swesmith/iterative__dvc.1d6ea681", "iterative/dvc", "1d6ea681", "Output.__init__", "output_state_init_and_hash_scope"),
    ("f3r-dvc-12", "iterative__dvc.1d6ea681.lm_rewrite__4h34bsm4", "swesmith/iterative__dvc.1d6ea681", "iterative/dvc", "1d6ea681", "Output.dumpd", "output_dumpd_shape_and_version_scope"),
    ("f3r-dvc-13", "iterative__dvc.1d6ea681.pr_9123", "swesmith/iterative__dvc.1d6ea681", "iterative/dvc", "1d6ea681", "StageCache", "stage_cache_filesystem_scope_and_reuse"),
    ("f3r-dvc-14", "iterative__dvc.1d6ea681.combine_module__8f6tx7o0", "swesmith/iterative__dvc.1d6ea681", "iterative/dvc", "1d6ea681", "StageCache._can_hash", "combined_gate_and_utils_scope"),
    ("f3r-tg-02", "agronholm__typeguard.b6a7e438.lm_rewrite__4zrtc4ic", "swesmith/agronholm__typeguard.b6a7e438", "agronholm/typeguard", "b6a7e438", "TransformMemo.ignored_names", "memo_ignored_names_scope_for_assignments"),
    ("f3r-tg-03", "agronholm__typeguard.b6a7e438.func_basic__2o971yci", "swesmith/agronholm__typeguard.b6a7e438", "agronholm/typeguard", "b6a7e438", "TransformMemo.imported_names", "memo_imported_names_module_binding"),
    ("f3r-tg-04", "agronholm__typeguard.b6a7e438.func_basic__6yo0fcfo", "swesmith/agronholm__typeguard.b6a7e438", "agronholm/typeguard", "b6a7e438", "TransformMemo.module_scope", "memo_module_scope_identity_for_typechecking"),
    ("f3r-dvc-15", "iterative__dvc.1d6ea681.lm_rewrite__s9km3gww", "swesmith/iterative__dvc.1d6ea681", "iterative/dvc", "1d6ea681", "Output._compute_meta_hash_info_from_files", "output_hash_meta_from_files_scope"),
    ("f3r-dvc-16", "iterative__dvc.1d6ea681.pr_8078", "swesmith/iterative__dvc.1d6ea681", "iterative/dvc", "1d6ea681", "ExperimentsRemove.queue", "experiment_queue_removal_scope"),
    ("f3r-dvc-17", "iterative__dvc.1d6ea681.combine_module__g0bfaibs", "swesmith/iterative__dvc.1d6ea681", "iterative/dvc", "1d6ea681", "StageCache.load", "stage_cache_load_scope_and_reuse"),
    ("f3r-conan-25", "conan-io__conan.86f29e13.combine_module__600np8st", "swesmith/conan-io__conan.86f29e13", "conan-io/conan", "86f29e13", "CacheAPI.removed_methods", "cache_api_method_presence_and_download_scope"),
    ("f3r-conan-27", "conan-io__conan.86f29e13.pr_13458", "swesmith/conan-io__conan.86f29e13", "conan-io/conan", "86f29e13", "conans.tools.download", "download_failure_retry_scope"),
    ("f3r-conan-28", "conan-io__conan.86f29e13.pr_15013", "swesmith/conan-io__conan.86f29e13", "conan-io/conan", "86f29e13", "UploadAPI.backup_sources", "backup_sources_upload_scope"),
]


def ACTIVE_TASKS():
    return ([t for t in TASKS if t[0] not in SUPERSEDED]
            + [t for t in SPARE_TASKS if t[0] not in SUPERSEDED])
