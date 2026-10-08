# P-CRA-U development preflight amendment 01

Before any canary optimizer step, the first feature-cache finalization attempt
extracted 61/61 canary feature pairs but stopped at JSON-schema validation because
the deliberately isolated RoboRefer venv does not install `jsonschema`.

The amendment changes only the validation wrapper: when `jsonschema` is absent in
the RoboRefer venv, the index is piped to `/usr/bin/python3` for Draft 2020-12
validation, matching the already qualified WP3 pattern. It does not alter model
loading, feature tensors, pooling, labels, optimizer, dataset or baseline.

The 61 content-addressed tensors are immutable and must be hash/tensor-verified on
resume. No feature is silently overwritten and no optimizer step occurred before
this amendment. The entire preflight and execution lock must be regenerated before
canary training.
