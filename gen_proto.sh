#!/usr/bin/env bash
# Thin wrapper: the schemas and their generator live in the proto/ submodule.
exec bash "$(dirname "${BASH_SOURCE[0]}")/proto/gen_proto.sh" "$@"
