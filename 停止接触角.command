#!/bin/zsh
# Reuse the launcher's Python discovery and exact data-directory selection.
set -u
studio_project_dir="${0:A:h}"
exec "$studio_project_dir/启动接触角.command" --stop "$@"
