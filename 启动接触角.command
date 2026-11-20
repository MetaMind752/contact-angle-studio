#!/bin/zsh
# Open the browser after the local background service is ready.
# :A resolves a shortcut/symlink to the source so moving the folder and spaces work.
set -u
studio_project_dir="${0:A:h}"
cd "$studio_project_dir" || exit 1
studio_python=""
if [[ -n "${CONTACT_ANGLE_PYTHON:-}" ]]; then
  studio_python="$CONTACT_ANGLE_PYTHON"
elif [[ -x "$studio_project_dir/.venv/bin/python" ]]; then
  studio_python="$studio_project_dir/.venv/bin/python"
else
  for studio_candidate in python3.13 python3.12 python3.11 python3.14 python3; do
    studio_candidate_path="$(command -v "$studio_candidate" 2>/dev/null || true)"
    if [[ -n "$studio_candidate_path" ]] && "$studio_candidate_path" -c 'import sys; raise SystemExit(sys.version_info < (3,11))' 2>/dev/null; then
      studio_python="$studio_candidate_path"
      break
    fi
  done
fi
if [[ -z "$studio_python" ]]; then
  print "需要 Python 3.11 或更高版本。安装后重新双击此文件。"
  print "若已有可用环境，可设置 CONTACT_ANGLE_PYTHON 指向 Python 解释器。"
  if [[ -t 0 ]]; then read -r "?按 Enter 关闭窗口……"; fi
  exit 1
fi
studio_mode=(--background)
for studio_argument in "$@"; do
  case "$studio_argument" in
    --background|--foreground|--stop|--status|--check) studio_mode=(); break ;;
  esac
done
"$studio_python" "$studio_project_dir/scripts/launch_studio.py" "${studio_mode[@]}" "$@"
studio_status=$?
if [[ "$studio_status" -ne 0 && -t 0 ]]; then
  read -r "?启动未完成。请查看上方信息，按 Enter 关闭窗口……"
fi
exit "$studio_status"
