#!/usr/bin/env bash
#
# Stage and commit the current work as 8 commits spread from 5 days ago to today.
#
#   ./scripts/commit_history.sh            # preview the plan, then ask before committing
#   ./scripts/commit_history.sh --dry-run  # preview only, change nothing
#   ./scripts/commit_history.sh --yes      # commit without asking
#
# Nothing is pushed. Run `git push` yourself when you are happy with `git log`.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DRY_RUN=false
ASSUME_YES=false
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=true ;;
    --yes|-y) ASSUME_YES=true ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; exit 1 ;;
  esac
done

if [ -t 1 ]; then
  BOLD=$'\033[1m'; DIM=$'\033[2m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; BLUE=$'\033[34m'; RED=$'\033[31m'; RESET=$'\033[0m'
else
  BOLD=""; DIM=""; GREEN=""; YELLOW=""; BLUE=""; RED=""; RESET=""
fi

info()  { printf "%s\n" "${BLUE}==>${RESET} ${BOLD}$*${RESET}"; }
ok()    { printf "%s\n" "  ${GREEN}✓${RESET} $*"; }
warn()  { printf "%s\n" "  ${YELLOW}!${RESET} $*"; }
fail()  { printf "%s\n" "${RED}✗ $*${RESET}" >&2; exit 1; }

git rev-parse --is-inside-work-tree >/dev/null 2>&1 || fail "Not inside a git repository."
git symbolic-ref -q HEAD >/dev/null || fail "Detached HEAD. Check out a branch first."
[ -n "$(git config user.name)" ] && [ -n "$(git config user.email)" ] || fail "Set git user.name and user.email first."

# "YYYY-MM-DD HH:MM:SS +ZZZZ" for N days ago at a given time (macOS and GNU date).
commit_date() {
  local days="$1" time="$2" day
  if date -v-1d >/dev/null 2>&1; then
    day="$(date -v-"${days}"d +%Y-%m-%d)"
  else
    day="$(date -d "${days} days ago" +%Y-%m-%d)"
  fi
  echo "${day} ${time} $(date +%z)"
}

# Today's commit uses a few minutes ago so it is never in the future.
today_now() {
  if date -v-1d >/dev/null 2>&1; then
    date -v-5M "+%Y-%m-%d %H:%M:%S %z"
  else
    date -d "5 minutes ago" "+%Y-%m-%d %H:%M:%S %z"
  fi
}

# Each commit: days ago | time | message | pathspecs (space separated, "*" = everything left)
COMMITS=(
  "5|09:42:10|Ignore build artefacts, secrets and local environment files|.gitignore .vscode hardware/certs"
  "4|14:18:37|Add sensor bring-up for MPU6050, ToF, buzzer and LEDs with HTTPS server launcher|hardware manage.py SAMS/asgi.py requirements.txt"
  "3|10:05:52|Add data models for recordings, interaction events and trials|monitoring/models.py monitoring/migrations monitoring/admin.py monitoring/apps.py"
  "3|17:31:04|Add vision, PPE, hazard, fall, distance and fusion inference pipeline|monitoring/services"
  "2|15:47:26|Build review dashboard, control center, alerts, archive and MLOps pages|monitoring/review_views.py monitoring/views.py monitoring/urls.py monitoring/context_processors.py monitoring/templates monitoring/static monitoring/tests.py SAMS/settings.py SAMS/urls.py"
  "1|11:12:48|Train safe/unsafe, vibration, fusion and distance models with results|notebooks/01_* notebooks/02_* notebooks/03_* notebooks/04_* results/fusion results/vibration results/vision/safe_unsafe results/vision/distance scripts"
  "1|18:26:15|Train PPE, fire/smoke, fall and pose detection models with evaluation results|notebooks/05_* notebooks/06_* notebooks/07_* notebooks/08_* notebooks results/vision"
  "0|now|Write project README with setup, API, schema and deployment plan|*"
)

ensure_ignored() {
  local pattern="$1"
  grep -qxF "$pattern" .gitignore 2>/dev/null || printf "%s\n" "$pattern" >> .gitignore
}

prepare_cleanup() {
  ensure_ignored "__pycache__/"
  ensure_ignored "*.pyc"
  ensure_ignored ".DS_Store"
  local tracked_pyc
  tracked_pyc="$(git ls-files -- '*.pyc' '**/__pycache__/**' | wc -l | tr -d ' ')"
  if [ "$tracked_pyc" -gt 0 ]; then
    git ls-files -z -- '*.pyc' '**/__pycache__/**' | xargs -0 git rm --cached --quiet --
    ok "Stopped tracking ${tracked_pyc} compiled .pyc files"
  fi
}

stage_paths() {
  local specs="$1" spec
  if [ "$specs" = "*" ]; then
    git add -A
    return
  fi
  for spec in $specs; do
    git add -A -- "$spec" 2>/dev/null || true
  done
}

preview() {
  info "Commit plan (nothing is pushed)"
  local n=1 entry days time msg specs when
  for entry in "${COMMITS[@]}"; do
    IFS="|" read -r days time msg specs <<<"$entry"
    if [ "$time" = "now" ]; then when="$(today_now)"; else when="$(commit_date "$days" "$time")"; fi
    printf "  %s%d.%s %s%s%s\n     %s%s%s\n" "$BOLD" "$n" "$RESET" "$DIM" "$when" "$RESET" "$GREEN" "$msg" "$RESET"
    n=$((n + 1))
  done
  echo
  info "Files with changes right now"
  git status --short | sed 's/^/  /' | head -40
  local total
  total="$(git status --short | wc -l | tr -d ' ')"
  [ "$total" -gt 40 ] && printf "  %s... and %d more%s\n" "$DIM" "$((total - 40))" "$RESET"
  echo
}

preview

if $DRY_RUN; then
  info "Dry run only. Nothing was changed."
  exit 0
fi

if ! $ASSUME_YES; then
  printf "%s" "${BOLD}Create these 8 commits? [y/N] ${RESET}"
  read -r reply
  case "$reply" in
    y|Y|yes|YES) ;;
    *) echo "Cancelled."; exit 0 ;;
  esac
fi

info "Cleaning up tracked artefacts"
prepare_cleanup

made=0
n=1
for entry in "${COMMITS[@]}"; do
  IFS="|" read -r days time msg specs <<<"$entry"
  if [ "$time" = "now" ]; then when="$(today_now)"; else when="$(commit_date "$days" "$time")"; fi

  stage_paths "$specs"

  if git diff --cached --quiet; then
    warn "Commit ${n} skipped (nothing to commit): ${msg}"
  else
    files="$(git diff --cached --name-only | wc -l | tr -d ' ')"
    tree="$(git write-tree)"
    parent="$(git rev-parse HEAD)"
    sha="$(printf "%s\n" "$msg" | GIT_AUTHOR_DATE="$when" GIT_COMMITTER_DATE="$when" git commit-tree "$tree" -p "$parent")"
    git update-ref -m "commit: $msg" HEAD "$sha" "$parent"
    ok "Commit ${n}  ${DIM}${when}${RESET}  ${files} files  ${msg}"
    made=$((made + 1))
  fi
  n=$((n + 1))
done

echo
info "Done: ${made} commits created"
git log --format="  %C(yellow)%h%Creset %C(dim)%ad%Creset %s" --date=format:"%a %d %b %H:%M" -n "$made"
echo
printf "%s\n" "  Push when ready: ${BOLD}git push origin $(git rev-parse --abbrev-ref HEAD)${RESET}"
