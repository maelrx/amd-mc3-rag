#!/usr/bin/env bash
# Anonymous pull check with curl only (no local credentials are used).
#   build/verify_pull.sh docker.io/<user>/mc3:v0      (also ghcr.io/<user>/mc3:v0)
# Exit 0 only if the manifest GET/HEAD answers 200; prints media type, layer count, compressed size.
set -euo pipefail
REFTAG=${1:?usage: $0 REF:TAG}
command -v jq >/dev/null || { echo "jq required" >&2; exit 1; }

NAME=${REFTAG%:*}; TAG=${REFTAG##*:}
[ "$NAME" = "$REFTAG" ] && { echo "REF needs a :TAG" >&2; exit 2; }
case "$NAME" in
  ghcr.io/*)
    HOST=ghcr.io; REPO=${NAME#ghcr.io/}
    TOKURL="https://ghcr.io/token?scope=repository:$REPO:pull" ;;
  *)
    HOST=registry-1.docker.io; REPO=${NAME#docker.io/}; REPO=${REPO#index.docker.io/}
    TOKURL="https://auth.docker.io/token?service=registry.docker.io&scope=repository:$REPO:pull" ;;
esac
ACCEPT='application/vnd.oci.image.index.v1+json, application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.list.v2+json, application/vnd.docker.distribution.manifest.v2+json'
URL="https://$HOST/v2/$REPO/manifests/$TAG"

TOKEN=$(curl -fsS "$TOKURL" | jq -r '.token // .access_token // empty') || { echo "FAIL: no anonymous token from $TOKURL" >&2; exit 1; }
[ -n "$TOKEN" ] || { echo "FAIL: empty token" >&2; exit 1; }
AUTH="Authorization: Bearer $TOKEN"

CODE=$(curl -sS -o /dev/null -I -w '%{http_code}' -H "$AUTH" -H "Accept: $ACCEPT" "$URL")
echo "manifest HEAD: HTTP $CODE"
[ "$CODE" = 200 ] || { echo "FAIL: anonymous manifest HEAD is $CODE (private repo/package or wrong ref)" >&2; exit 1; }

BODY=$(curl -fsS -H "$AUTH" -H "Accept: $ACCEPT" "$URL")
MT=$(jq -r '.mediaType // "?"' <<<"$BODY")
echo "mediaType: $MT"
if jq -e '.manifests' >/dev/null <<<"$BODY"; then
  # index: follow the linux/amd64 entry
  DG=$(jq -r '[.manifests[] | select(.platform.os=="linux" and .platform.architecture=="amd64")][0].digest' <<<"$BODY")
  BODY=$(curl -fsS -H "$AUTH" -H "Accept: $ACCEPT" "https://$HOST/v2/$REPO/manifests/$DG")
fi
N=$(jq '.layers | length' <<<"$BODY")
SZ=$(jq '[.layers[].size] | add' <<<"$BODY")
echo "layers: $N"
echo "compressed total: $SZ bytes ($(awk -v s="$SZ" 'BEGIN{printf "%.2f GiB", s/1073741824}'))"
# one blob HEAD (last layer) proves layer blobs are anonymously readable too
LD=$(jq -r '.layers[-1].digest' <<<"$BODY")
BC=$(curl -sS -o /dev/null -I -w '%{http_code}' -H "$AUTH" "https://$HOST/v2/$REPO/blobs/$LD")
echo "last layer blob HEAD: HTTP $BC"
case "$BC" in 200|307) ;; *) echo "FAIL: blob not readable anonymously" >&2; exit 1;; esac
echo "OK anonymous pull works for $REFTAG"
