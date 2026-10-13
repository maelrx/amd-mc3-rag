# MC3 image build (no Docker)

The image = ROCm base (never downloaded, reused by digest) + layers appended with `crane append` on a Colab CPU VM:
`00-deps` (cp314 wheels from `app/requirements.txt`), `10-model-meta`, `11..14-shard`, `20-app`. `Dockerfile` is a reference only.
`checks.py` runs inside the image (`deps`, `ocr`, `weights`); `colab_build.py` does everything on the VM (phases tools, wheels, weights, tars, push, verify, app; idempotent via `/content/mc3/state/*.done`, `MC3_FORCE=1` reruns); `append_app.sh` pushes only a new app layer (seconds); `verify_pull.sh` / `udocker_smoke.sh` check the result.
Tags: `:tmp` (chain), `:heavy-noapp` (deps+model), `:heavy` and `:$MC3_TAG` (+app). CMD `python3 /app/server.py`, WORKDIR `/app`, ENV in `colab_build.py:CONTAINER_ENV`.

Colab commands (session is a Jupyter kernel; `exec` default timeout is 30 s, so always pass `--timeout`):
```
python3 build/colab_build.py --dry-run                                  # local, prints the plan
tar cf /tmp/src.tar app build && colab new -s mc3-build
echo 'import os; os.makedirs("/content/mc3/secrets", exist_ok=True)' > /tmp/mk.py && colab exec -s mc3-build -f /tmp/mk.py
colab upload /tmp/src.tar /content/mc3/src.tar
colab upload ~/.config/mc3/dockerhub.user  /content/mc3/secrets/dockerhub.user
colab upload ~/.config/mc3/dockerhub.token /content/mc3/secrets/dockerhub.token
colab exec -s mc3-build --timeout 7200 --env MC3_REF=docker.io/<user>/mc3 --env MC3_TAG=v0 -f build/colab_build.py
colab exec -s mc3-build --timeout 7200 --env MC3_PHASE=app --env MC3_TAG=v1 --env MC3_REF=docker.io/<user>/mc3 -f build/colab_build.py   # later: new app layer (upload a fresh src.tar first)
colab stop -s mc3-build
```
Run phases one exec at a time with `--env MC3_PHASE=tools,wheels` etc. to keep each call short; re-running `push` resumes (per-layer markers + layer-count check on `:tmp`).
The script runs `main()` when `__name__ == "__main__"` or under IPython; if a `colab exec` shows no output at all, exec the file and then a second one-line file calling `main()`.

Credentials: a Docker Hub Personal Access Token (Read & Write) in `~/.config/mc3/dockerhub.{user,token}` (chmod 600; `ghcr.{user,token}` for ghcr.io). They are uploaded as files, read only inside `registry_login()` and piped to `crane auth login --password-stdin`: never in argv, env, logs or the image. Delete the VM (`colab stop`) when done; the token lives only in `/content/mc3/secrets` there.
After the push: `build/verify_pull.sh docker.io/<user>/mc3:v0` (anonymous, curl only); optional `build/udocker_smoke.sh` on a fresh Colab VM (needs ~90 GB free).
