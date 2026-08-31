# 2. Setup

All placeholders (`<USER>`, `<GFX1250_NODE>`, `<CK_IMAGE_TAG>`, ...) must be filled in from your own
environment. See `config.example.env`.

## 2.1 Node access

The GPU node is managed by your lab's reservation system: you need a live reservation and a registered SSH
key before the host will accept a connection. The login banner is a large block of ASCII art — ignore it.

```bash
cp config.example.env config.env     # then edit config.env
source config.env

ssh "$GPU_USER@$GFX1250_NODE"
ls /dev/kfd /dev/dri/card*           # expect card0 .. card7
```

Work under `$HOME`. Scratch mounts such as `/data/work` are frequently read-only for normal users.

## 2.2 Container

```bash
export CK_CONTAINER="ck-$USER"
export CK_WS="$HOME/ck_ws"
export CK_IMAGE="<CK_IMAGE_TAG>"     # confirm the current tag with your team; tags rot
mkdir -p "$CK_WS"

docker run -d --name "$CK_CONTAINER" \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --ipc=host --shm-size=16G \
  --network host \
  -v "$CK_WS":/ck -w /ck \
  "$CK_IMAGE" sleep infinity

docker exec -it "$CK_CONTAINER" bash
```

Check the GPU from inside — note `rocminfo` is **not** on `PATH`:

```bash
/opt/rocm/bin/rocminfo | grep -i gfx1250
```

### Environment facts worth knowing before you lose an hour

| Symptom | Cause | Fix |
|---|---|---|
| `docker run` fails on `--group-add render` | there is no `render` group on this host | drop it; `video` is sufficient |
| results vanish after the container is removed | you wrote outside the bind mount | always write under `/ck/...` (host `$CK_WS`) |
| 8 cards visible on the host, only 1 usable in the container | device visibility inside the container | check, then use `--devices 0` |

## 2.3 Build CK

Shallow-clone the **standalone** repo (~261 MB, ~20 s). The `rocm-libraries` monorepo blobless clone is
extremely slow — avoid it for this work.

```bash
cd /ck && git clone --depth 1 https://github.com/ROCm/composable_kernel.git ck
git config --global --add safe.directory /ck/ck        # bind-mount UID mismatch

cd /ck/ck && mkdir -p build && cd build
cmake -DGPU_TARGETS=gfx1250 \
      -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
      -DCMAKE_PREFIX_PATH=/opt/rocm ..
make -j$(nproc)                                        # the image has no ninja
```

`-DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc` is **not optional**. Without it, CMake's supported-target list
comes out empty and every target — including `gfx1250` — is reported as "unknown".

## 2.4 Running long jobs without losing them

Use `docker exec -d` plus a **done-marker file**. Do not poll with `pgrep`.

```bash
docker exec -d "$CK_CONTAINER" bash -lc '
  cd /ck/ck/build && make -j$(nproc) > /ck/build.log 2>&1; echo $? > /ck/build.done'

# poll for the marker
while ! docker exec "$CK_CONTAINER" test -f /ck/build.done; do sleep 60; done
docker exec "$CK_CONTAINER" cat /ck/build.done       # 0 = success
```

Why: a `nohup` launched inside `docker exec` dies when the SSH session ends **and** leaves a zombie process
that `pgrep -f ...` still matches. A `pgrep` polling loop therefore waits forever on a job that is already
dead.

## 2.5 Stage the scripts

Copy the three scripts from this repo into the container workspace:

```bash
scp scripts/*.py "$GPU_USER@$GFX1250_NODE:~/ck_ws/"     # $CK_WS is bind-mounted to /ck
```
