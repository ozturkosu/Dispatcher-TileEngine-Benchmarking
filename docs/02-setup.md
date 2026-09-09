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

## 2.3 Locate the prebuilt CK-Tile

There is **no CK-Tile to build**. The `ck-wmma-instances` image already ships the CK-Tile source with the
gfx1250 WMMA instances in place, and this is **CK-Tile / TileEngine** — the benchmark driver compiles the
kernel variants it needs on demand (its Phase 1 hands each config to the dispatcher, which codegens +
compiles a `.so`). So you only need to locate the checkout and point everything at it via `$CK_SRC`:

```bash
export CK_SRC=$(find / -name gemm_full_benchmark.py 2>/dev/null | head -1 \
  | sed 's:/tile_engine/ops/gemm/gemm_full_benchmark.py::')
echo "CK_SRC=$CK_SRC"                                     # e.g. /composable_kernel
ls "$CK_SRC"/tile_engine/ops/gemm/configs/ | grep 1250   # confirm the gfx1250 configs exist
```

Use `cd "$CK_SRC"` and `--ck "$CK_SRC"` everywhere below. Don't hardcode a path — the image layout can move.

> **Fallback only — if `CK_SRC` came back empty** (the image has no CK-Tile source). Shallow-clone the
> standalone repo and point `CK_SRC` at it; the driver still compiles the kernels itself, so you do **not**
> run a full `make`. Avoid the `rocm-libraries` monorepo (blobless clone is painfully slow).
>
> ```bash
> cd /ck && git clone --depth 1 https://github.com/ROCm/composable_kernel.git ck
> git config --global --add safe.directory /ck/ck     # bind-mount UID mismatch
> export CK_SRC=/ck/ck
> ```

## 2.4 Running long jobs without losing them

Use `docker exec -d` plus a **done-marker file**. Do not poll with `pgrep`.

```bash
docker exec -d "$CK_CONTAINER" bash -lc "
  cd $CK_SRC && python3 /ck/batch_mi400_pipeline.py \
    --run-csv /ck/data/run_8611.csv --out /ck/work/run8611 --ck $CK_SRC \
    --devices 0 --label run8611 > /ck/work/run8611.log 2>&1"

# poll for the marker the pipeline writes last
while ! docker exec "$CK_CONTAINER" test -f /ck/work/run8611/file.done; do sleep 60; done
docker exec "$CK_CONTAINER" cat /ck/work/run8611/file.done
```

Why: a `nohup` launched inside `docker exec` dies when the SSH session ends **and** leaves a zombie process
that `pgrep -f ...` still matches. A `pgrep` polling loop therefore waits forever on a job that is already
dead.

## 2.5 Stage the scripts

Copy the three scripts from this repo into the container workspace:

```bash
scp scripts/*.py "$GPU_USER@$GFX1250_NODE:~/ck_ws/"     # $CK_WS is bind-mounted to /ck
```
