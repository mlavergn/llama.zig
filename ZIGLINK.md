# ZIGLINK.md

Remote gate machines, reached over the ziglink protocol through `zupdater`.

## Tools

| Binary | Path | Role |
|---|---|---|
| `zupdater` | `../zigupdater/zig-out/bin/zupdater` | Host; paired with every gate machine |
| `ziglink` | `../ziglink/zig-out/bin/ziglink` | Library's own CLI; not paired here, not used |

- `zupdater` is built on `../ziglink`: same wire, same `tty`/`push`.
- Data dir: `zupdater`'s, `~/.<org>/updater` for the organization its branding names (`peers.zon`, `bin/`, `cache/`).

## Agents

| Agent | Machine | Target | Gates |
|---|---|---|---|
| `spark-dbff` | GB10, Ubuntu 24.04 | `aarch64-linux`, CUDA 13.0 `sm_121a`, Vulkan | Linux arm64, CUDA, Vulkan, everything driven twice |
| `DESKTOP-BR7J5S9` | i7-4980HQ, Windows 11 | `x86_64-windows-gnu`, `x86_64_v3` | x86 AVX2, Windows |
| `JWHLPTOP` | i5-1135G7, Windows 11 | `x86_64_v4`, `tigerlake` | x86 AVX-512 |

- `zupdater peers`: pairings, last seen.
- An agent runs only while its app runs (window open on Windows).

## Commands

| Command | Use |
|---|---|
| `zupdater peers` | Check that agents are online |
| `zupdater tty <agent>` | Shell there (bash / PowerShell); leave with Return `~` `.` |
| `zupdater push <agent> <exe> --name <n> --run -- <args>` | Install a binary in the agent's `bin/`, run it, stream its output back |
| `scripts/win-run <exe> <name> [args]` | `push --run` to `X86_AGENT` (default `DESKTOP-BR7J5S9`), ConPTY output cleaned |

## Moving the tree

Never through GitHub. Everything goes over the link.

| What | How |
|---|---|
| Committed history | `git bundle create x.bundle <base>..<head>` here; carried over; `git fetch x.bundle` + fast-forward there |
| Uncommitted work | `git diff --binary <head>` (add `--cached` for staged work); carried over; `git apply` there |
| Sibling repos (`../zigjinja`) | Same bundle method, fast-forward only |
| Carrier | A scratch C program that `.incbin`s a tar and writes it to `argv[1]`; built for the agent's target, `push --run`; SHA-256 of the tar checked on arrival (`push` takes executables only) |
| Fixes between rounds | Re-carry every changed **and untracked** file whole; compare an MD5 over the file list on both sides before running |
| Before applying | Remote tree clean; `git bundle verify`; patch checked to apply byte-exactly to a `git archive` of the base |
| Windows gates | Not carried: cross-built here, the `.exe` pushed and run; comparison happens there, only the verdict returns |

- GB10 checkout: `~/src/mlavergn/llama.zig`; `llama.cpp.zmake` symlinked to the sibling `~/src/mlavergn/llama.cpp.zmake`.
- GB10 scratch: `~/pickup/`.
- Windows output: `%TEMP%\llamazig`; models on the Desktop (`X86_MODEL` is relative to `%USERPROFILE%`).
- Remote checkout dirty: stop, report; never discard.
- No commits made on any side as part of a pickup.

## Driving `tty` from a script

- Needs a terminal: wrap in Python `pty.fork()`.
- Wait for the prompt (`$ `, `PS ...> `) before sending.
- `PAGER=cat GIT_PAGER=cat`.
- Strip OSC 133 marks (`BEL`- and `ESC \`-terminated).
- ConPTY: `ESC[<row>;1H` is a line break; lines wrap at 80 columns; keep verdict lines short.
- Long jobs: `( nohup ... > ~/pickup/x.log 2>&1 & )`, then poll the log every 60 s (one short session per poll).
- Completion token computed at run time (`echo FIN$((40+2))`), never a word in the sent command.
- Never `pgrep -f name` in a wait loop: matches itself.

## Pitfalls

| Symptom | Cause |
|---|---|
| `waiting for <agent>` forever | Another `zupdater` (an open console, a long push) holds the agents; one session at a time |
| `could not open a terminal` (Windows) | Application Control blocking that exe's hash; rebuild or run it from `bin/` in a `tty` |
| `wsl` prompts, Return answers it | No distro on `DESKTOP-BR7J5S9`; never run `wsl` there |
| gdb cannot attach (GB10) | Yama scope 1; `LD_PRELOAD` a `prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY)` shim |
| `push` refuses a file (Windows) | PE executables only; fetch data and models on the machine |
| A gate fails on the agent only | Free-memory figures are read live (GB10): mask them; upstream's queued logger races its aborts: compare stderr and log streams apart |

## After a pickup

- No `zupdater`, server or wait loop left running on the Mac.
- No server left running on the agents.
- Results in `NOTES.md`; open items in `PLAN.md`'s pickup lists.
