#!/usr/bin/env python3
"""A plain-HTTP stand-in for the Hugging Face hub and for a file host, for
`server-diff`'s download cases (Stage 10 step 7). Standard library only.

    mock_hub.py <root> <portfile> <logfile>

serves, from <root>:

  The hub API, as `common/hf-cache.cpp` calls it, for every repository
  <root>/repos/<user>/<repo>/ (its files, recursively):
    GET  /api/models/<user>/<repo>/refs          {"branches": [main]}
    GET  /api/models/<user>/<repo>/tree/<commit>?recursive=true
    HEAD|GET /<user>/<repo>/resolve/<commit>/<path>
  A repository's commit is the SHA-1 of its name. A `.gguf` is an LFS file
  (`lfs.oid`, the SHA-256 of its bytes); anything else has a git `oid`. A
  file <root>/repos/<user>/<repo>.refs, when present, replaces the refs
  answer verbatim (bad branches, bad commits); <repo>.tree the tree's. A
  repository whose name starts with "slow" sends its files over ~15 s:
  long enough for both sides' requests to reach it mid-download, as
  server-diff sends each request to one side and then the other.

  Plain files, <root>/files/<name>, under these prefixes:
    /files/<name>         ETag "<sha-1>", ranges
    /noetag/<name>        no ETag
    /etag/<tag>/<name>    ETag "<tag>"
    /noranges/<name>      Accept-Ranges: none
    /redirect/<name>      302 to /files/<name>
    /redirect-esc/<name>  the same, the name's first letter percent-encoded
    /redir<code>/<name>   that status, with a Location to /files/<name>
    /redirect-dots/<name> 302 to /x/../files/./<name>
    /flaky/<name>         the first GET sends half the body and closes;
                          later ones (a Range resume) are whole
    /trunc/<name>         every GET sends half the body and closes
    /headonly/<name>      HEAD answers, GET is 404
    /status/<code>        that status, a small JSON body
    /slow/<name>          like /files, the body dribbled over ~15 s
    /r416/<name>          a GET with a Range is 416, without one whole
    /norange200/<name>    says it takes ranges, answers a Range with 200

A request in absolute-form (`GET http://host/path`) is what a client sends
to a proxy: it is logged as it came and served as its path.

Every request is logged to <logfile>, one line each: the method, the path,
and the headers the downloader sets (Range, Authorization, User-Agent,
Accept), in a fixed order -- so two mocks, one per side, can be compared.
"""

import hashlib
import http.server
import json
import urllib.parse
import os
import socketserver
import sys
import threading
import time

ROOT, PORTFILE, LOGFILE = sys.argv[1], sys.argv[2], sys.argv[3]
LOCK = threading.Lock()
FLAKY = {}


def log(line):
    with LOCK:
        with open(LOGFILE, "a") as f:
            f.write(line + "\n")


def commit_of(repo):
    return hashlib.sha1(repo.encode()).hexdigest()


def git_oid(data):
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def record(self):
        hs = [f"{k}={self.headers.get(k)}" for k in ("Range", "Authorization", "Proxy-Authorization", "User-Agent", "Accept") if self.headers.get(k) is not None]
        log(f"{self.command} {self.path} {' '.join(hs)}".rstrip())

    def send_json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def send_file(self, data, etag=None, ranges=True, mode="whole"):
        rng = self.headers.get("Range")
        start = 0
        code = 200
        if rng and ranges and rng.startswith("bytes=") and rng.endswith("-"):
            start = int(rng[6:-1])
            code = 206
        part = data[start:]
        self.send_response(code)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(part)))
        if etag is not None:
            self.send_header("ETag", etag)
        self.send_header("Accept-Ranges", "bytes" if ranges else "none")
        if code == 206:
            self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
        if mode == "half":
            self.send_header("Connection", "close")
        self.end_headers()
        if self.command == "HEAD":
            return
        if mode == "half":
            self.wfile.write(part[: len(part) // 2])
            self.wfile.flush()
            self.close_connection = True
            return
        if mode == "slow":
            step = max(1, len(part) // 30)
            for i in range(0, len(part), step):
                self.wfile.write(part[i:i + step])
                self.wfile.flush()
                time.sleep(0.5)
            return
        self.wfile.write(part)

    def repo_files(self, repo):
        base = os.path.join(ROOT, "repos", repo)
        out = []
        for dirpath, _, names in sorted(os.walk(base)):
            for n in sorted(names):
                full = os.path.join(dirpath, n)
                out.append((os.path.relpath(full, base), full))
        return out

    def handle_any(self):
        self.record()
        target = self.path
        if target.startswith("http://"):
            target = "/" + target[len("http://"):].partition("/")[2]
        path, _, query = target.partition("?")
        parts = path.split("/")
        # hub API
        if path.startswith("/api/models/"):
            rest = parts[3:]
            if len(rest) >= 3 and rest[2] == "refs":
                repo = "/".join(rest[:2])
                custom = os.path.join(ROOT, "repos", repo + ".refs")
                if os.path.isfile(custom):
                    return self.send_json(200, json.load(open(custom)))
                if not os.path.isdir(os.path.join(ROOT, "repos", repo)):
                    return self.send_json(404, {"error": "Repository not found"})
                return self.send_json(200, {"branches": [{"name": "main", "ref": "refs/heads/main", "targetCommit": commit_of(repo)}], "tags": [], "converts": []})
            if len(rest) >= 4 and rest[2] == "tree":
                repo = "/".join(rest[:2])
                custom = os.path.join(ROOT, "repos", repo + ".tree")
                if os.path.isfile(custom):
                    return self.send_json(200, json.load(open(custom)))
                if not os.path.isdir(os.path.join(ROOT, "repos", repo)):
                    return self.send_json(404, {"error": "Repository not found"})
                items = []
                for rel, full in self.repo_files(repo):
                    data = open(full, "rb").read()
                    item = {"type": "file", "oid": git_oid(data), "size": len(data), "path": rel}
                    if rel.endswith(".gguf"):
                        item["lfs"] = {"oid": hashlib.sha256(data).hexdigest(), "size": len(data), "pointerSize": 134}
                    items.append(item)
                return self.send_json(200, items)
            return self.send_json(404, {"error": "not found"})
        # resolve
        if len(parts) >= 6 and parts[3] == "resolve":
            repo = "/".join(parts[1:3])
            rel = "/".join(parts[5:])
            full = os.path.join(ROOT, "repos", repo, rel)
            if not os.path.isfile(full):
                return self.send_json(404, {"error": "Entry not found"})
            data = open(full, "rb").read()
            # a repository named slow* is dribbled over ~3 s
            return self.send_file(data, etag='"' + git_oid(data) + '"', mode="slow" if parts[2].startswith("slow") else "whole")
        if len(parts) >= 3 and parts[1] == "status":
            return self.send_json(int(parts[2]), {"error": f"status {parts[2]}"})
        kind = parts[1] if len(parts) > 2 else ""
        # a file name may arrive percent-encoded (redirect-esc), and is served
        # either way; the log keeps the raw path
        name = urllib.parse.unquote(parts[-1])
        full = os.path.join(ROOT, "files", name)
        if not os.path.isfile(full):
            return self.send_json(404, {"error": "no such file"})
        data = open(full, "rb").read()
        etag = '"' + hashlib.sha1(data).hexdigest() + '"'
        if kind == "files":
            return self.send_file(data, etag=etag)
        if kind == "noetag":
            return self.send_file(data)
        if kind == "etag":
            return self.send_file(data, etag='"' + parts[2] + '"')
        if kind == "noranges":
            return self.send_file(data, etag=etag, ranges=False)
        # cpp-httplib 0.64: only 301, 302, 303, 307 and 308 are followed, and
        # an absolute-path Location loses its dot segments
        if kind.startswith("redir") and kind[5:].isdigit():
            self.send_response(int(kind[5:]))
            self.send_header("Location", "/files/" + name)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if kind == "redirect-dots":
            self.send_response(302)
            self.send_header("Location", "/x/../files/./" + name)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if kind in ("redirect", "redirect-esc"):
            self.send_response(302)
            # redirect-esc spells the name's first letter as %XX: since
            # cpp-httplib 0.60 the client follows the Location as written,
            # 0.59 decoded it first (!mocklog shows which)
            self.send_header("Location", "/files/" + (name if kind == "redirect" else "%%%02X" % ord(name[0]) + name[1:]))
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if kind == "flaky":
            with LOCK:
                k = (self.command, name)
                FLAKY[k] = FLAKY.get(k, 0) + 1
                first = FLAKY[k] == 1
            return self.send_file(data, etag=etag, mode="half" if (first and self.command == "GET") else "whole")
        if kind == "trunc":
            return self.send_file(data, etag=etag, mode="half")
        if kind == "headonly":
            if self.command == "HEAD":
                return self.send_file(data, etag=etag)
            return self.send_json(404, {"error": "gone"})
        if kind == "slow":
            return self.send_file(data, etag=etag, mode="slow")
        if kind == "r416":
            if self.command == "GET" and self.headers.get("Range"):
                return self.send_json(416, {"error": "range"})
            return self.send_file(data, etag=etag)
        if kind == "norange200":
            if self.command == "GET" and self.headers.get("Range"):
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            return self.send_file(data, etag=etag)
        return self.send_json(404, {"error": "no such route"})

    def do_GET(self):
        self.handle_quietly()

    def do_HEAD(self):
        self.handle_quietly()

    def handle_quietly(self):
        # a cancelled download hangs up mid-body; that is the point of it
        try:
            self.handle_any()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    srv = Server(("127.0.0.1", 0), Handler)
    with open(PORTFILE + ".tmp", "w") as f:
        f.write(str(srv.server_address[1]))
    os.rename(PORTFILE + ".tmp", PORTFILE)
    srv.serve_forever()


if __name__ == "__main__":
    main()
