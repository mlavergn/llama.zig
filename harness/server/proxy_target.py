#!/usr/bin/env python3
"""The server the CORS proxy forwards to in proxy.cases (scripts/server-diff):
raw sockets, so every byte of each answer is fixed.

    proxy_target.py <port file> [lifetime seconds]

Writes the port it listens on into <port file>. Paths:
    /echo...           200, JSON: method, target, headers in arrival order, body
    /status/<code>     that status, a short body (none for 204 and 304)
    /redirect/<to>     302 to /<to> on this host, Connection: close
    /keep/<to>         302 to /<to>, the connection kept open for it
    /redirect-abs/<to> 302 to http://localhost:<port>/<to> (another host name)
    /see-other         303 to /echo
    /loop              302 to itself
    /redirect-ftp      302 to an ftp URL
    /redirect-empty    302 with no Location
    /chunked           200, three chunks and a trailer
    /close             200, no length: the body runs to the close
    /http10            an HTTP/1.0 answer
    /dup-headers       one name set three times, in two cases
    /gzip              a gzip Content-Encoding this build cannot decode
    /bad-status        a status line that does not parse
    /conflict          both a Transfer-Encoding and a Content-Length
    /short             a Content-Length longer than the body sent
    /continue-reject   417 to an Expect: 100-continue, before the body
A request with Expect: 100-continue is answered 100 first, unless the
path is /no-continue (then the client waits out its timeout).
"""
import json
import socket
import sys
import threading
import time


def read_request(conn):
    f = conn.makefile("rb")
    line = f.readline()
    if not line:
        return None
    method, target, _ = line.decode("latin-1").rstrip("\r\n").split(" ", 2)
    headers = []
    while True:
        h = f.readline().decode("latin-1")
        if h in ("\r\n", "\n", ""):
            break
        k, _, v = h.rstrip("\r\n").partition(":")
        headers.append([k, v.strip()])
    return f, method, target, headers


def get(headers, name):
    for k, v in headers:
        if k.lower() == name:
            return v
    return None


def respond(conn, status, reason, headers, body):
    out = f"HTTP/1.1 {status} {reason}\r\n"
    for k, v in headers:
        out += f"{k}: {v}\r\n"
    out += "\r\n"
    conn.sendall(out.encode("latin-1") + body)


def handle(conn, port):
    try:
        while serve_one(conn, port):
            pass
    except (OSError, ValueError):
        pass
    finally:
        try:
            conn.shutdown(socket.SHUT_WR)
        except OSError:
            pass
        conn.close()


def serve_one(conn, port):
    """One request; true when the connection is kept for another."""
    if True:
        req = read_request(conn)
        if req is None:
            return False
        f, method, target, headers = req
        path = target.split("?")[0]
        expect = (get(headers, "expect") or "").lower() == "100-continue"
        if path == "/continue-reject" and expect:
            respond(conn, 417, "Expectation Failed", [("Content-Length", "8"), ("Connection", "close")], b"rejected")
            return False
        if expect and path != "/no-continue":
            conn.sendall(b"HTTP/1.1 100 Continue\r\n\r\n")
        n = int(get(headers, "content-length") or 0)
        body = f.read(n) if n else b""
        common = [("Server", "proxy-target"), ("Access-Control-Allow-Origin", "*"), ("Keep-Alive", "timeout=5"), ("X-Target", "yes")]
        if path.startswith("/echo") or path == "/no-continue":
            data = json.dumps({"method": method, "target": target, "headers": headers,
                               "body": body.decode("utf-8", "replace")}).encode()
            respond(conn, 200, "OK", common + [("Content-Type", "application/json"), ("Content-Length", str(len(data))), ("Connection", "close")], data)
        elif path.startswith("/status/"):
            code = int(path[len("/status/"):])
            data = b"" if code in (204, 304) else f"status {code}".encode()
            hs = common + [("Content-Type", "text/plain")]
            if code not in (204, 304):
                hs.append(("Content-Length", str(len(data))))
            respond(conn, code, "Whatever", hs, data)
        elif path.startswith("/redirect/"):
            respond(conn, 302, "Found", [("Location", "/" + path[len("/redirect/"):]), ("Content-Length", "5"), ("Connection", "close")], b"moved")
        elif path.startswith("/keep/"):
            respond(conn, 302, "Found", [("Location", "/" + path[len("/keep/"):]), ("Content-Length", "4")], b"kept")
            return True
        elif path.startswith("/redirect-abs/"):
            respond(conn, 302, "Found", [("Location", f"http://localhost:{port}/" + path[len("/redirect-abs/"):]), ("Content-Length", "0")], b"")
        elif path == "/see-other":
            respond(conn, 303, "See Other", [("Location", "/echo?after=303"), ("Content-Length", "0"), ("Connection", "close")], b"")
        elif path == "/loop":
            respond(conn, 302, "Found", [("Location", "/loop"), ("Content-Length", "0"), ("Connection", "close")], b"")
        elif path == "/redirect-ftp":
            respond(conn, 302, "Found", [("Location", "ftp://example.invalid/x"), ("Content-Length", "0"), ("Connection", "close")], b"")
        elif path == "/redirect-empty":
            respond(conn, 302, "Found", [("Content-Length", "0"), ("Connection", "close")], b"")
        elif path == "/chunked":
            conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nTransfer-Encoding: chunked\r\nTrailer: X-Tail\r\n\r\n")
            for part in (b"first ", b"second ", b"third"):
                conn.sendall(f"{len(part):x}\r\n".encode() + part + b"\r\n")
                time.sleep(0.05)
            conn.sendall(b"0\r\nX-Tail: done\r\n\r\n")
        elif path == "/close":
            conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nConnection: close\r\n\r\nall of this until the close")
        elif path == "/http10":
            conn.sendall(b"HTTP/1.0 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 6\r\n\r\nold ok")
        elif path == "/dup-headers":
            respond(conn, 200, "OK", [("X-Dup", "1"), ("x-dup", "2"), ("X-Dup", "3"), ("Content-Type", "text/a"), ("content-type", "text/b"), ("Content-Length", "2")], b"ok")
        elif path == "/gzip":
            respond(conn, 200, "OK", [("Content-Type", "text/plain"), ("Content-Encoding", "gzip"), ("Content-Length", "4")], b"\x1f\x8b\x08\x00")
        elif path == "/bad-status":
            conn.sendall(b"HTTP/2 200 OK\r\nContent-Length: 0\r\n\r\n")
        elif path == "/conflict":
            conn.sendall(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nContent-Length: 5\r\n\r\n5\r\nhello\r\n0\r\n\r\n")
        elif path == "/short":
            conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 50\r\n\r\nonly this")
        else:
            respond(conn, 404, "Not Found", [("Content-Length", "9")], b"not found")
        return False


def main():
    port_file = sys.argv[1]
    lifetime = float(sys.argv[2]) if len(sys.argv) > 2 else 3600
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(64)
    port = srv.getsockname()[1]
    with open(port_file, "w") as f:
        f.write(str(port))
    deadline = time.time() + lifetime
    srv.settimeout(1.0)
    while time.time() < deadline:
        try:
            conn, _ = srv.accept()
        except socket.timeout:
            continue
        threading.Thread(target=handle, args=(conn, port), daemon=True).start()


if __name__ == "__main__":
    main()
