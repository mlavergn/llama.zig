#!/usr/bin/env python3
"""http-diff's driver: raw bytes to two HTTP servers, raw bytes compared.

Starts harness/http_ref.cpp (upstream's server_http_context over
cpp-httplib) and harness/http_diff.zig (the Zig port) under each
configuration, sends every case's bytes to each over its own TCP (or Unix)
connection, and requires the bytes that come back -- status line, headers,
body, chunk framing, and whether and how the connection ends -- to be
identical. Then each server is stopped and everything it logged, and its
exit code, must be identical too.

Masked, because they differ by design: the listening port and Unix socket
path in the log, and cpp-httplib's random multipart/byteranges boundary.

Not a port: test scaffolding. Python 3 standard library only.

Usage: http_driver.py <http_ref> <http_port> [--only CONFIG[,CONFIG]] [--keep DIR] [--case SUBSTR]
"""

import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time

# -----------------------------------------------------------------------------
# Steps

READ_TIMEOUT = 8.0


class Conn:
    """One client connection and the transcript of what it saw."""

    def __init__(self, target):
        self.target = target
        if isinstance(target, str):
            self.s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.s.connect(target)
        else:
            self.s = socket.create_connection(target)
        self.s.settimeout(READ_TIMEOUT)
        self.buf = b""
        self.out = []
        self.eof = False

    def note(self, text):
        self.out.append(b"\n<<" + text.encode() + b">>\n")

    def fill(self):
        """Reads more into buf; returns False at EOF, reset or timeout."""
        if self.eof:
            return False
        try:
            data = self.s.recv(65536)
        except socket.timeout:
            self.note("TIMEOUT")
            self.eof = True
            return False
        except ConnectionResetError:
            self.note("RESET")
            self.eof = True
            return False
        if not data:
            self.eof = True
            return False
        self.buf += data
        return True

    def take(self, n):
        piece, self.buf = self.buf[:n], self.buf[n:]
        self.out.append(piece)

    def until(self, pat):
        while pat not in self.buf:
            if not self.fill():
                self.take(len(self.buf))
                self.note("EOF before %r" % pat)
                return False
        self.take(self.buf.index(pat) + len(pat))
        return True

    def exactly(self, n):
        while len(self.buf) < n:
            if not self.fill():
                self.take(len(self.buf))
                self.note("EOF in body")
                return False
        self.take(n)
        return True

    def response(self, head_only=False):
        """One response, framed as the server framed it."""
        start = len(b"".join(self.out))
        if not self.until(b"\r\n\r\n"):
            return
        head = b"".join(self.out)[start:]
        lines = head.split(b"\r\n")
        headers = {}
        for line in lines[1:]:
            if b":" in line:
                k, v = line.split(b":", 1)
                headers.setdefault(k.strip().lower(), v.strip())
        status = lines[0].split(b" ")
        if len(status) > 1 and status[1].startswith(b"1"):
            return
        if head_only:
            return
        if headers.get(b"transfer-encoding", b"").lower() == b"chunked":
            while True:
                if not self.until(b"\r\n"):
                    return
                size_line = b"".join(self.out).rsplit(b"\r\n", 2)[-2]
                size = int(size_line.split(b";")[0], 16)
                if size == 0:
                    self.until(b"\r\n")
                    return
                if not self.exactly(size + 2):
                    return
        elif b"content-length" in headers:
            self.exactly(int(headers[b"content-length"]))
        else:
            self.rest()

    def ws_frame(self):
        """One server frame, as the bytes arrive."""
        while len(self.buf) < 2:
            if not self.fill():
                self.take(len(self.buf))
                self.note("EOF in frame")
                return
        n = self.buf[1] & 0x7F
        head = 2
        if n == 126:
            head = 4
        elif n == 127:
            head = 10
        while len(self.buf) < head:
            if not self.fill():
                self.take(len(self.buf))
                self.note("EOF in frame")
                return
        if n == 126:
            n = int.from_bytes(self.buf[2:4], "big")
        elif n == 127:
            n = int.from_bytes(self.buf[2:10], "big")
        self.exactly(head + n)

    def rest(self):
        """Everything until the server closes."""
        while self.fill():
            pass
        self.take(len(self.buf))
        self.note("EOF")

    def run(self, steps):
        for st in steps:
            op = st[0]
            if op == "send":
                try:
                    self.s.sendall(st[1])
                except OSError as e:
                    self.note("SEND " + type(e).__name__)
            elif op == "resp":
                self.response()
            elif op == "head":
                self.response(head_only=True)
            elif op == "upto":
                self.until(st[1])
            elif op == "wsframe":
                self.ws_frame()
            elif op == "sleep":
                time.sleep(st[1] / 1000.0)
            elif op == "shut":
                try:
                    self.s.shutdown(socket.SHUT_WR)
                except OSError:
                    pass
            elif op == "eof":
                self.rest()
            elif op == "close":
                self.s.close()
                return
        # end every case the same way: half-close, read to the server's close
        try:
            self.s.shutdown(socket.SHUT_WR)
        except OSError:
            pass
        self.rest()
        self.s.close()


# -----------------------------------------------------------------------------
# Cases

H = b"Host: x\r\n"


def req(line, *headers, body=b""):
    return line.encode() + b"\r\n" + H + b"".join(h.encode() + b"\r\n" for h in headers) + b"\r\n" + body


def simple(name, data, *more):
    """A case: send `data`, read one response, then whatever `more` says."""
    return (name, [("send", data), ("resp",)] + list(more))


def mp(boundary, parts, end=True):
    out = b""
    for headers, content in parts:
        out += b"--" + boundary + b"\r\n" + b"".join(h + b"\r\n" for h in headers) + b"\r\n" + content + b"\r\n"
    if end:
        out += b"--" + boundary + b"--\r\n"
    return out


def post(path, body, *headers, ct="application/json"):
    hs = list(headers)
    if ct:
        hs.append("Content-Type: " + ct)
    hs.append("Content-Length: %d" % len(body))
    return req("POST %s HTTP/1.1" % path, *hs, body=body)


def base_cases():
    c = []
    # well-formed requests
    c.append(simple("get-echo", req("GET /echo HTTP/1.1")))
    c.append(simple("get-query", req("GET /echo?a=1&b=%41%42&c=x+y&a=2&a=1&flag&=v&d=%zz&e=%4&f=%u0041&g=%2B%26%3D HTTP/1.1")))
    c.append(simple("get-query-empty", req("GET /echo? HTTP/1.1")))
    c.append(simple("get-query-nul", req("GET /echo?n=a%00b&u=%C3%A9&bad=%FF HTTP/1.1")))
    c.append(simple("get-path-escape", req("GET /ec%68o HTTP/1.1")))
    c.append(simple("get-path-u", req("GET /%u0065cho HTTP/1.1")))
    c.append(simple("get-fragment", req("GET /echo?a=1#frag?b=2 HTTP/1.1")))
    c.append(simple("get-spaces", b"GET   /echo   HTTP/1.1\r\n" + H + b"\r\n"))
    c.append(simple("get-header-cases", req("GET /echo HTTP/1.1", "host: y", "X-Dup: 1", "x-dup: 2", "X-Empty:", "X-Spaces:   a b   ")))
    c.append(simple("get-body", req("GET /echo HTTP/1.1", "Content-Length: 5", body=b"hello")))
    c.append(simple("get-health", req("GET /health HTTP/1.1")))
    c.append(simple("get-root", req("GET / HTTP/1.1")))
    c.append(simple("get-empty", req("GET /empty HTTP/1.1")))
    c.append(simple("http10", b"GET /echo HTTP/1.0\r\n" + H + b"\r\n"))
    c.append(simple("http10-keepalive", b"GET /echo HTTP/1.0\r\n" + H + b"Connection: keep-alive\r\n\r\n",
                    ("send", req("GET /health HTTP/1.1")), ("resp",)))
    c.append(simple("conn-close", req("GET /echo HTTP/1.1", "Connection: Close")))
    c.append(simple("conn-close-list", req("GET /echo HTTP/1.1", "Connection: keep-alive, close")))
    c.append(simple("post-echo", post("/echo", b'{"hello":123}')))
    c.append(simple("post-echo-nonutf8", post("/echo", b"\xff\xfe\x00ok", ct="text/plain")))
    c.append(simple("post-no-ct", post("/echo", b"plain", ct=None)))
    c.append(simple("post-form", post("/echo?q=1", b"a=1&b=two+words&a=3&q=2", ct="application/x-www-form-urlencoded; charset=utf-8")))
    c.append(simple("post-chunked", req("POST /echo HTTP/1.1", "Transfer-Encoding: chunked",
                                        body=b"5\r\nhello\r\n6;ext=1\r\n world\r\n0\r\n\r\n")))
    c.append(simple("post-chunked-upper", req("POST /echo HTTP/1.1", "Transfer-Encoding: gzip, CHUNKED",
                                              body=b"A\r\n0123456789\r\n0\r\n\r\n")))
    c.append(simple("post-chunked-trailer", req("POST /echo HTTP/1.1", "Transfer-Encoding: chunked", "Trailer: X-T, Host",
                                                body=b"3\r\nabc\r\n0\r\nX-T: v\r\nHost: z\r\nX-U: w\r\n\r\n")))
    c.append(simple("post-chunked-longsize", req("POST /echo HTTP/1.1", "Transfer-Encoding: chunked",
                                                 body=b"0" * 70 + b"3\r\nabc\r\n0\r\n\r\n")))
    c.append(simple("post-chunked-badsize", req("POST /echo HTTP/1.1", "Transfer-Encoding: chunked", body=b"zz\r\nabc\r\n0\r\n\r\n")))
    c.append(simple("post-chunked-nocrlf", req("POST /echo HTTP/1.1", "Transfer-Encoding: chunked", body=b"3\r\nabcX\r\n0\r\n\r\n")))
    c.append(simple("post-chunked-badtrailer", req("POST /echo HTTP/1.1", "Transfer-Encoding: chunked", body=b"3\r\nabc\r\n0\r\nbad trailer\r\n\r\n")))
    c.append(simple("post-chunked-on-get", req("GET /echo HTTP/1.1", "Transfer-Encoding: chunked", body=b"2\r\nhi\r\n0\r\n\r\n")))
    c.append(simple("post-ce-gzip", post("/echo", b"xx", "Content-Encoding: gzip")))
    c.append(simple("post-ce-identity", post("/echo", b"xx", "Content-Encoding: identity")))
    c.append(simple("post-ce-list", post("/echo", b"xx", "Content-Encoding: gzip, br")))
    c.append(simple("post-cl-dup-same", req("POST /echo HTTP/1.1", "Content-Length: 2", "Content-Length: 2", body=b"ok")))
    c.append(simple("post-cl-dup-diff", req("POST /echo HTTP/1.1", "Content-Length: 2", "Content-Length: 3", body=b"ok!")))
    c.append(simple("post-cl-bad", req("POST /echo HTTP/1.1", "Content-Length: abc", body=b"ok")))
    c.append(simple("post-cl-huge", req("POST /echo HTTP/1.1", "Content-Length: 99999999999999999999999", body=b"ok")))
    c.append(simple("post-cl-zero", req("POST /echo HTTP/1.1", "Content-Length: 0")))
    c.append(simple("post-te-and-cl", req("POST /echo HTTP/1.1", "Transfer-Encoding: chunked", "Content-Length: 5", body=b"0\r\n\r\n")))
    c.append(simple("post-te-and-cl0", req("POST /echo HTTP/1.1", "Transfer-Encoding: chunked", "Content-Length: 0", body=b"2\r\nok\r\n0\r\n\r\n")))
    c.append(simple("post-te-gzip", req("POST /echo HTTP/1.1", "Transfer-Encoding: gzip", body=b"abc")))
    c.append(simple("post-nolength-close", req("POST /echo HTTP/1.1", "Connection: close", body=b"until-eof"), ))
    c[-1] = ("post-nolength-close", [("send", c[-1][1][0][1]), ("shut",), ("resp",)])
    c.append(simple("post-nolength-keepalive", req("POST /echo HTTP/1.1", body=b"GET /health HTTP/1.1\r\nHost: x\r\n\r\n"), ("resp",)))
    c.append(("body-short-close", [("send", req("POST /echo HTTP/1.1", "Content-Length: 10", body=b"abc")), ("shut",), ("resp",)]))
    c.append(simple("expect-100", req("POST /echo HTTP/1.1", "Content-Length: 4", "Expect: 100-continue"), ("send", b"body"), ("resp",)))
    c.append(simple("expect-100-list", req("POST /echo HTTP/1.1", "Content-Length: 4", "Expect: foo, 100-CONTINUE"), ("send", b"body"), ("resp",)))
    c.append(simple("expect-100-http10", b"POST /echo HTTP/1.0\r\n" + H + b"Content-Length: 4\r\nExpect: 100-continue\r\n\r\nbody"))
    # cpp-httplib 0.57: `100 Continue` is sent only when the body is read, and
    # a body still withheld for it closes the connection
    c.append(simple("expect-100-404", req("POST /nope HTTP/1.1", "Content-Length: 4", "Expect: 100-continue")))
    c.append(simple("expect-100-get", req("GET /echo HTTP/1.1", "Content-Length: 4", "Expect: 100-continue")))
    c.append(simple("expect-100-get-nobody", req("GET /echo HTTP/1.1", "Expect: 100-continue"), ("send", req("GET /health HTTP/1.1")), ("resp",)))
    # answered before routing reads the body: the connection closes
    c.append(simple("expect-100-options", req("OPTIONS /echo HTTP/1.1", "Content-Length: 4", "Expect: 100-continue"),
                    ("send", req("GET /health HTTP/1.1")), ("resp",)))
    c.append(simple("expect-100-chunked", req("POST /echo HTTP/1.1", "Transfer-Encoding: chunked", "Expect: 100-continue"), ("send", b"3\r\nabc\r\n0\r\n\r\n"), ("resp",)))
    # cpp-httplib 0.57: the chunk-size line is bounded by its terminator, and
    # only a chunk extension of visible characters may follow the size
    c.append(simple("post-chunked-ext", req("POST /echo HTTP/1.1", "Transfer-Encoding: chunked", body=b"3;name=\"v a\"\r\nabc\r\n0\r\n\r\n")))
    c.append(simple("post-chunked-ext-ctl", req("POST /echo HTTP/1.1", "Transfer-Encoding: chunked", body=b"3;a\x01b\r\nabc\r\n0\r\n\r\n")))
    c.append(simple("post-chunked-size-nul", req("POST /echo HTTP/1.1", "Transfer-Encoding: chunked", body=b"3\x00junk\r\nabc\r\n0\r\n\r\n")))
    c.append(simple("post-chunked-ext-lf", req("POST /echo HTTP/1.1", "Transfer-Encoding: chunked", body=b"3;a\nb\r\nabc\r\n0\r\n\r\n")))
    c.append(simple("post-chunked-ext-tab", req("POST /echo HTTP/1.1", "Transfer-Encoding: chunked", body=b"3 \t;a=\x80\t\r\nabc\r\n0\r\n\r\n")))
    c.append(simple("post-chunked-size-trail", req("POST /echo HTTP/1.1", "Transfer-Encoding: chunked", body=b"3 x\r\nabc\r\n0\r\n\r\n")))
    # cpp-httplib 0.57: every trailer field received counts toward the limit
    c.append(simple("post-chunked-100-trailers", req("POST /echo HTTP/1.1", "Transfer-Encoding: chunked",
                                                     body=b"3\r\nabc\r\n0\r\n" + b"X-U: 1\r\n" * 100 + b"\r\n")))
    c.append(simple("post-chunked-101-trailers", req("POST /echo HTTP/1.1", "Transfer-Encoding: chunked",
                                                     body=b"3\r\nabc\r\n0\r\n" + b"X-U: 1\r\n" * 101 + b"\r\n")))
    c.append(simple("post-chunked-declared-trailers", req("POST /echo HTTP/1.1", "Transfer-Encoding: chunked",
                                                          "Trailer: " + ", ".join("X-T%d" % i for i in range(150)),
                                                          body=b"3\r\nabc\r\n0\r\nX-T120: late\r\nX-T3: early\r\n\r\n")))
    # multipart
    b = b"XyZ"
    form = mp(b, [
        ([b'Content-Disposition: form-data; name="name"'], b"alice"),
        ([b'Content-Disposition: form-data; name="name"'], b"bob"),
        ([b'Content-Disposition: form-data; name="name"'], b"carol"),
        ([b'content-disposition: FORM-DATA; name=x; filename="a;b.txt"', b"Content-Type:  text/plain "], b"line1\r\nline2"),
        ([b"Content-Disposition: form-data; name=\"u\"; filename*=UTF-8''%C3%A9t%C3%A9.bin"], b"\x00\x01\xff"),
        ([b'Content-Disposition: form-data; name="x"; filename="second.txt"'], b"replaces"),
        ([b'Content-Disposition: form-data; name="q";'], b"semi"),
    ])
    c.append(simple("multipart", post("/echo", form, ct='multipart/form-data; boundary="XyZ"')))
    c.append(simple("multipart-preamble", post("/echo", b"preamble\r\n" + mp(b, [([b'Content-Disposition: form-data; name="a"'], b"1")]) + b"epilogue",
                                               ct="multipart/form-data; boundary=XyZ")))
    c.append(simple("multipart-nonutf8", post("/echo", mp(b, [([b'Content-Disposition: form-data; name="a"'], b"\xff\xfe")]), ct="multipart/form-data; boundary=XyZ")))
    c.append(simple("multipart-unterminated", post("/echo", mp(b, [([b'Content-Disposition: form-data; name="a"'], b"1")], end=False), ct="multipart/form-data; boundary=XyZ")))
    c.append(simple("multipart-noboundary", post("/echo", b"--XyZ\r\n", ct="multipart/form-data")))
    c.append(simple("multipart-longboundary", post("/echo", b"x", ct="multipart/form-data; boundary=" + "b" * 71)))
    c.append(simple("multipart-noname", post("/echo", mp(b, [([b'Content-Disposition: form-data; filename="f"'], b"1")]), ct="multipart/form-data; boundary=XyZ")))
    c.append(simple("multipart-badfilenamestar", post("/echo", mp(b, [([b"Content-Disposition: form-data; name=a; filename*=latin1''x"], b"1")]), ct="multipart/form-data; boundary=XyZ")))
    c.append(simple("multipart-badheader", post("/echo", mp(b, [([b"No colon here"], b"1")]), ct="multipart/form-data; boundary=XyZ")))
    c.append(simple("multipart-chunked", req("POST /echo HTTP/1.1", "Transfer-Encoding: chunked", "Content-Type: multipart/form-data; boundary=XyZ",
                                             body=b"%x\r\n" % 10 + mp(b, [([b'Content-Disposition: form-data; name="k"'], b"v")])[:10] + b"\r\n"
                                             + b"%x\r\n" % (len(mp(b, [([b'Content-Disposition: form-data; name="k"'], b"v")])) - 10)
                                             + mp(b, [([b'Content-Disposition: form-data; name="k"'], b"v")])[10:] + b"\r\n0\r\n\r\n")))
    # methods and routes
    c.append(simple("delete-echo", req("DELETE /echo?x=1 HTTP/1.1", "Content-Length: 3", body=b"del")))
    c.append(simple("delete-nobody", req("DELETE /echo HTTP/1.1")))
    c.append(simple("head-echo", req("HEAD /echo HTTP/1.1"), ("send", req("GET /health HTTP/1.1")), ("resp",)))
    c[-1] = ("head-echo", [("send", req("HEAD /echo HTTP/1.1")), ("head",), ("send", req("GET /health HTTP/1.1")), ("resp",)])
    c.append(("head-stream", [("send", req("HEAD /stream HTTP/1.1")), ("head",), ("send", req("GET /health HTTP/1.1")), ("resp",)]))
    c.append(simple("options", req("OPTIONS /echo HTTP/1.1", "Origin: http://a.example")))
    c.append(simple("options-unknown", req("OPTIONS /nowhere HTTP/1.1")))
    c.append(simple("options-range", req("OPTIONS /echo HTTP/1.1", "Range: bytes=0-5")))
    c.append(simple("origin-echo", req("GET /echo HTTP/1.1", "Origin: http://localhost:8080")))
    c.append(simple("not-found", req("GET /nope HTTP/1.1")))
    c.append(simple("post-on-get-route", post("/health", b"{}")))
    c.append(simple("put", req("PUT /echo HTTP/1.1", "Content-Length: 2", body=b"hi")))
    c.append(simple("patch", req("PATCH /echo HTTP/1.1")))
    c.append(simple("connect", req("CONNECT /echo HTTP/1.1")))
    c.append(simple("trace", req("TRACE /echo HTTP/1.1")))
    c.append(simple("slots", post("/slots/3?action=save", b'{"f":"x"}')))
    c.append(simple("slots-escaped", post("/slots/conv%3A%3Amodel%2", b"")))
    # cpp-httplib decodes the whole path first, so server-http.cpp's own
    # decode of a path parameter only shows on a double-encoded one
    c.append(simple("slots-double-escaped", post("/slots/a%253Ab%2541%25%2", b"")))
    c.append(simple("slots-empty", post("/slots/", b"")))
    c.append(simple("slots-extra", post("/slots/1/2", b"")))
    c.append(simple("slots-shadow", post("/slots/9?id_slot=q&x=1", b"")))
    c.append(simple("two-params", req("GET /two/p%20q/x/r HTTP/1.1")))
    c.append(simple("two-params-double", req("GET /two/p%2520q/x/%25zz%2541 HTTP/1.1")))
    c.append(simple("two-params-miss", req("GET /two/p/y/r HTTP/1.1")))
    # malformed
    c.append(simple("bad-method", req("BREW /echo HTTP/1.1")))
    c.append(simple("bad-method-lower", req("get /echo HTTP/1.1")))
    c.append(simple("bad-version", req("GET /echo HTTP/2.0")))
    c.append(simple("bad-version-09", b"GET /echo\r\n\r\n"))
    c.append(simple("bad-four-parts", req("GET /echo HTTP/1.1 x")))
    c.append(simple("bad-tab", b"GET\t/echo HTTP/1.1\r\n" + H + b"\r\n"))
    c.append(simple("bad-nul-line", b"GET /echo\x00 HTTP/1.1\r\n" + H + b"\r\n"))
    c.append(("bad-bare-lf", [("send", b"GET /echo HTTP/1.1\n" + H + b"\n"), ("shut",), ("resp",)]))
    c.append(simple("bad-header-nocolon", req("GET /echo HTTP/1.1", "NoColon")))
    c.append(simple("bad-header-space-name", req("GET /echo HTTP/1.1", "Bad Name: v")))
    c.append(simple("bad-header-ctl", req("GET /echo HTTP/1.1", "X-Ctl: a\x01b")))
    c.append(simple("bad-header-fold", req("GET /echo HTTP/1.1", "X-A: 1", " continued")))
    c.append(simple("bad-header-lf-inside", b"GET /echo HTTP/1.1\r\nHost: x\nX-A: 1\r\n\r\n"))
    c.append(simple("bad-header-empty-name", req("GET /echo HTTP/1.1", ": v")))
    c.append(simple("header-obs-text", req("GET /echo HTTP/1.1") [:-2] + b"X-Obs: caf\xe9\r\n\r\n"))
    c.append(simple("header-8192", req("GET /echo HTTP/1.1", "X-L: " + "a" * (8192 - 7))))
    c.append(simple("header-8193", req("GET /echo HTTP/1.1", "X-L: " + "a" * (8192 - 6))))
    c.append(simple("headers-100", req("GET /health HTTP/1.1", *["X-%d: v" % i for i in range(99)])))
    c.append(simple("headers-101", req("GET /health HTTP/1.1", *["X-%d: v" % i for i in range(100)])))
    c.append(simple("line-32768", ("GET /" + "a" * (32768 - 16) + " HTTP/1.1\r\n").encode() + H + b"\r\n"))
    c.append(("line-32769", [("send", ("GET /" + "a" * (32768 - 15) + " HTTP/1.1\r\n").encode() + H + b"\r\n")]))
    c.append(simple("accept-ok", req("GET /echo HTTP/1.1", "Accept: text/html;q=0.5, */*;q=0.1, application/json", "Accept: ,,")))
    c.append(simple("accept-bad", req("GET /echo HTTP/1.1", "Accept: foo")))
    c.append(simple("accept-badq", req("GET /echo HTTP/1.1", "Accept: text/html;q=abc")))
    c.append(simple("accept-q2", req("GET /echo HTTP/1.1", "Accept: text/html;Q=2")))
    c.append(simple("accept-qtrail", req("GET /echo HTTP/1.1", "Accept: text/html;q=0.5xyz;q=7")))
    # ranges
    c.append(simple("range-on-200", req("GET /echo HTTP/1.1", "Range: bytes=0-4")))
    c.append(simple("range-out", req("GET /echo HTTP/1.1", "Range: bytes=5000-6000")))
    c.append(simple("range-bad", req("GET /echo HTTP/1.1", "Range: items=0-1")))
    c.append(simple("range-bad2", req("GET /echo HTTP/1.1", "Range: bytes=5-1")))
    c.append(simple("range-206", req("GET /status?code=206&body=0123456789 HTTP/1.1", "Range: bytes=2-5")))
    c.append(simple("range-206-suffix", req("GET /status?code=206&body=0123456789 HTTP/1.1", "Range: bytes=-3")))
    c.append(simple("range-206-open", req("GET /status?code=206&body=0123456789 HTTP/1.1", "Range: bytes=7-")))
    c.append(simple("range-206-multi", req("GET /status?code=206&body=0123456789&ct=text/x HTTP/1.1", "Range: bytes=0-1, 4-5")))
    c.append(simple("range-206-coalesce", req("GET /status?code=206&body=0123456789 HTTP/1.1", "Range: bytes=4-6,0-2,2-4,0-2")))
    c.append(simple("range-206-overlap3", req("GET /status?code=206&body=0123456789 HTTP/1.1", "Range: bytes=0-5,1-5,2-5,3-5")))
    c.append(simple("range-206-head", req("HEAD /status?code=206&body=0123456789 HTTP/1.1", "Range: bytes=2-5")))
    c[-1] = ("range-206-head", [("send", c[-1][1][0][1]), ("head",)])
    c.append(simple("range-stream", req("GET /stream HTTP/1.1", "Range: bytes=0-2")))
    # handlers' responses
    c.append(simple("status-404", req("GET /status?code=404&body=hidden HTTP/1.1")))
    c.append(simple("status-500", req("GET /status?code=500&body=err&ct=text/plain HTTP/1.1")))
    c.append(simple("status-418", req("GET /status?code=418 HTTP/1.1")))
    c.append(simple("status-299", req("GET /status?code=299&body=x HTTP/1.1")))
    c.append(simple("status-bad", req("GET /status?code=abc HTTP/1.1")))
    c.append(simple("status-range", req("GET /status?code=99999999999 HTTP/1.1")))
    c.append(simple("status-headers", req("GET /status?body=abc&h_X-Test=1&h_Bad%20Name=v&h_X-Inj=a%0D%0Ab&h_Content-Type=text/x&h_Content-Length=5&h_a=%20sp HTTP/1.1")))
    c.append(simple("status-ct-empty", req("GET /status?ct=&body=x HTTP/1.1")))
    c.append(simple("throw-invalid", post("/throw-invalid", b"why")))
    c.append(simple("throw-runtime", post("/throw-runtime", b"")))
    c.append(simple("throw-unknown", req("GET /throw-unknown HTTP/1.1")))
    c.append(simple("throw-raw", req("GET /throw-raw HTTP/1.1")))
    c.append(simple("throw-raw-head", req("GET /throw-raw HTTP/1.1", "Range: bytes=0-1")))
    c.append(simple("websocket-upgrade", req("GET /echo HTTP/1.1", "Upgrade: websocket", "Connection: Upgrade",
                                             "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==", "Sec-WebSocket-Version: 13")))
    # keep-alive and pipelining
    c.append(("keepalive-3", [("send", req("GET /echo?n=1 HTTP/1.1")), ("resp",), ("send", req("POST /echo HTTP/1.1", "Content-Length: 2", body=b"n2")),
                              ("resp",), ("send", req("GET /echo?n=3 HTTP/1.1")), ("resp",)]))
    c.append(("keepalive-after-404", [("send", req("GET /nope HTTP/1.1")), ("resp",), ("send", req("GET /health HTTP/1.1")), ("resp",)]))
    c.append(("keepalive-after-badline", [("send", req("BREW / HTTP/1.1")), ("resp",), ("send", req("GET /health HTTP/1.1")), ("resp",)]))
    c.append(("pipelined", [("send", req("GET /echo?n=1 HTTP/1.1") + req("GET /echo?n=2 HTTP/1.1")), ("resp",), ("resp",)]))
    c.append(("pipelined-body", [("send", req("POST /echo HTTP/1.1", "Content-Length: 3", body=b"abc") + req("GET /health HTTP/1.1")), ("resp",), ("resp",)]))
    c.append(("unread-body-drained", [("send", req("GET /health HTTP/1.1", "Content-Length: 3", body=b"abc")), ("resp",),
                                      ("send", req("GET /health HTTP/1.1")), ("resp",)]))
    c.append(("unread-body-404", [("send", req("POST /nope HTTP/1.1", "Content-Length: 3", body=b"abc")), ("resp",)]))
    # streaming
    c.append(("stream", [("send", req("GET /stream HTTP/1.1")), ("resp",)]))
    c.append(("stream-then-request", [("send", req("GET /stream HTTP/1.1")), ("resp",), ("send", req("GET /health HTTP/1.1")), ("resp",)]))
    c.append(("stream-http10", [("send", b"GET /stream HTTP/1.0\r\n" + H + b"\r\n"), ("resp",)]))
    c.append(("stream-abandoned", [("send", req("GET /stream-wait HTTP/1.1")), ("upto", b"first\n"), ("sleep", 300), ("close",)]))
    c.append(simple("stream-abandoned-done", req("GET /wait-complete HTTP/1.1")))
    # cpp-httplib 0.58/0.59: pipelined requests are served from the stream's
    # buffer, the connection closes after any response saying `Connection:
    # close`, and a rejected request line or header ends it
    c.append(("pipelined-3", [("send", req("GET /echo?n=1 HTTP/1.1") + req("POST /echo HTTP/1.1", "Content-Length: 2", body=b"n2")
                                       + req("GET /echo?n=3 HTTP/1.1")), ("resp",), ("resp",), ("resp",)]))
    c.append(("pipelined-badline", [("send", req("BREW / HTTP/1.1") + req("GET /health HTTP/1.1")), ("resp",), ("resp",)]))
    c.append(("pipelined-after-404", [("send", req("GET /nope HTTP/1.1") + req("GET /health HTTP/1.1")), ("resp",), ("resp",)]))
    c.append(("keepalive-after-badheader", [("send", req("GET /echo HTTP/1.1", "NoColon")), ("resp",), ("send", req("GET /health HTTP/1.1")), ("resp",)]))
    c.append(("keepalive-after-500", [("send", req("GET /status?code=500&body=x HTTP/1.1")), ("resp",), ("send", req("GET /health HTTP/1.1")), ("resp",)]))
    # RFC 9112 2.2: one empty line before a request line is skipped
    c.append(simple("leading-crlf", b"\r\n" + req("GET /echo?lead=1 HTTP/1.1")))
    c.append(simple("leading-crlf-2", b"\r\n\r\n" + req("GET /echo?lead=2 HTTP/1.1")))
    c.append(("body-then-crlf", [("send", req("POST /echo HTTP/1.1", "Content-Length: 2", body=b"ok") + b"\r\n"), ("resp",),
                                 ("send", req("GET /echo?after=1 HTTP/1.1")), ("resp",)]))
    c.append(("leading-crlf-eof", [("send", b"\r\n"), ("shut",), ("eof",)]))
    # the request target is visible characters and obs-text only
    c.append(simple("target-tab", b"GET /ec\tho HTTP/1.1\r\n" + H + b"\r\n"))
    c.append(simple("target-ctl", b"GET /echo?a=\x01 HTTP/1.1\r\n" + H + b"\r\n"))
    c.append(simple("target-del", b"GET /echo\x7f HTTP/1.1\r\n" + H + b"\r\n"))
    c.append(simple("target-cr", b"GET /echo?a=\rb HTTP/1.1\r\n" + H + b"\r\n"))
    c.append(simple("target-obs-text", b"GET /echo?u=\xc3\xa9 HTTP/1.1\r\n" + H + b"\r\n"))
    # a Content-Length that is not a decimal length is a 400 on any route
    c.append(simple("cl-plus", req("GET /echo HTTP/1.1", "Content-Length: +2", body=b"ok")))
    c.append(simple("cl-list", req("POST /echo HTTP/1.1", "Content-Length: 2, 2", body=b"ok")))
    c.append(simple("cl-empty", req("GET /health HTTP/1.1", "Content-Length:")))
    c.append(simple("cl-space-inside", req("GET /health HTTP/1.1", "Content-Length: 1 2")))
    c.append(("cl-bad-keepalive", [("send", req("GET /health HTTP/1.1", "Content-Length: x")), ("resp",), ("send", req("GET /health HTTP/1.1")), ("resp",)]))
    # a whole q-value or none
    c.append(simple("accept-q-suffix", req("GET /echo HTTP/1.1", "Accept: text/html;q=1.0x")))
    c.append(simple("accept-q-exp", req("GET /echo HTTP/1.1", "Accept: text/html;q=0.5e1")))
    c.append(simple("accept-q-ok", req("GET /echo HTTP/1.1", "Accept: text/html;q=0.500, */*;q=0")))
    # a body from CPPHTTPLIB_SEND_BUFSIZ is written apart from the head
    c.append(simple("echo-large", post("/echo", b"L" * 20000, ct="text/plain")))
    c.append(simple("echo-16383", post("/echo", b"M" * 16300, ct="text/plain")))
    c.append(("echo-large-keepalive", [("send", post("/echo", b"K" * 30000, ct="text/plain")), ("resp",), ("send", req("GET /health HTTP/1.1")), ("resp",)]))
    c.append(("echo-large-head", [("send", b"HEAD /status?body=" + b"h" * 17000 + b" HTTP/1.1\r\n" + H + b"\r\n"), ("head",),
                                  ("send", req("GET /health HTTP/1.1")), ("resp",)]))
    # server-http.cpp v0.6.0: with the UI on, no /sw.js route
    c.append(simple("sw-js-ui-on", req("GET /sw.js HTTP/1.1")))
    # cpp-httplib 0.60: more than one Host field line, or a Host value that
    # is not uri-host [":" port], is a 400 that closes the connection; it is
    # checked after the framing and before Accept and Range
    # ("get-header-cases" above sends a second, lower-case Host since then)
    c.append(simple("get-header-cases-1host", req("GET /echo HTTP/1.1", "X-Dup: 1", "x-dup: 2", "X-Empty:", "X-Spaces:   a b   ")))
    c.append(simple("host-dup", req("GET /echo HTTP/1.1", "Host: x")))
    c.append(simple("host-dup-case", req("GET /echo HTTP/1.1", "HOST: y")))
    c.append(simple("host-dup-empty", req("GET /echo HTTP/1.1", "Host:")))
    c.append(simple("host-none", b"GET /echo HTTP/1.1\r\nX-A: 1\r\n\r\n"))
    c.append(simple("host-empty", b"GET /echo HTTP/1.1\r\nHost:\r\n\r\n"))
    c.append(simple("host-http10-none", b"GET /echo HTTP/1.0\r\n\r\n"))
    c.append(simple("host-all-ok", b"GET /echo HTTP/1.1\r\nHost: aZ09-._~!$&'()*+,;=%:[]\r\n\r\n"))
    c.append(simple("host-v6-port", b"GET /echo HTTP/1.1\r\nHost: [::1]:8080\r\n\r\n"))
    for name, v in [("slash", b"x/y"), ("at", b"user@x"), ("space", b"a b"), ("quote", b'"x"'), ("hash", b"x#y"),
                    ("question", b"x?y"), ("backslash", b"x\\y"), ("lt", b"<x>"), ("caret", b"x^y"), ("brace", b"{x}"),
                    ("pipe", b"x|y"), ("backtick", b"x`y"), ("obs-text", b"caf\xc3\xa9")]:
        c.append(simple("host-bad-" + name, b"GET /echo HTTP/1.1\r\nHost: " + v + b"\r\n\r\n"))
    c.append(("host-dup-keepalive", [("send", req("GET /health HTTP/1.1", "Host: x")), ("resp",), ("send", req("GET /health HTTP/1.1")), ("resp",)]))
    c.append(("host-dup-pipelined", [("send", req("GET /health HTTP/1.1", "Host: x") + req("GET /health HTTP/1.1")), ("resp",), ("resp",)]))
    c.append(simple("host-dup-range-bad", req("GET /echo HTTP/1.1", "Host: x", "Range: items=0-1")))
    c.append(simple("host-bad-cl-bad", b"GET /echo HTTP/1.1\r\nHost: x/y\r\nContent-Length: +1\r\n\r\nx"))
    c.append(simple("host-dup-expect", req("POST /echo HTTP/1.1", "Host: x", "Content-Length: 4", "Expect: 100-continue")))
    c.append(simple("host-dup-websocket", req("GET /echo HTTP/1.1", "Host: y", "Upgrade: websocket", "Connection: Upgrade",
                                              "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==", "Sec-WebSocket-Version: 13")))
    # cpp-httplib 0.61: an Accept quoted-string is dropped before parsing, a
    # quoted-pair with it, an unclosed one to the end
    c.append(simple("accept-quoted-comma", req("GET /echo HTTP/1.1", 'Accept: text/html;x="a,b", application/json')))
    c.append(simple("accept-quoted-semi", req("GET /echo HTTP/1.1", 'Accept: text/html;x="a;q=7";q=0.5')))
    c.append(simple("accept-quoted-pair", req("GET /echo HTTP/1.1", 'Accept: text/html;x="a\\",foo", */*')))
    c.append(simple("accept-quoted-unclosed", req("GET /echo HTTP/1.1", 'Accept: text/html;x="abc, foo')))
    c.append(simple("accept-quoted-only", req("GET /echo HTTP/1.1", 'Accept: "foo"')))
    c.append(simple("accept-quoted-bad-after", req("GET /echo HTTP/1.1", 'Accept: text/html;x="y", foo')))
    c.append(simple("accept-quoted-two-lines", req("GET /echo HTTP/1.1", 'Accept: text/html;x="a', 'Accept: b", foo')))
    c.append(simple("accept-quoted-mediatype", req("GET /echo HTTP/1.1", 'Accept: te"x,y"xt/html')))
    # cpp-httplib 0.60: the range unit is case-insensitive, and a suffix
    # longer than the representation selects all of it
    c.append(simple("range-unit-upper", req("GET /status?code=206&body=0123456789 HTTP/1.1", "Range: BYTES=2-5")))
    c.append(simple("range-unit-mixed", req("GET /status?code=206&body=0123456789 HTTP/1.1", "Range: Bytes=-3")))
    c.append(simple("range-unit-alt", req("GET /status?code=206&body=0123456789&ct=text/x HTTP/1.1", "Range: bYtEs=0-1,7-")))
    c.append(simple("range-unit-short", req("GET /status?code=206&body=0123456789 HTTP/1.1", "Range: BYTES=")))
    c.append(simple("range-suffix-long", req("GET /status?code=206&body=0123456789 HTTP/1.1", "Range: bytes=-20")))
    c.append(simple("range-suffix-exact", req("GET /status?code=206&body=0123456789 HTTP/1.1", "Range: bytes=-10")))
    c.append(simple("range-suffix-11", req("GET /status?code=206&body=0123456789 HTTP/1.1", "Range: bytes=-11")))
    c.append(simple("range-suffix-zero", req("GET /status?code=206&body=0123456789 HTTP/1.1", "Range: bytes=-0")))
    c.append(simple("range-suffix-long-multi", req("GET /status?code=206&body=0123456789&ct=text/x HTTP/1.1", "Range: bytes=-99, 0-1")))
    c.append(simple("range-suffix-long-200", req("GET /echo HTTP/1.1", "Range: bytes=-100000")))
    c.append(simple("range-suffix-huge", req("GET /status?code=206&body=0123456789 HTTP/1.1", "Range: bytes=-99999999999999999999")))
    return c


# cpp-httplib 0.62: what the request line, the framing and the Host rules
# reject or close, what a bodyless status keeps, and the websocket route
# v0.8.0's /v1/realtime stands on.
def v062_cases():
    c = []
    c.append(simple("rl-two-spaces", b"GET  /echo HTTP/1.1\r\n" + H + b"\r\n"))
    c.append(simple("rl-trailing-space", b"GET /echo HTTP/1.1 \r\n" + H + b"\r\n"))
    c.append(simple("rl-leading-space", b" GET /echo HTTP/1.1\r\n" + H + b"\r\n"))
    c.append(simple("rl-tab", b"GET\t/echo HTTP/1.1\r\n" + H + b"\r\n"))
    c.append(simple("rl-no-version", b"GET /echo \r\n" + H + b"\r\n"))
    c.append(simple("rl-fragment", b"GET /echo#f HTTP/1.1\r\n" + H + b"\r\n"))
    c.append(simple("rl-fragment-q", b"GET /echo?a=1#f HTTP/1.1\r\n" + H + b"\r\n"))
    c.append(simple("host-none-11", b"GET /echo HTTP/1.1\r\n\r\n"))
    c.append(simple("host-none-10", b"GET /echo HTTP/1.0\r\n\r\n"))
    c.append(simple("host-two", b"GET /echo HTTP/1.1\r\nHost: a\r\nHost: b\r\n\r\n"))
    c.append(simple("te-with-cl", req("POST /echo HTTP/1.1", "Transfer-Encoding: chunked", "Content-Length: 0", body=b"2\r\nhi\r\n0\r\n\r\n"),
                    ("send", req("GET /health HTTP/1.1")), ("resp",)))
    c.append(simple("te-http10", b"POST /echo HTTP/1.0\r\n" + H + b"Transfer-Encoding: chunked\r\n\r\n2\r\nhi\r\n0\r\n\r\n",
                    ("send", req("GET /health HTTP/1.1")), ("resp",)))
    c.append(simple("prohibited-header-lower", req("GET /echo HTTP/1.1", "remote_addr: 1.2.3.4", "Local_Port: 1")))
    c.append(simple("status-204-body", req("GET /status?code=204&body=abc HTTP/1.1"), ("send", req("GET /health HTTP/1.1")), ("resp",)))
    c.append(simple("status-304-body", req("GET /status?code=304&body=abc HTTP/1.1"), ("send", req("GET /health HTTP/1.1")), ("resp",)))
    c.append(simple("status-204-cl", req("GET /status?code=204&h_Content-Length=9 HTTP/1.1")))
    c.append(simple("status-handler-cl", req("GET /status?code=200&body=abcd&h_Content-Length=2 HTTP/1.1")))
    c.append(simple("status-handler-te", req("GET /status?code=200&body=abcd&h_Transfer-Encoding=chunked HTTP/1.1")))
    c.append(simple("head-echo", req("HEAD /echo HTTP/1.1")))
    c.append(simple("head-status-cl", req("HEAD /status?h_Content-Length=7 HTTP/1.1")))
    c.append(simple("head-stream", req("HEAD /stream HTTP/1.1")))
    c.append(simple("multipart-upper", post("/echo", mp(b"BB", [([b'Content-Disposition: form-data; name="a"'], b"1")]), ct="MULTIPART/Form-Data; BOUNDARY=BB")))
    c.append(simple("multipart-quoted", post("/echo", mp(b"BB", [([b'Content-Disposition: form-data; name="a\\"b"; filename="c:\\x.txt"'], b"1")]), ct='multipart/form-data; boundary="BB"')))
    c.append(simple("form-upper", post("/echo", b"a=1", ct="APPLICATION/X-WWW-FORM-URLENCODED")))
    c.append(simple("path-bad-u", req("GET /%uD800cho HTTP/1.1")))

    # websockets: the upgrade
    c.append(simple("ws-echo-open", ws_upgrade("/ws/one?x=1&y=two", "X-T: v"), ("wsframe",),
                    ("send", ws_frame(1, b"hello")), ("wsframe",),
                    ("send", ws_frame(2, b"\x00\xff")), ("wsframe",),
                    ("send", ws_frame(1, b"binary:xy")), ("wsframe",),
                    ("send", ws_frame(1, b"open")), ("wsframe",),
                    ("send", ws_frame(8, b"\x03\xe8bye")), ("wsframe",)))
    c.append(simple("ws-server-close", ws_upgrade("/ws/a"), ("wsframe",), ("send", ws_frame(1, b"close")), ("wsframe",),
                    ("send", ws_frame(8, b"\x03\xe8"))))
    c.append(simple("ws-empty", ws_upgrade("/ws/a"), ("wsframe",), ("send", ws_frame(1, b"")), ("wsframe",), ("send", ws_frame(8, b"")), ("wsframe",)))
    c.append(simple("ws-len-126", ws_upgrade("/ws/a"), ("wsframe",), ("send", ws_frame(1, b"a" * 300)), ("wsframe",), ("send", ws_frame(8, b"")), ("wsframe",)))
    c.append(simple("ws-len-127", ws_upgrade("/ws/a"), ("wsframe",), ("send", ws_frame(2, b"b" * 70000)), ("wsframe",), ("send", ws_frame(8, b"")), ("wsframe",)))
    c.append(simple("ws-ping", ws_upgrade("/ws/a"), ("wsframe",), ("send", ws_frame(9, b"pp")), ("wsframe",), ("send", ws_frame(10, b"")), ("send", ws_frame(8, b"")), ("wsframe",)))
    c.append(simple("ws-fragments", ws_upgrade("/ws/a"), ("wsframe",),
                    ("send", ws_frame(1, b"ab", fin=False)), ("send", ws_frame(9, b"m")), ("wsframe",),
                    ("send", ws_frame(0, b"cd", fin=False)), ("send", ws_frame(0, b"ef")), ("wsframe",),
                    ("send", ws_frame(8, b"")), ("wsframe",)))
    c.append(simple("ws-frag-bad-next", ws_upgrade("/ws/a"), ("wsframe",), ("send", ws_frame(1, b"ab", fin=False)), ("send", ws_frame(1, b"cd"))))
    c.append(simple("ws-frag-close", ws_upgrade("/ws/a"), ("wsframe",), ("send", ws_frame(1, b"ab", fin=False)), ("send", ws_frame(8, b"\x03\xe8")), ("wsframe",)))
    c.append(simple("ws-bad-utf8", ws_upgrade("/ws/a"), ("wsframe",), ("send", ws_frame(1, b"\xff\xfe")), ("wsframe",), ("send", ws_frame(8, b"\x03\xef"))))
    c.append(simple("ws-rsv", ws_upgrade("/ws/a"), ("wsframe",), ("send", b"\xc1\x80\x00\x00\x00\x00")))
    c.append(simple("ws-unmasked", ws_upgrade("/ws/a"), ("wsframe",), ("send", b"\x81\x02hi")))
    c.append(simple("ws-frag-control", ws_upgrade("/ws/a"), ("wsframe",), ("send", ws_frame(9, b"x", fin=False))))
    c.append(simple("ws-long-control", ws_upgrade("/ws/a"), ("wsframe",), ("send", ws_frame(9, b"x" * 126))))
    c.append(simple("ws-bad-opcode", ws_upgrade("/ws/a"), ("wsframe",), ("send", ws_frame(3, b"x"))))
    c.append(simple("ws-too-long", ws_upgrade("/ws/a"), ("wsframe",), ("send", b"\x82\xff\x00\x00\x00\x00\x01\x00\x00\x01")))
    c.append(simple("ws-msb-len", ws_upgrade("/ws/a"), ("wsframe",), ("send", b"\x82\xff\x80\x00\x00\x00\x00\x00\x00\x01")))
    c.append(simple("ws-close-bad-code", ws_upgrade("/ws/a"), ("wsframe",), ("send", ws_frame(8, b"\x03\xed")), ("wsframe",)))
    c.append(simple("ws-close-short", ws_upgrade("/ws/a"), ("wsframe",), ("send", ws_frame(8, b"\x03")), ("wsframe",)))
    c.append(simple("ws-close-bad-reason", ws_upgrade("/ws/a"), ("wsframe",), ("send", ws_frame(8, b"\x03\xe8\xff")), ("wsframe",)))
    c.append(simple("ws-hangup", ws_upgrade("/ws/a"), ("wsframe",)))
    # websockets: what is not an upgrade, or finds no route
    c.append(simple("ws-no-route", ws_upgrade("/nope")))
    c.append(simple("ws-plain-route", ws_upgrade("/echo")))
    c.append(simple("ws-bad-version", ws_upgrade("/ws/a", version="12")))
    c.append(simple("ws-bad-key", ws_upgrade("/ws/a", key="short")))
    c.append(simple("ws-get-without-upgrade", req("GET /ws/a HTTP/1.1")))
    c.append(simple("ws-post", ws_upgrade("/ws/a", method="POST")))
    return c


def chunks(n, ext=b"", zeros=0, data=b"a"):
    """n one-byte chunks (`data`), each size line padded with `zeros` leading
    zeros and followed by `ext`, then the last chunk."""
    one = b"0" * zeros + b"%x" % len(data) + ext + b"\r\n" + data + b"\r\n"
    return one * n + b"0\r\n\r\n"


# cpp-httplib 0.64: what a request may no longer do -- a transfer coding but
# a lone "chunked" is a 501, a Range on anything but GET and HEAD is not
# read, an unread body is drained only up to 256 KiB, a request without a
# length never has a body, and the bytes around a chunk's data are bounded
def v064_cases():
    c = []
    hz = req("GET /health HTTP/1.1")
    # RFC 9112 6.1: a lone "chunked", in any case, is the only coding
    c.append(simple("te-chunked-mixed-case", req("POST /echo HTTP/1.1", "Transfer-Encoding: ChUnKeD", body=b"2\r\nhi\r\n0\r\n\r\n"),
                    ("send", hz), ("resp",)))
    c.append(simple("te-chunked-twice-lines", req("POST /echo HTTP/1.1", "Transfer-Encoding: chunked", "Transfer-Encoding: chunked",
                                                  body=b"2\r\nhi\r\n0\r\n\r\n"), ("send", hz), ("resp",)))
    c.append(simple("te-chunked-twice-list", req("POST /echo HTTP/1.1", "Transfer-Encoding: chunked, chunked", body=b"2\r\nhi\r\n0\r\n\r\n"),
                    ("send", hz), ("resp",)))
    c.append(simple("te-gzip-chunked", req("POST /echo HTTP/1.1", "Transfer-Encoding: gzip, chunked", body=b"2\r\nhi\r\n0\r\n\r\n")))
    c.append(simple("te-identity-chunked-get", req("GET /health HTTP/1.1", "Transfer-Encoding: identity,chunked", body=b"0\r\n\r\n")))
    c.append(simple("te-chunked-space", req("POST /echo HTTP/1.1", "Transfer-Encoding:  chunked ", body=b"2\r\nhi\r\n0\r\n\r\n")))
    c.append(simple("te-chunked-and-cl-501", req("POST /echo HTTP/1.1", "Transfer-Encoding: x, chunked", "Content-Length: 0",
                                                 body=b"0\r\n\r\n")))
    c.append(simple("te-websocket", ws_upgrade("/ws/a", "Transfer-Encoding: deflate, chunked")))
    # RFC 9110 14.2: a Range is read for GET and HEAD only
    c.append(simple("range-post-bad", post("/echo", b"0123456789", "Range: items=0-1", ct="text/plain")))
    c.append(simple("range-post-ok", post("/echo", b"0123456789", "Range: bytes=0-1", ct="text/plain")))
    c.append(simple("range-delete-bad", req("DELETE /echo HTTP/1.1", "Range: bytes=9-1")))
    c.append(simple("range-head-bad", req("HEAD /echo HTTP/1.1", "Range: bytes=9-1")))
    c.append(simple("range-get-bad-still", req("GET /echo HTTP/1.1", "Range: bytes=9-1")))
    # an unread body is skipped up to CPPHTTPLIB_UNREAD_PAYLOAD_MAX_LENGTH: the
    # middleware answers OPTIONS before routing reads any body
    limit = 256 * 1024
    for name, n in (("at-limit", limit), ("over-limit", limit + 1), ("small", 3)):
        c.append(("unread-body-" + name, [("send", req("OPTIONS /echo HTTP/1.1", "Content-Length: %d" % n, body=b"u" * n)), ("resp",),
                                          ("send", hz), ("resp",)]))
        c.append(("unread-chunked-" + name, [("send", req("OPTIONS /echo HTTP/1.1", "Transfer-Encoding: chunked",
                                                          body=b"%x\r\n" % n + b"c" * n + b"\r\n0\r\n\r\n")), ("resp",),
                                             ("send", hz), ("resp",)]))
    c.append(("unread-chunks-over-limit", [("send", req("OPTIONS /echo HTTP/1.1", "Transfer-Encoding: chunked",
                                                        body=(b"8000\r\n" + b"k" * 0x8000 + b"\r\n") * 8 + b"1\r\nk\r\n0\r\n\r\n")),
                                           ("resp",), ("send", hz), ("resp",)]))
    c.append(("unread-body-404-over", [("send", req("POST /nope HTTP/1.1", "Content-Length: %d" % (limit + 1), body=b"u" * (limit + 1))),
                                       ("resp",), ("send", hz), ("resp",)]))
    # RFC 9112 6: no length, no body, whether or not the connection persists
    c.append(simple("nolength-http10", b"POST /echo HTTP/1.0\r\n" + H + b"\r\nstray-bytes"))
    c.append(simple("nolength-close-big", req("POST /echo HTTP/1.1", "Connection: close", body=b"z" * 70000)))
    # the bytes around a chunk's data: 16 a chunk and twice its data are
    # free, past that they add up to one header field (8192)
    c.append(simple("chunk-overhead-170", req("POST /echo HTTP/1.1", "Transfer-Encoding: chunked", "Content-Type: text/plain",
                                              body=chunks(170, ext=b";" + b"x" * 60)), ("send", hz), ("resp",)))
    c.append(simple("chunk-overhead-171", req("POST /echo HTTP/1.1", "Transfer-Encoding: chunked", "Content-Type: text/plain",
                                              body=chunks(171, ext=b";" + b"x" * 60)), ("send", hz), ("resp",)))
    c.append(simple("chunk-overhead-zeros", req("POST /echo HTTP/1.1", "Transfer-Encoding: chunked", "Content-Type: text/plain",
                                                body=chunks(171, zeros=61))))
    c.append(simple("chunk-overhead-big-data", req("POST /echo HTTP/1.1", "Transfer-Encoding: chunked", "Content-Type: text/plain",
                                                   body=chunks(400, ext=b";" + b"x" * 60, data=b"d" * 40))))
    c.append(simple("chunk-overhead-unread", req("GET /health HTTP/1.1", "Transfer-Encoding: chunked", body=chunks(171, ext=b";" + b"x" * 60)),
                    ("send", hz), ("resp",)))
    return c


WS_KEY = "dGhlIHNhbXBsZSBub25jZQ=="


def ws_upgrade(path, *headers, key=WS_KEY, version="13", method="GET"):
    return req("%s %s HTTP/1.1" % (method, path), "Upgrade: websocket", "Connection: Upgrade",
               "Sec-WebSocket-Key: " + key, "Sec-WebSocket-Version: " + version, *headers)


def ws_frame(opcode, data, fin=True):
    """A client frame, masked with a fixed key."""
    out = bytes([(0x80 if fin else 0) | opcode])
    n = len(data)
    if n < 126:
        out += bytes([0x80 | n])
    elif n <= 0xFFFF:
        out += bytes([0x80 | 126]) + n.to_bytes(2, "big")
    else:
        out += bytes([0x80 | 127]) + n.to_bytes(8, "big")
    mask = b"\x11\x22\x33\x44"
    return out + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(data))


def ws_key_cases():
    c = []
    # middleware runs before the switch
    c.append(simple("ws-nokey", ws_upgrade("/ws/a")))
    c.append(simple("ws-key", ws_upgrade("/ws/a", "Authorization: Bearer k2"), ("wsframe",), ("send", ws_frame(8, b"")), ("wsframe",)))
    # cpp-httplib 0.64: an upgrade answered before it, with its body unread,
    # closes the connection, which would read that body as the next request
    hz = req("GET /health HTTP/1.1")
    c.append(simple("ws-nokey-body", ws_upgrade("/ws/a", "Content-Length: 4") + b"abcd", ("send", hz), ("resp",)))
    c.append(simple("ws-nokey-body-request", ws_upgrade("/ws/a", "Content-Length: %d" % len(hz)) + hz, ("resp",)))
    c.append(simple("ws-nokey-chunked", ws_upgrade("/ws/a", "Transfer-Encoding: chunked") + b"0\r\n\r\n", ("send", hz), ("resp",)))
    c.append(simple("ws-nokey-cl0", ws_upgrade("/ws/a", "Content-Length: 0"), ("send", hz), ("resp",)))
    c.append(simple("ws-nokey-nobody", ws_upgrade("/ws/a"), ("send", hz), ("resp",)))
    c.append(simple("ws-noroute-body", ws_upgrade("/nope", "Authorization: Bearer k2", "Content-Length: 4") + b"abcd",
                    ("send", hz), ("resp",)))
    return c


def key_cases():
    c = []
    c.extend(ws_key_cases())
    c.append(simple("nokey", req("GET /echo HTTP/1.1")))
    # cpp-httplib 0.57: a request the middleware rejects is never sent `100 Continue`
    c.append(simple("expect-100-nokey", req("POST /echo HTTP/1.1", "Content-Length: 4", "Expect: 100-continue")))
    c.append(simple("bearer", req("GET /echo HTTP/1.1", "Authorization: Bearer sk-test-1")))
    c.append(simple("bare-key", req("GET /echo HTTP/1.1", "Authorization: k2")))
    c.append(simple("x-api-key", req("GET /echo HTTP/1.1", "X-Api-Key: k2")))
    c.append(simple("bearer-lower", req("GET /echo HTTP/1.1", "Authorization: bearer k2")))
    c.append(simple("auth-wins", req("GET /echo HTTP/1.1", "Authorization: Bearer nope", "X-Api-Key: k2")))
    c.append(simple("key-dup", req("GET /echo HTTP/1.1", "Authorization: nope", "Authorization: k2")))
    c.append(simple("key-empty-auth", req("GET /echo HTTP/1.1", "Authorization:", "X-Api-Key: k2")))
    c.append(simple("health-public", req("GET /health HTTP/1.1")))
    c.append(simple("v1-health-public", req("GET /v1/health HTTP/1.1")))
    c.append(simple("root-public", req("GET / HTTP/1.1")))
    c.append(simple("health-query", req("GET /health?x=1 HTTP/1.1")))
    c.append(simple("options-nokey", req("OPTIONS /echo HTTP/1.1", "Origin: http://localhost:1")))
    c.append(simple("origin-localhost", req("GET /health HTTP/1.1", "Origin: http://localhost:3000")))
    c.append(simple("origin-127", req("GET /health HTTP/1.1", "Origin: https://127.0.0.1")))
    c.append(simple("origin-v6", req("GET /health HTTP/1.1", "Origin: http://[::1]:8080/x")))
    c.append(simple("origin-user", req("GET /health HTTP/1.1", "Origin: http://evil@localhost")))
    c.append(simple("origin-evil", req("GET /health HTTP/1.1", "Origin: https://evil.example")))
    c.append(simple("origin-notaurl", req("GET /health HTTP/1.1", "Origin: localhost")))
    c.append(simple("origin-badport", req("GET /health HTTP/1.1", "Origin: http://localhost:x")))
    c.append(simple("origin-bigport", req("GET /health HTTP/1.1", "Origin: http://localhost:99999999999")))
    c.append(simple("origin-unclosed", req("GET /health HTTP/1.1", "Origin: http://[::1")))
    c.append(simple("post-nokey-body", post("/echo", b"secret")))
    c.append(("keepalive-401", [("send", req("GET /echo HTTP/1.1")), ("resp",), ("send", req("GET /health HTTP/1.1")), ("resp",)]))
    c.append(simple("stream-key", req("GET /stream HTTP/1.1", "Authorization: Bearer k2")))
    # cpp-httplib 0.60: a duplicate Host is refused before the middleware
    c.append(simple("host-dup-nokey", req("GET /echo HTTP/1.1", "Host: x")))
    c.append(simple("host-bad-key", b"GET /echo HTTP/1.1\r\nHost: a/b\r\nAuthorization: Bearer k2\r\n\r\n"))
    return c


def prefix_cases():
    c = []
    k = "Authorization: Bearer abc"
    c.append(simple("prefixed", req("GET /api/echo?z=1 HTTP/1.1", k)))
    c.append(simple("prefixed-nokey", req("GET /api/echo HTTP/1.1")))
    c.append(simple("unprefixed", req("GET /echo HTTP/1.1", k)))
    c.append(simple("unprefixed-nokey", req("GET /echo HTTP/1.1")))
    c.append(simple("prefixed-health-nokey", req("GET /api/health HTTP/1.1")))
    c.append(simple("unprefixed-health", req("GET /health HTTP/1.1")))
    c.append(simple("prefixed-slots", post("/api/slots/7", b"{}", k)))
    c.append(simple("prefixed-two", req("GET /api/two/a/x/b HTTP/1.1", k)))
    c.append(simple("api-root", req("GET /api HTTP/1.1", k)))
    c.append(simple("options", req("OPTIONS /api/echo HTTP/1.1", "Origin: http://x")))
    c.append(simple("cors-fixed", req("GET /api/health HTTP/1.1", k, "Origin: http://x")))
    c.append(simple("sw-js-prefixed", req("GET /api/sw.js HTTP/1.1", k)))
    c.append(("prefixed-stream", [("send", req("GET /api/stream HTTP/1.1", k)), ("resp",)]))
    return c


def small_cases():
    c = []
    c.append(simple("echo", req("GET /echo?a=b HTTP/1.1", "Origin: http://o")))
    c.append(simple("post", post("/echo", b"x", "Authorization: Bearer abc")))
    c.append(simple("health", req("GET /health HTTP/1.1")))
    c.append(("stream", [("send", req("GET /stream HTTP/1.1", "Authorization: abc")), ("resp",)]))
    # server-http.cpp v0.6.0: with the UI off ("auto"), /sw.js is a worker
    # that removes itself; "reuse" has it on, and an API key
    c.append(simple("sw-js", req("GET /sw.js HTTP/1.1", "Authorization: Bearer abc")))
    c.append(simple("sw-js-nokey", req("GET /sw.js HTTP/1.1")))
    c.append(simple("sw-js-post", post("/sw.js", b"{}", "Authorization: Bearer abc")))
    c.append(("sw-js-head", [("send", req("HEAD /sw.js HTTP/1.1", "Authorization: Bearer abc")), ("head",)]))
    return c


def notready_cases():
    c = []
    c.append(simple("echo", req("GET /echo HTTP/1.1")))
    c.append(simple("health", req("GET /health HTTP/1.1")))
    c.append(simple("root", req("GET / HTTP/1.1")))
    c.append(simple("options", req("OPTIONS /echo HTTP/1.1")))
    c.append(simple("quit-refused", req("GET /quit HTTP/1.1")))
    c.append(simple("bad-line", req("BREW / HTTP/1.1")))
    # cpp-httplib 0.64: the 503 an upgrade gets keeps its body unread
    c.append(simple("ws-body", ws_upgrade("/ws/a", "Content-Length: 4") + b"abcd", ("send", req("GET /health HTTP/1.1")), ("resp",)))
    c.append(simple("ws-nobody", ws_upgrade("/ws/a"), ("send", req("GET /health HTTP/1.1")), ("resp",)))
    return c


def gcp_cases():
    c = []

    def predict(body):
        return post("/predict-x", body)

    c.append(simple("hc", req("GET /hc HTTP/1.1")))
    c.append(simple("health", req("GET /health HTTP/1.1")))
    c.append(simple("predict", predict(b'{"instances":[{"@requestFormat":"health"},{"@requestFormat":"/echo","k":[1,2]},'
                                       b'{"@requestFormat":"echo","stream":true,"x":"y"},{"@requestFormat":"hc"},{"@requestFormat":"slots"},'
                                       b'{"@requestFormat":"two"},{"@requestFormat":"status"},{"@requestFormat":"empty"}]}')))
    c.append(simple("predict-errors", predict(b'{"instances":[1,{"x":1},{"@requestFormat":2},{"@requestFormat":"nowhere"},'
                                              b'{"@requestFormat":"stream"},{"@requestFormat":"/throw-raw"}]}')))
    c.append(simple("predict-throw-invalid", predict(b'{"instances":[{"@requestFormat":"throwInvalid"}]}')))
    c.append(simple("predict-throw-unknown", predict(b'{"instances":[{"@requestFormat":"throwUnknown"}]}')))
    c.append(simple("predict-empty", predict(b'{"instances":[]}')))
    c.append(simple("predict-notjson", predict(b'{"instances":')))
    c.append(simple("predict-array", predict(b"[1,2]")))
    c.append(simple("predict-noinstances", predict(b'{"x":1}')))
    c.append(simple("predict-instances-obj", predict(b'{"instances":{}}')))
    c.append(simple("predict-129", predict(b'{"instances":[' + b",".join([b"1"] * 129) + b"]}")))
    c.append(simple("predict-128", predict(b'{"instances":[' + b",".join([b"1"] * 128) + b"]}")))
    c.append(simple("predict-query", post("/predict-x?q=1", b'{"instances":[{"@requestFormat":"echo"}]}', "X-H: 1")))
    return c


def multi_cases():
    # v0.5.0: every listener has every route; "sock-" cases go to the Unix
    # socket, the others to the TCP address
    c = []
    c.append(simple("echo", req("GET /echo?a=b HTTP/1.1", "Authorization: Bearer abc")))
    c.append(simple("post-nokey", post("/echo", b"x")))
    c.append(simple("health", req("GET /health HTTP/1.1")))
    c.append(simple("sock-echo", req("GET /echo?s=1 HTTP/1.1", "X-Api-Key: abc")))
    c.append(simple("sock-post", post("/echo", b"y", "Authorization: Bearer abc")))
    c.append(simple("sock-404", req("GET /nope HTTP/1.1", "X-Api-Key: abc")))
    c.append(("sock-stream", [("send", req("GET /stream HTTP/1.1", "X-Api-Key: abc")), ("resp",)]))
    return c


def multitcp_cases():
    # "v6-" cases go to ::1, the others to 127.0.0.1, on the same port
    c = []
    c.append(simple("echo", req("GET /echo?a=b HTTP/1.1")))
    c.append(simple("v6-echo", req("GET /echo?a=c HTTP/1.1")))
    c.append(simple("v6-post", post("/echo", b"v6")))
    c.append(("v6-stream", [("send", req("GET /stream HTTP/1.1")), ("resp",)]))
    c.append(simple("post", post("/echo", b"v4")))
    return c


def unix_cases():
    c = []
    c.append(simple("echo", req("GET /echo?u=1 HTTP/1.1")))
    c.append(("stream", [("send", req("GET /stream HTTP/1.1")), ("resp",)]))
    return c


# name -> (environment, cases, how it stops)
CONFIGS = [
    ("default", {}, base_cases),
    ("v062", {}, v062_cases),
    ("v064", {}, v064_cases),
    ("apikey", {}, key_cases),
    ("prefix", {}, prefix_cases),
    ("auto", {}, small_cases),
    ("reuse", {}, small_cases),
    ("notready", {}, notready_cases),
    ("gcp", {"AIP_MODE": "PREDICTION", "AIP_HTTP_PORT": "FREE", "AIP_HEALTH_ROUTE": "hc", "AIP_PREDICT_ROUTE": "/predict-x"}, gcp_cases),
    ("gcpconflict", {"AIP_MODE": "PREDICTION", "AIP_HTTP_PORT": "0", "AIP_PREDICT_ROUTE": "echo"}, None),
    ("unix", {}, unix_cases),
    ("ssl", {}, None),
    ("badhost", {}, None),
    ("multi", {}, multi_cases),
    ("multitcp", {"HTTP_DIFF_PORT": "FREE"}, multitcp_cases),
    ("multiport0", {}, None),
    ("multibad", {"HTTP_DIFF_PORT": "FREE"}, None),
]

# -----------------------------------------------------------------------------
# Running

BOUNDARY = re.compile(rb"--cpp-httplib-multipart-data-[0-9A-Za-z]{16}")


def mask_bytes(b):
    return BOUNDARY.sub(b"--cpp-httplib-multipart-data-XXXXXXXXXXXXXXXX", b)


def run_side(binary, config, env_extra, cases, tmp, side, only_case):
    port_file = os.path.join(tmp, "%s-%s.port" % (config, side))
    # relative, in `tmp`, the working directory: sun_path holds 104 bytes
    sock_path = "%s.sock" % side
    for p in (port_file, port_file + ".quit", sock_path):
        if os.path.exists(p):
            os.unlink(p)
    env = dict(os.environ)
    env.update(env_extra)
    env["HTTP_DIFF_SOCK"] = sock_path
    out = open(os.path.join(tmp, "%s-%s.out" % (config, side)), "wb")
    err = open(os.path.join(tmp, "%s-%s.err" % (config, side)), "wb")
    proc = subprocess.Popen([binary, config, port_file], stdout=out, stderr=err, env=env, cwd=tmp)
    results = []
    deadline = time.time() + 60
    while not os.path.exists(port_file) or not open(port_file).read().endswith("\n"):
        if proc.poll() is not None or time.time() > deadline:
            break
        time.sleep(0.02)
    if cases is not None and proc.poll() is None and os.path.exists(port_file):
        port = int(open(port_file).read())
        target = sock_path if config == "unix" else ("127.0.0.1", port)
        for name, steps in cases():
            if only_case and only_case not in name:
                continue
            if name.startswith("sock-"):
                conn = Conn(sock_path)
            elif name.startswith("v6-"):
                conn = Conn(("::1", port))
            else:
                conn = Conn(target)
            conn.run(steps)
            results.append((name, mask_bytes(b"".join(conn.out))))
        with open(os.path.join(tmp, "%s-%s.transcript" % (config, side)), "wb") as f:
            for name, data in results:
                f.write(b"=== " + name.encode() + b"\n" + data + b"\n")
        # stop it: every connection has closed, so nothing is in flight
        open(port_file + ".quit", "w").close()
    try:
        rc = proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        rc = "TIMEOUT"
    out.close()
    err.close()
    logs = []
    for stream in ("out", "err"):
        text = open(os.path.join(tmp, "%s-%s.%s" % (config, side, stream)), "rb").read()
        text = re.sub(rb"(http://127\.0\.0\.1:)\d+", rb"\1PORT", text)
        text = text.replace(sock_path.encode(), b"SOCK")
        logs.append(text)
    return results, logs, rc


def main():
    args = sys.argv[1:]
    only = None
    keep = None
    only_case = None
    while len(args) > 2:
        if args[-2] == "--only":
            only = set(args[-1].split(","))
        elif args[-2] == "--keep":
            keep = args[-1]
        elif args[-2] == "--case":
            only_case = args[-1]
        else:
            break
        args = args[:-2]
    if len(args) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    ref_bin, port_bin = args
    ref_bin, port_bin = os.path.abspath(ref_bin), os.path.abspath(port_bin)
    tmp = os.path.abspath(keep or tempfile.mkdtemp(prefix="http-diff-"))
    os.makedirs(tmp, exist_ok=True)
    os.chdir(tmp)
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    failures = 0
    checks = 0
    t0 = time.time()
    for config, env, cases in CONFIGS:
        if only and config not in only:
            continue
        # one free port, the same for both sides, where a config names one
        env = dict(env)
        for k, v in env.items():
            if v == "FREE":
                probe = socket.socket()
                probe.bind(("127.0.0.1", 0))
                env[k] = str(probe.getsockname()[1])
                probe.close()
        ref = run_side(ref_bin, config, env, cases, tmp, "ref", only_case)
        port = run_side(port_bin, config, env, cases, tmp, "port", only_case)
        rres, rlogs, rrc = ref
        pres, plogs, prc = port
        bad = 0
        for (name, rb), (_, pb) in zip(rres, pres):
            checks += 1
            if rb != pb:
                bad += 1
                print("FAIL %s/%s" % (config, name))
                print("  ref : %r" % rb[:1500])
                print("  port: %r" % pb[:1500])
        if len(rres) != len(pres):
            bad += 1
            print("FAIL %s: %d cases on the reference, %d on the port" % (config, len(rres), len(pres)))
        for label, r, p in (("stdout", rlogs[0], plogs[0]), ("stderr", rlogs[1], plogs[1])):
            checks += 1
            if r != p:
                bad += 1
                print("FAIL %s: %s differs" % (config, label))
                rl, pl = r.splitlines(), p.splitlines()
                for i in range(max(len(rl), len(pl))):
                    a = rl[i] if i < len(rl) else b"<missing>"
                    b = pl[i] if i < len(pl) else b"<missing>"
                    if a != b:
                        print("  line %d\n    ref : %r\n    port: %r" % (i + 1, a, b))
                        break
        checks += 1
        if rrc != prc:
            bad += 1
            print("FAIL %s: exit code %r on the reference, %r on the port" % (config, rrc, prc))
        status = "ok" if bad == 0 else "%d FAILED" % bad
        print("%-12s %3d cases, logs, exit %s: %s" % (config, len(rres), rrc, status))
        failures += bad
    print("http-diff: %d checks, %d failed, %.0f s" % (checks, failures, time.time() - t0))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
