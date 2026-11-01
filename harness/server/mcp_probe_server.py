#!/usr/bin/env python3
"""An MCP server (NDJSON JSON-RPC 2.0 over stdio) for mcp.cases
(scripts/server-diff), beside upstream's own fixtures in
llama.cpp/tools/server/tests/fixtures/: it covers what those do not.

    mcp_probe_server.py [--tools NAME,...] [--init-error KIND] [--list-error KIND]
                        [--stderr TEXT] [--exit-after-init]

Every tool it lists answers by name:
    env       the value of MCP_PROBE (or "<unset>") and the working directory
    args      its arguments, as JSON text
    is_error  an isError result with two text parts
    is_error0 an isError result with no text part
    mixed     text, image and text parts
    nocontent a result with no content
    errstr    an error that is a bare string
    errobj    an error object without a message
    errnum    an error whose message is a number
    noresult  a reply with neither result nor error
    silent    no reply at all
    die       "dying", then a line on stderr, then exit
anything else echoes its name. --init-error and --list-error make
initialize or tools/list fail: "msg" an error object, "str" a string,
"none" a reply with no result. --tools sets the listed names; a name
ending in ":null" lists it with a null inputSchema, ":str" with a string
one, ":none" with none at all.
"""
import json
import os
import sys

args = sys.argv[1:]


def opt(name, default=None):
    if name in args:
        return args[args.index(name) + 1]
    return default


names = opt("--tools", "env,args,is_error,is_error0,mixed,nocontent,errstr,errobj,errnum,noresult,silent,die").split(",")


def schema_for(spec):
    if spec.endswith(":null"):
        return spec[:-5], {"inputSchema": None}
    if spec.endswith(":str"):
        return spec[:-4], {"inputSchema": "not an object"}
    if spec.endswith(":none"):
        return spec[:-5], {}
    return spec, {"inputSchema": {"type": "object", "properties": {"x": {"type": "string"}}}}


def failure(kind, req_id):
    if kind == "msg":
        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -1, "message": "refused by probe"}}
    if kind == "str":
        return {"jsonrpc": "2.0", "id": req_id, "error": "plain string error"}
    return {"jsonrpc": "2.0", "id": req_id}


def call(name, arguments, req_id):
    def ok(content):
        return {"jsonrpc": "2.0", "id": req_id, "result": {"content": content}}
    if name == "env":
        return ok([{"type": "text", "text": os.environ.get("MCP_PROBE", "<unset>")},
                   {"type": "text", "text": os.getcwd()}])
    if name == "args":
        return ok([{"type": "text", "text": json.dumps(arguments, sort_keys=True)}])
    if name == "is_error":
        return {"jsonrpc": "2.0", "id": req_id, "result": {"isError": True, "content": [
            {"type": "text", "text": "first"}, {"type": "text", "text": "second"}]}}
    if name == "is_error0":
        return {"jsonrpc": "2.0", "id": req_id, "result": {"isError": True, "content": [{"type": "image", "data": "x"}]}}
    if name == "mixed":
        return ok([{"type": "text", "text": "a"}, {"type": "image", "data": "x"}, "bare", {"type": "text", "text": "b"}])
    if name == "nocontent":
        return {"jsonrpc": "2.0", "id": req_id, "result": {}}
    if name == "errstr":
        return {"jsonrpc": "2.0", "id": req_id, "error": "bare string"}
    if name == "errobj":
        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": 5}}
    if name == "errnum":
        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": 5, "message": 42}}
    if name == "noresult":
        return {"jsonrpc": "2.0", "id": req_id}
    if name == "silent":
        return None
    if name == "die":
        return ok([{"type": "text", "text": "dying"}])
    return ok([{"type": "text", "text": name}])


def main():
    if opt("--stderr"):
        sys.stderr.write(opt("--stderr") + "\n")
        sys.stderr.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue
        method = request.get("method")
        req_id = request.get("id")
        params = request.get("params", {})
        if req_id is None:
            if method == "notifications/initialized" and "--exit-after-init" in args:
                sys.exit(0)
            continue
        if method == "initialize":
            if opt("--init-error"):
                response = failure(opt("--init-error"), req_id)
            else:
                response = {"jsonrpc": "2.0", "id": req_id, "result": {
                    "protocolVersion": "2024-11-05", "capabilities": {"tools": {}},
                    "serverInfo": {"name": "probe", "version": "1.0"}}}
        elif method == "tools/list":
            if opt("--list-error"):
                response = failure(opt("--list-error"), req_id)
            else:
                tools = []
                for spec in names:
                    name, extra = schema_for(spec)
                    tools.append({"name": name, "description": f"probe tool {name}", **extra})
                response = {"jsonrpc": "2.0", "id": req_id, "result": {"tools": tools}}
        elif method == "tools/call":
            response = call(params.get("name"), params.get("arguments", {}), req_id)
            if response is None:
                continue
            if params.get("name") == "die":
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()
                sys.stderr.write("probe died\n")
                sys.stderr.flush()
                os._exit(3)
        else:
            response = {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Method not found: {method}"}}
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
