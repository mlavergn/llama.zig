// http-diff's reference side: upstream's server_http_context, compiled
// unmodified (tools/server/server-http.cpp, server-common.cpp, and
// cpp-httplib with llama.cpp.zmake's httplib_defines), serving a fixed set
// of test routes. harness/http_diff.zig is the same program over the Zig
// port; scripts/http-diff drives both with the same raw bytes and compares
// what comes back. Not a port: test scaffolding.
//
// Usage: http_ref <config> <port-file>
//   The listening port is written to <port-file> once the server is up;
//   GET /quit stops it. Every line logged is compared by the driver.

#include "server-http.h"
#include "server-common.h"

#include "common.h"
#include "log.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

static std::atomic<bool> g_quit{false};
static std::atomic<bool> g_wait_complete{false};

// Copied from tools/server/server.cpp:54 @b29c606e2, unmodified: every
// route the server registers goes through it.
static server_http_context::handler_t ex_wrapper(server_http_context::handler_t func) {
    return [func = std::move(func)](const server_http_req & req) -> server_http_res_ptr {
        std::string message;
        error_type error;
        try {
            return func(req);
        } catch (const std::invalid_argument & e) {
            // treat invalid_argument as invalid request (400)
            error = ERROR_TYPE_INVALID_REQUEST;
            message = e.what();
        } catch (const std::exception & e) {
            // treat other exceptions as server error (500)
            error = ERROR_TYPE_SERVER;
            message = e.what();
        } catch (...) {
            error = ERROR_TYPE_SERVER;
            message = "unknown error";
        }

        auto res = std::make_unique<server_http_res>();
        res->status = 500;
        try {
            json error_data = format_error_response(message, error);
            res->status = json_value(error_data, "code", 500);
            res->data = safe_json_to_str({{ "error", error_data }});
            SRV_WRN("got exception: %s\n", res->data.c_str());
        } catch (const std::exception & e) {
            SRV_ERR("got another exception: %s | while handling exception: %s\n", e.what(), message.c_str());
            res->data = "Internal Server Error";
        }
        return res;
    };
}

static server_http_res_ptr json_res(const json & j) {
    auto res = std::make_unique<server_http_res>();
    res->data = safe_json_to_str(j);
    return res;
}

static json echo_json(const char * tag, const server_http_req & req) {
    json j = json::object();
    j["tag"] = tag;
    j["path"] = req.path;
    j["query_string"] = req.query_string;
    json p = json::object();
    for (const auto & [k, v] : req.params) {
        p[k] = v;
    }
    j["params"] = p;
    json h = json::object();
    for (const auto & [k, v] : req.headers) {
        h[k] = v;
    }
    j["headers"] = h;
    j["body"] = req.body;
    json f = json::object();
    for (const auto & [k, file] : req.files) {
        json o = json::object();
        o["filename"] = file.filename;
        o["content_type"] = file.content_type;
        o["size"] = (int64_t) file.data.size();
        o["data"] = std::string(file.data.begin(), file.data.end());
        f[k] = o;
    }
    j["files"] = f;
    return j;
}

static server_http_context::handler_t echo(const char * tag) {
    return [tag](const server_http_req & req) { return json_res(echo_json(tag, req)); };
}

struct wait_res : server_http_res {
    void on_complete() override {
        LOG_INF("harness: stream-wait complete\n");
        g_wait_complete = true;
    }
};

static void configure(const std::string & name, common_params & params, bool & ready) {
    params.hostname       = "127.0.0.1";
    params.port           = 0;
    params.timeout_read   = 3;
    params.timeout_write  = 3;
    params.n_threads_http = 2;
    ready = true;
    if (name == "apikey") {
        params.api_keys         = {"sk-test-1", "k2"};
        params.n_threads_http   = 1;
        params.cors_origins     = "localhost";
        params.cors_credentials = false;
    } else if (name == "prefix") {
        params.api_prefix       = "/api";
        params.api_keys         = {"abc"};
        params.n_threads_http   = 4;
        params.cors_origins     = "https://example.com";
        params.cors_credentials = false;
        params.cors_methods     = "GET";
        params.cors_headers     = "Authorization";
    } else if (name == "auto") {
        params.n_threads_http = -1;
        params.n_parallel     = 3;
        params.ui             = false;
    } else if (name == "notready") {
        ready = false;
    } else if (name == "unix") {
        const char * sock = std::getenv("HTTP_DIFF_SOCK");
        params.hostname = sock ? sock : "http-diff.sock";
    } else if (name == "ssl") {
        params.ssl_file_key  = "key.pem";
        params.ssl_file_cert = "cert.pem";
    } else if (name == "badhost") {
        params.hostname = "256.1.1.1";
    } else if (name == "reuse") {
        params.reuse_port = true;
        params.api_keys   = {"abc"};
    }
}

int main(int argc, char ** argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <config> <port-file>\n", argv[0]);
        return 2;
    }
    const std::string config = argv[1];
    common_log_set_verbosity_thold(LOG_LEVEL_DEBUG);

    common_params params;
    bool ready = true;
    configure(config, params, ready);

    server_http_context ctx_http;
    if (!ctx_http.init(params)) {
        LOG_INF("harness: init failed\n");
        common_log_flush(common_log_main());
        return 1;
    }

    ctx_http.get ("/health", ex_wrapper([](const server_http_req &) { return json_res({{"status", "ok"}}); }));
    ctx_http.get ("/echo",   ex_wrapper(echo("get")));
    ctx_http.post("/echo",   ex_wrapper(echo("post")));
    ctx_http.del ("/echo",   ex_wrapper(echo("delete")));
    ctx_http.post("/slots/:id_slot", ex_wrapper(echo("slots")));
    ctx_http.get ("/two/:a/x/:b",    ex_wrapper(echo("two")));
    ctx_http.get ("/empty", ex_wrapper([](const server_http_req &) {
        return std::make_unique<server_http_res>();
    }));
    ctx_http.get ("/status", ex_wrapper([](const server_http_req & req) {
        auto res = std::make_unique<server_http_res>();
        res->status = std::stoi(req.get_param("code", "200"));
        res->data = req.get_param("body", "");
        if (req.params.count("ct")) {
            res->content_type = req.get_param("ct");
        }
        for (const auto & [k, v] : req.params) {
            if (k.rfind("h_", 0) == 0) {
                res->headers[k.substr(2)] = v;
            }
        }
        return res;
    }));
    ctx_http.get ("/stream", ex_wrapper([](const server_http_req & req) {
        auto res = std::make_unique<server_http_res>();
        res->content_type = "text/event-stream";
        res->headers["Cache-Control"] = "no-cache";
        auto step = std::make_shared<int>(0);
        const auto * stop = &req.should_stop;
        res->next = [step, stop](std::string & out) -> bool {
            switch ((*step)++) {
                case 0: out = "data: a\n\n"; return true;
                case 1: return true;
                case 2: out = "data: bc\n\n"; return true;
                case 3: return true;
                default:
                    out = std::string("data: stop=") + ((*stop)() ? "1" : "0") + "\n\n";
                    return false;
            }
        };
        return res;
    }));
    ctx_http.get ("/stream-wait", ex_wrapper([](const server_http_req & req) -> server_http_res_ptr {
        auto res = std::make_unique<wait_res>();
        res->content_type = "text/plain";
        auto step = std::make_shared<int>(0);
        const auto * stop = &req.should_stop;
        res->next = [step, stop](std::string & out) -> bool {
            if ((*step)++ == 0) {
                out = "first\n";
                return true;
            }
            for (int i = 0; i < 300; i++) {
                if ((*stop)()) {
                    LOG_INF("harness: should_stop fired\n");
                    return false;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            LOG_INF("harness: should_stop timed out\n");
            out = "late\n";
            return false;
        };
        return res;
    }));
    ctx_http.get ("/wait-complete", ex_wrapper([](const server_http_req &) {
        bool complete = false;
        for (int i = 0; i < 300 && !complete; i++) {
            complete = g_wait_complete.exchange(false);
            if (!complete) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }
        return json_res({{"complete", complete}});
    }));
    ctx_http.post("/throw-invalid", ex_wrapper([](const server_http_req & req) -> server_http_res_ptr {
        throw std::invalid_argument("bad input: " + req.body);
    }));
    ctx_http.post("/throw-runtime", ex_wrapper([](const server_http_req &) -> server_http_res_ptr {
        throw std::runtime_error("boom");
    }));
    ctx_http.get ("/throw-unknown", ex_wrapper([](const server_http_req &) -> server_http_res_ptr {
        throw 42;
    }));
    // not wrapped: the exception reaches cpp-httplib's exception handler
    ctx_http.get ("/throw-raw", [](const server_http_req &) -> server_http_res_ptr {
        throw std::runtime_error("raw boom");
    });
    ctx_http.get ("/quit", ex_wrapper([](const server_http_req &) {
        g_quit = true;
        return json_res({{"bye", true}});
    }));

    ctx_http.register_gcp_compat();

    if (!ctx_http.start()) {
        LOG_INF("harness: start failed\n");
        common_log_flush(common_log_main());
        return 1;
    }
    if (ready) {
        ctx_http.is_ready.store(true);
    }
    LOG_INF("harness: listening on %s\n", ctx_http.listening_address.c_str());
    common_log_flush(common_log_main());

    FILE * f = fopen(argv[2], "w");
    if (f) {
        fprintf(f, "%d\n", ctx_http.port);
        fclose(f);
    }

    // GET /quit, or <port-file>.quit appearing: a server that is not ready
    // answers /quit with 503
    const std::string quit_file = std::string(argv[2]) + ".quit";
    while (!g_quit) {
        if (FILE * q = fopen(quit_file.c_str(), "r")) {
            fclose(q);
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    ctx_http.stop();
    if (ctx_http.thread.joinable()) {
        ctx_http.thread.join();
    }
    LOG_INF("harness: stopped\n");
    common_log_flush(common_log_main());
    return 0;
}
