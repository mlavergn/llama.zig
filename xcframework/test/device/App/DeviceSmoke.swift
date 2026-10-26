// `test/smoke.swift`'s loop as an iOS app, for `make smoke-device`: it runs
// every prompt on the CPU and on Metal, prints each output as one
// `SMOKE <mode> <n> <base64>` line -- the exact bytes, newlines included --
// and exits. `xcrun devicectl device process launch --console` returns the
// lines; `smoke.sh device-split` turns them back into files.
//
// The model is the bundle's `model.gguf`, which the Makefile links into this
// folder when it stages the project; it is not checked in.

import Foundation
import SwiftUI
import os
import llama

// parity-port's six prompts, as `smoke.sh` has them.
let prompts = [
    "The capital of France is",
    "Explain gravity in one sentence:",
    "def fibonacci(n):",
    "Write a haiku about winter:",
    "List three prime numbers:",
    "Translate to French: good morning",
]
let nPredict = 32

/// The model on one device. `smoke.swift` loads it once per prompt, one
/// process each; here it loads once per device and every prompt gets a fresh
/// context and sampler -- the same output, since nothing a decode reads from
/// the model changes, and five fewer 1.2 GB loads.
func load(_ path: String, cpu: Bool) -> OpaquePointer? {
    var mp = llama_model_default_params()
    mp.n_gpu_layers = 99
    var cpuOnly: [ggml_backend_dev_t?] = [nil, nil]
    return cpuOnly.withUnsafeMutableBufferPointer { devices -> OpaquePointer? in
        if cpu {
            devices[0] = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU)
            mp.devices = devices.baseAddress
            mp.n_gpu_layers = 0
        }
        return llama_model_load_from_file(path, mp)
    }
}

/// One greedy completion, `smoke.swift`'s loop over a fresh context and
/// sampler.
func generate(_ model: OpaquePointer, _ prompt: String, cpu: Bool) -> Data? {
    let vocab = llama_model_get_vocab(model)

    var toks = [llama_token](repeating: 0, count: 512)
    let n = llama_tokenize(vocab, prompt, Int32(prompt.utf8.count), &toks, 512, true, true)
    if n < 0 { return nil }

    var cp = llama_context_default_params()
    cp.n_ctx = 512
    cp.n_batch = 512
    if cpu {
        cp.offload_kqv = false
        cp.op_offload = false
    }
    guard let ctx = llama_init_from_model(model, cp) else { return nil }
    defer { llama_free(ctx) }
    let smpl = llama_sampler_chain_init(llama_sampler_chain_default_params())
    defer { llama_sampler_free(smpl) }
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy())

    var out = Data()
    var id: llama_token = 0
    var piece = [CChar](repeating: 0, count: 256)
    for i in 0..<nPredict {
        let rc: Int32 = i == 0
            ? toks.withUnsafeMutableBufferPointer { llama_decode(ctx, llama_batch_get_one($0.baseAddress, n)) }
            : withUnsafeMutablePointer(to: &id) { llama_decode(ctx, llama_batch_get_one($0, 1)) }
        if rc != 0 { return nil }
        id = llama_sampler_sample(smpl, ctx, -1)
        if llama_vocab_is_eog(vocab, id) { break }
        let k = llama_token_to_piece(vocab, id, &piece, 256, 0, true)
        if k > 0 { piece.withUnsafeBytes { out.append(contentsOf: $0.prefix(Int(k))) } }
    }
    out.append(0x0A)
    return out
}

/// The app's physical footprint and what iOS will still give it, in MiB --
/// the two numbers jetsam decides by. Printed after each prompt as a
/// `SMOKE-MEM` line, which the comparison ignores.
func memory() -> String {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
    let kr = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) }
    }
    let footprint = kr == KERN_SUCCESS ? info.phys_footprint >> 20 : 0
    return "footprint \(footprint) MiB, available \(os_proc_available_memory() >> 20) MiB"
}

/// Every prompt on both devices, printed as `SMOKE` lines; then the app
/// exits.
func run() {
    setvbuf(stdout, nil, _IOLBF, 0)
    guard let path = Bundle.main.path(forResource: "model", ofType: "gguf") else {
        print("SMOKE-FAIL no model.gguf in the bundle")
        exit(1)
    }
    var info = utsname()
    uname(&info)
    let machine = withUnsafeBytes(of: &info.machine) { String(decoding: $0.prefix { $0 != 0 }, as: UTF8.self) }
    print("SMOKE-DEVICE \(machine)")
    print("SMOKE-MEM start \(memory())")
    llama_backend_init()
    for mode in ["cpu", "gpu"] {
        // A background thread has no run loop to drain autoreleased Metal
        // objects, so each device gets a pool of its own.
        autoreleasepool {
            guard let model = load(path, cpu: mode == "cpu") else {
                print("SMOKE-FAIL \(mode) load")
                exit(1)
            }
            for (i, p) in prompts.enumerated() {
                guard let text = generate(model, p, cpu: mode == "cpu") else {
                    print("SMOKE-FAIL \(mode) \(i + 1)")
                    exit(1)
                }
                print("SMOKE \(mode) \(i + 1) \(text.base64EncodedString())")
                print("SMOKE-MEM \(mode) \(i + 1) \(memory())")
            }
            llama_model_free(model)
        }
        print("SMOKE-MEM \(mode) freed \(memory())")
    }
    llama_backend_free()
    print("SMOKE-DONE")
    exit(0)
}

/// The work runs on a thread of its own once the app has launched: iOS
/// kills an app that has not finished launching within 20 s of wall-clock
/// time (`0x8BADF00D`, "process-launch watchdog transgression"), which twelve
/// completions of a 2B model take far longer than.
@main
struct DeviceSmoke: App {
    init() {
        let worker = Thread { run() }
        worker.stackSize = 8 << 20
        worker.start()
    }

    var body: some Scene {
        WindowGroup { Text("llama.xcframework device smoke") }
    }
}
