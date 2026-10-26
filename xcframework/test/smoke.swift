// `test/driver.c`'s loop, line for line, in Swift through `import llama`.
//
// Proves the framework is usable from Swift -- module map, autolinked system
// frameworks -- and, diffed against the driver linked to the port's own
// libraries, that packaging changed no output. `make dropin-macos` also
// links it against upstream's framework and runs it on ours.
//
// Usage: smoke <model.gguf> <prompt> <n_predict> [cpu|gpu]

import Foundation
import llama

let args = CommandLine.arguments
guard args.count >= 4, let nPredict = Int(args[3]) else {
    fputs("usage: smoke <model.gguf> <prompt> <n_predict> [cpu|gpu]\n", stderr)
    exit(2)
}
let cpu = args.count > 4 && args[4] == "cpu"

llama_backend_init()
var mp = llama_model_default_params()
mp.n_gpu_layers = 99
// The loader copies the device list, so it only has to outlive the call.
var cpuOnly: [ggml_backend_dev_t?] = [nil, nil]
let loaded = cpuOnly.withUnsafeMutableBufferPointer { devices -> OpaquePointer? in
    if cpu {
        devices[0] = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU)
        mp.devices = devices.baseAddress
        mp.n_gpu_layers = 0
    }
    return llama_model_load_from_file(args[1], mp)
}
guard let model = loaded else { exit(1) }
let vocab = llama_model_get_vocab(model)

let prompt = args[2]
var toks = [llama_token](repeating: 0, count: 512)
let n = llama_tokenize(vocab, prompt, Int32(prompt.utf8.count), &toks, 512, true, true)
if n < 0 { exit(1) }

var cp = llama_context_default_params()
cp.n_ctx = 512
cp.n_batch = 512
if cpu {
    cp.offload_kqv = false
    cp.op_offload = false
}
guard let ctx = llama_init_from_model(model, cp) else { exit(1) }
let smpl = llama_sampler_chain_init(llama_sampler_chain_default_params())
llama_sampler_chain_add(smpl, llama_sampler_init_greedy())

var id: llama_token = 0
var piece = [CChar](repeating: 0, count: 256)
for i in 0..<nPredict {
    let rc: Int32 = i == 0
        ? toks.withUnsafeMutableBufferPointer { llama_decode(ctx, llama_batch_get_one($0.baseAddress, n)) }
        : withUnsafeMutablePointer(to: &id) { llama_decode(ctx, llama_batch_get_one($0, 1)) }
    if rc != 0 { exit(1) }
    id = llama_sampler_sample(smpl, ctx, -1)
    if llama_vocab_is_eog(vocab, id) { break }
    let k = llama_token_to_piece(vocab, id, &piece, 256, 0, true)
    if k > 0 { piece.withUnsafeBytes { _ = fwrite($0.baseAddress, 1, Int(k), stdout) } }
}
fputs("\n", stdout)

llama_sampler_free(smpl)
llama_free(ctx)
llama_model_free(model)
llama_backend_free()
