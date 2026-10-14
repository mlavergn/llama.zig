import llamazig

/// Returns llama.cpp's system-info line, through the binary target.
public func systemInfo() -> String {
    llama_backend_init()
    defer { llama_backend_free() }
    return String(cString: llama_print_system_info())
}
