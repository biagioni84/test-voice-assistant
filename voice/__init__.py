# llama_cpp DEBE importarse antes que faster_whisper/onnxruntime/piper/etc.: otro paquete
# deja una libggml incompatible en site-packages/lib64 y, si carga primero, libllama.so
# falla con "undefined symbol: gguf_init_from_file_ptr".
import llama_cpp  # noqa: F401
