load("@rules_python//python:defs.bzl", "py_binary")
load("@pip//:requirements.bzl", "requirement")

# bazel run //:catGPTserver
py_binary(
    name = "catGPTserver",
    srcs = ["server.py"],
    main = "server.py",
    deps = [
        requirement("fastapi"),
        requirement("uvicorn"),
        requirement("websockets"),
        requirement("python-dotenv"),
        requirement("pytz"),
        requirement("python-multipart"),
    ],
    data = [
        "system_prompt.txt",
    ],
)