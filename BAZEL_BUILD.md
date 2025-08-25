# Bazel Build Instructions

## Prerequisites

1. Install Bazel or Bazelisk:
   ```bash
   wget -O bazelisk https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-arm64
   chmod +x bazelisk
   sudo mv bazelisk /usr/local/bin/bazel
   ```

## Building and Running

### Build the server
```bash
bazel build //:catGPTserver
```

### Run the server
```bash
bazel run //:catGPTserver
```

### Update dependencies
If you modify `requirements.in`, regenerate the lock file:
```bash
pip-compile requirements.in --output-file requirements_lock.txt
```

## File Structure

- `MODULE.bazel` - Bazel module configuration
- `BUILD` - Build targets definition  
- `requirements.in` - Python dependencies
- `requirements_lock.txt` - Locked dependency versions
- `.bazelversion` - Pinned Bazel version

