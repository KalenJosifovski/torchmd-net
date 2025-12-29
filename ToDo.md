⸻

Plan for a coding agent

Goal: (1) Fix the CPU wheel so get_neighbor_pairs works on CPU, and (2) ship a macOS arm64 wheel with an actual MPS (Metal) kernel (not just “fallback-to-CPU”).

I’ll structure this like an engineering task list with clear acceptance criteria.

⸻

0) Repro + environment capture (fast, deterministic)

Tasks
	•	Create a clean venv/conda env on macOS arm64.
	•	Install:
	•	torch (the exact version you’re using)
	•	torchmd-net-cpu (the wheel you installed)
	•	Run a tiny reproducer that calls:
	•	torch.ops.torchmdnet_extensions.get_neighbor_pairs(...) directly with CPU tensors.
	•	Record:
	•	torch.__version__, platform.platform(), python --version
	•	torch.backends.mps.is_available()
	•	torchmdnet.__version__

Acceptance
	•	Reproducer fails with the same NotImplementedError for CPU backend.

⸻

1) Identify what kernels are actually registered in your build

Tasks
	•	In Python, probe dispatcher registrations:
	•	Use torch._C._dispatch_has_kernel_for_dispatch_key("torchmdnet_extensions::get_neighbor_pairs", "CPU") and similarly for "MPS", "AutogradCPU", etc.
	•	Dump operator schema to ensure you’re querying the right overload.
	•	Confirm whether the wheel contains the compiled extension binary:
	•	Locate site-packages/torchmdnet_extensions*.so (or .dylib).

Acceptance
	•	You can prove (with dispatch probes) that CPU is missing, or that the kernel is registered under an unexpected key (your trace hints at odd keys appearing).

⸻

2) Fix (1): CPU wheel — compile + register a real CPU kernel

TorchMD-Net’s neighborlist op is implemented as an autograd-capable extension and is expected to be usable broadly. The TorchMD-Net 2.0 paper describes a reference CPU implementation and CUDA registration for forward, with autograd support overall.  ￼

2.1 Locate the extension code

Tasks
	•	In the TorchMD-Net repo, find the sources for get_neighbor_pairs:
	•	likely under something like torchmdnet/extensions/neighbors/ (e.g., neighbors_cpu.cpp, neighbors_cuda.cu, etc.)
	•	Find where the operator is defined and registered:
	•	TORCH_LIBRARY(torchmdnet_extensions, m) { ... }
	•	TORCH_LIBRARY_IMPL(torchmdnet_extensions, CPU, m) { ... }

Acceptance
	•	You can point to the exact file(s) that should register the CPU kernel.

2.2 Fix registration mistakes

Tasks
	•	If the CPU implementation is currently registered under AutogradCPU only (or some wrong key), correct it:
	•	Register the forward kernel under CPU.
	•	If they use custom autograd, ensure autograd is wired correctly (either via Autograd key registration, or via torch::autograd::Function wrapper, depending on current design).
	•	Ensure the extension build always includes the CPU source file(s) for macOS arm64 builds.

Acceptance
	•	torch._C._dispatch_has_kernel_for_dispatch_key(..., "CPU") == True.
	•	The minimal CPU reproducer runs and returns outputs of correct shape/dtype.

2.3 Add tests (so it never regresses)

Tasks
	•	Add unit tests that run on:
	•	CPU tensors
	•	(later) MPS tensors
	•	Validate correctness vs a pure-PyTorch reference implementation for small sizes:
	•	neighbor pairs set equality (order-insensitive) and distances within tolerance.

Acceptance
	•	CI passes and the test fails on the current broken wheel but passes after the fix.

2.4 Build and publish a repaired CPU wheel

Tasks
	•	Update CI packaging (cibuildwheel) to build macOS arm64 CPU wheels reliably.
	•	Verify wheel locally:
	•	pip install dist/*.whl
	•	run reproducer + unit tests

Acceptance
	•	Published torchmd-net-cpu (or equivalent) wheel works on macOS arm64 CPU.

⸻

3) Fix (2): Implement a true MPS (Metal) kernel + macOS arm64 “MPS” wheel

Important nuance: PyTorch supports the MPS device for many ops, but custom operators do not become MPS-capable automatically—you must implement/register an MPS kernel for that op (or accept CPU fallback). PyTorch’s dispatcher model makes this explicit.  ￼

3.1 Define the target behavior

Tasks
	•	Decide whether the MPS kernel will support:
	•	Forward only, or forward + backward.
	•	Decide whether to support both neighborlist strategies (cell list + brute force) or just one initially.

Recommendation
	•	Start with forward-only MPS brute force for correctness and simplicity, then optimize (cell lists / tiling) later.

Acceptance
	•	A single call on device="mps" works end-to-end without falling back.

3.2 Implementation approach

You have two realistic routes:

Route A (fastest to “working”): MPS tensors → CPU kernel fallback (temporary)
	•	Once CPU is fixed, users can enable MPS fallback via PYTORCH_ENABLE_MPS_FALLBACK=1 (documented by PyTorch).  ￼
	•	This is not “real Metal acceleration”, but it unblocks training/inference quickly.

Acceptance
	•	On MPS runs, the op executes (with warning/perf penalty), no crash.

Route B (what you asked for): Real Metal compute kernel

Tasks
	•	Add an Objective-C++ / Metal implementation for neighbor pair search:
	•	New files like neighbors_mps.mm + neighbors_mps.metal
	•	Use the official pattern for custom PyTorch operators (C++ extension) and dispatch registration:
	•	TORCH_LIBRARY_IMPL(torchmdnet_extensions, MPS, m) { ... }  ￼
	•	Use Metal compute to:
	•	compute neighbor pairs within cutoff
	•	write out fixed-size padded outputs (TorchMD-Net already uses padding strategies for CUDA-graph compatibility per their design discussions).  ￼
	•	Ensure memory layout assumptions:
	•	pos contiguous float32
	•	batch indices int64 or int32 (choose one and enforce)
	•	Gradients:
	•	If backward is implemented in PyTorch (common pattern), keep it that way.
	•	If backward needs kernel support, implement a second Metal kernel.

References for implementing Metal custom ops
	•	Follow PyTorch’s C++ custom operator build/packaging approach.  ￼
	•	Use a known working template for a PyTorch + Metal shader extension as a starting scaffold.  ￼

Acceptance
	•	torch._C._dispatch_has_kernel_for_dispatch_key(..., "MPS") == True
	•	Running the reproducer with MPS tensors:
	•	produces identical results (within tolerance) to CPU reference
	•	shows no “fall back to run on the CPU” warning for this operator

3.3 Wheel strategy for macOS arm64

Tasks
	•	Decide naming:
	•	Option 1: publish torchmd-net-mps as a new wheel
	•	Option 2: publish a unified macOS wheel that supports both CPU and MPS
	•	Add CI build job on macOS arm64 that:
	•	compiles .mm / .metal
	•	links required frameworks (Metal, Foundation)
	•	runs CPU + MPS tests

Acceptance
	•	pip install ... on macOS arm64 gives a wheel where MPS path is available and tested.

⸻

4) Extra: Document clearly what’s supported on macOS

Tasks
	•	Update TorchMD-Net docs/README:
	•	Current install instructions list CPU/CUDA wheels only.  ￼
	•	Add a macOS section:
	•	“MPS wheel available as …”
	•	“Fallback option via PYTORCH_ENABLE_MPS_FALLBACK=1” (with warning about speed).  ￼
	•	Add a troubleshooting entry for the exact error you hit.

Acceptance
	•	Users don’t waste time installing the wrong wheel and can self-diagnose.

⸻