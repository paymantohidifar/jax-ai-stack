# **JAX-in-the-Box: Unboxing Stateful Training with Flax NNX and Orbax**

## **1. Background: The Evolution of JAX**

Born from years of scaling TensorFlow and engineered to meet the demands of modern, highly flexible AI research, JAX has become Google’s flagship platform for high-performance numerical computing. Today, JAX powers the vast majority of Google’s breakthrough research and generative AI initiatives, including AlphaFold, Gemini, Gemma, *etc*.

Architecturally, JAX achieves unrivaled hardware portability compared to other legacy deep learning frameworks [\[1\]](https://arxiv.org/pdf/2309.07181). By leveraging the XLA (Accelerated Linear Algebra) compiler as its unified backend, JAX can compile identical Python code into optimized machine instructions.

For years, however, choosing between PyTorch and JAX required accepting a stark philosophical tradeoff: PyTorch offered developers intuitive, object-oriented state management (via `torch.nn.Module`), while JAX strictly demanded functional purity. To manage neural network weights in traditional JAX frameworks, developers had to manually extract, pass, and return dictionary states (`pytrees`) through every single computational function.

That paradigm has officially shifted. With the stabilization of **Flax NNX**, JAX now features a native stateful, object-oriented API. It delivers the modular feel of PyTorch while preserving the raw, composable functional power that makes JAX unique.

In this post, we will walk through an end-to-end training and checkpointing pipeline for a simple Multi-Layer Perceptron (MLP) using Flax NNX and Orbax. We will also explore how this modern stack achieves seamless hardware portability across CPUs, GPUs, and TPUs without changing a single line of code—a capability you can easily verify yourself using the free accelerator tiers in Google Colab. You can experiment this using the [notebook]() version of this blog on Colab 

---

## **2. End-to-End Implementation: Building an MLP Pipeline**

In the following section, we will build a self-contained AI stack step-by-step: (i) initialize an MLP with two layers, (ii) train it on a synthetic batch using an encapsulated compile loop, (iii) track loss convergence, and (iv) export the optimized state safely using Orbax.

### **Step 2.1: Environment Setup and Core Imports**

Before writing model logic, we pull in the foundational libraries of our ecosystem: JAX for array operations, Optax for optimization algorithms, Flax NNX for stateful neural network blocks, and Orbax for high-performance checkpointing.

```python
import shutil
from pathlib import Path
import jax
import jax.numpy as jnp
from flax import nnx
import optax
import orbax.checkpoint as ocp
import matplotlib.pyplot as plt

```

### **Step 2.2: Defining the Stateful Model Architecture**

In classic Flax (Linen), variables were kept completely separate from the architecture definition. In `Flax.nnx`, parameters live directly inside our custom objects (e.g., `self.dense_layer_1`). NNX uses dynamic proxy objects to automatically discover child modules and their weights under the hood.

```python
# Create a stateful MLP by inheriting from nnx.Module
class MySimpleMlp(nnx.Module):
    def __init__(self, din: int, dout: int, rngs: nnx.Rngs):
        # Parameters (weights/biases) live inside these submodules automatically
        self.dense_layer_1 = nnx.Linear(din, 5, rngs=rngs)
        self.dense_layer_2 = nnx.Linear(5, dout, rngs=rngs)

    def __call__(self, x) -> jnp.ndarray:
        # Define the forward pass using standard imperative execution
        x = self.dense_layer_1(x)
        x = nnx.relu(x)
        x = self.dense_layer_2(x)
        return x

```

### **Step 2.3: Purely Composed Functional Train Step and In-Place JIT Mutation**

JAX demands functional purity—meaning side-effects like in-place variable mutation are strictly forbidden inside a `@jax.jit` compiled function.

NNX solves this beautifully with `nnx.jit`. When we pass an `nnx.Module` or `nnx.Optimizer` into a compiled function, NNX automatically executes an `nnx.split()` right before entering the JIT boundary. This breaks our object into a static structure graph and a dynamic array pytree. Once the function finishes running, it automatically merges them back together, allowing us to use intuitive patterns like `optimizer_arg.update(grads)` smoothly.

```python
# The nnx.jit decorator safely handles stateful objects crossing the JIT boundary
@nnx.jit
def train_step(model_arg: nnx.Module, optimizer_arg: nnx.Optimizer,
               x_batch: jnp.ndarray, y_batch: jnp.ndarray) -> jnp.ndarray:
    
    # Internal pure function mapping input parameters to scalar loss
    def loss_fn_for_grad(model_in_grad_fn: nnx.Module) -> float:
        y_pred = model_in_grad_fn(x_batch)
        return jnp.mean((y_batch - y_pred) ** 2)
    
    # nnx.value_and_grad cleanly extracts the dynamic pytree values and their gradients
    loss_value, grads = nnx.value_and_grad(loss_fn_for_grad)(model_arg)
    
    # Mutates optimizer state in-place; NNX abstracts this side effect out safely
    optimizer_arg.update(model_arg, grads)  
    return loss_value

```

### **Step 2.4: Deterministic Random Number Generation and State Initialization**

JAX handles randomness via explicit PRNG state keys rather than global hidden seeds. In this step, we safely split our source entropy into dedicated keys for the network weights, the input features, and the target labels, followed by instantiating our `nnx.Optimizer` tracker.

```python
# Initialize base random key state
main_key = jax.random.PRNGKey(42)
test_key, main_key = jax.random.split(main_key, 2)
test_key_model, test_key_x, test_key_y = jax.random.split(test_key, 3)

# Establish dimensions for our synthetic data batches
batch_size, din, dout = 8, 4, 2
x_batch = jax.random.normal(test_key_x, shape=(batch_size, din))
y_batch = jax.random.normal(test_key_y, shape=(batch_size, dout))

# Pass explicit tracking RNG containers into the NNX model
model_rng = nnx.Rngs(params=test_key_model)
my_model = MySimpleMlp(din, dout, rngs=model_rng)

# Bind model parameters directly to an Optax-backed stateful optimizer
tx = optax.adam(learning_rate=0.1)
my_optimizer = nnx.Optimizer(my_model, tx, wrt=nnx.Param) # wrt=With Respect To

```

### **Step 2.5: Executing the Optimization Loop**

With our compiled training step locked and loaded, we run a training loop across 40 epochs. Because `train_step` is optimized by XLA, subsequent iterations bypass Python overhead entirely, executing at raw hardware speed.

```python
num_epochs = 40
loss_all = []
for epoch in range(num_epochs):
    # This call triggers the hardware-optimized fused execution kernel
    loss = train_step(my_model, my_optimizer, x_batch, y_batch)
    loss_all.append(float(loss))

```

Now, let's visualize loss evolution over epochs:

```python
fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(range(1, num_epochs+1), loss_all, 'o-')
ax.set_title("Training loss over epochs")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
plt.show()
```

<div align='left'>
    <img src='assets/simple-ai-stack-jax-loss.png' width=400px>
</div>


### **Step 2.6: Initializing Production-Grade Checkpointing via Orbax**

When saving model weights using `orbax.checkpoint`, you might encounter serialization failures if you pass a relative path (like `../model-checkpoints`). Orbax relies heavily on Google's **TensorStore** engine to coordinate asynchronous, multi-threaded array caching. TensorStore requires strict **absolute paths** (`Path.resolve()`) to ensure no file shards are misplaced during intensive write routines.

```python
# Convert your storage folder into a strict absolute path to satisfy TensorStore
checkpoint_dir = Path.cwd().parent.resolve().joinpath('model-checkpoints')
if checkpoint_dir.exists():
    shutil.rmtree(checkpoint_dir) # Clean out any existing checkpoints from prior runs

# Initialize manager configurations
current_step = 40
mngr_options = ocp.CheckpointManagerOptions(save_interval_steps=1, max_to_keep=1)
mngr = ocp.CheckpointManager(checkpoint_dir, options=mngr_options)

```

### **Step 2.7: Deconstructing Objects into Pure State Bundles**

Before saving, we extract the underlying raw dictionary data weights out of our stateful classes using `nnx.split()`. This creates a serialization-ready bundle containing pure arrays that TensorStore can effortlessly stream onto the disk.

```python
# Strip away object metadata, leaving behind pure JAX dictionary states (pytrees)
_, model_state_to_save = nnx.split(my_model)
_, optimizer_state_to_save = nnx.split(my_optimizer)

# Construct a dictionary containing the pure states
save_bundle = {
    'model': model_state_to_save,
    'optimizer': optimizer_state_to_save,
    'current_step': current_step
}

# Stream out your saved objects asynchronously and block until completion
mngr.save(current_step, args=ocp.args.StandardSave(save_bundle))
mngr.wait_until_finished()
print(f"Model successfully checkpointed to absolute path: {checkpoint_dir}")

```

---

## **3. Hardware Portability: Zero Code Adjustments from CPU to TPU**

One of the most remarkable strengths of this specific JAX/Flax AI stack is its absolute architecture portability.

In frameworks like PyTorch, transitioning a model from a local testing laptop (CPU) to a cloud cluster (GPU or TPU) requires manual device context handling, such as scattering calls like `.to('cuda')` or `.to(device)` throughout your model definition and data loading routines.

```text
PyTorch Approach:   [CPU Data] -> .to('cuda') -> [GPU Processing]
JAX/NNX Approach:   [Unified Arrays] -> Automatic JIT Virtual Device Allocation

```

JAX handles this completely differently through its XLA compiler backend:

### **Transparent Array Allocation**

JAX arrays (`jnp.ndarray`) are fundamentally device-agnostic abstractions. When you run JAX code, the arrays are automatically placed on your system's primary default accelerator. If you have a discrete GPU available, JAX initializes your parameters directly inside GPU VRAM; if you are running on a Google Cloud TPU node, they are routed to TPU HBM memory.

### **The Compiling Layer (`@nnx.jit`)**

The `@nnx.jit` decorator acts as a high-performance wrapper around `jax.jit`. Rather than interpreting Python line-by-line, XLA acts as an optimizing compiler. It fuses consecutive mathematical steps (like our `Linear` layers and `relu` activation) into unified, hardware-optimized kernel operations tailored specifically for the target silicon architecture.

Because XLA abstracts the underlying instruction set, the exact same machine learning pipeline scales seamlessly from single-core CPU testing up to massive TPU-pod configurations with zero architectural changes required.