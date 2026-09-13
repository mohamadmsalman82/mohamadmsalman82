<div align="center">
  <img src="assets/banner.svg" alt="Mohamad Salman. ML systems, GPU infrastructure, and software that ships." width="100%">
</div>

<p align="center">
  <a href="https://www.linkedin.com/in/msalman06"><img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn"></a>
  <a href="mailto:mohamad.salman@mail.utoronto.ca"><img src="https://img.shields.io/badge/Email-C71610?style=for-the-badge&logo=maildotru&logoColor=white" alt="Email"></a>
  <a href="https://physsplat.vercel.app"><img src="https://img.shields.io/badge/Live_Demo-2EA44F?style=for-the-badge&logo=vercel&logoColor=white" alt="Live demo"></a>
  <img src="https://img.shields.io/badge/Toronto,_ON-1F2937?style=for-the-badge&logo=googlemaps&logoColor=white" alt="Toronto">
</p>

---

## About me in a nutshell

- **Machine learning researcher at the University of Toronto,** co-author on a paper submitting to EuCAP 2027.
- **Built an iOS app alone** that hit 10,000 users and $4,000 MRR, then demoed it live on CP24.
- **Wrote a Llama inference engine in raw CUDA** that saturates 79% of an RTX 4090's memory bus.
- **Wrote a distributed training framework** that beats NVIDIA's Megatron-LM by 37%.
- **Found a CVSS 9.1 flaw** in a 30k-star AI serving project and took it through to a CVE.

**I am going all in on machine learning.** The work I want is the layer underneath the model:
training systems, inference runtimes, and the GPU engineering that makes both fast.

<div align="center">
  <br>
  <img src="assets/cp24.jpg" alt="Mohamad Salman on CP24, introduced on screen as CrowdKick Founder" width="760">
  <br>
  <sub>Live on CP24 demoing CrowdKick, one day before the World Cup.</sub>
</div>

<br>

# &nbsp;🔬&nbsp; Research

### Machine Learning Researcher, University of Toronto &nbsp;<sub>with Prof. Parinaz Naseri</sub>

<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white"> <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white"> <img src="https://img.shields.io/badge/EuCAP_2027-00629B?style=flat-square&logo=ieee&logoColor=white">

- **What I work on.** Surrogate models that stand in for electromagnetic solvers on antenna metasurfaces. A structure that costs hours of solver time becomes a forward pass in milliseconds, which is what makes searching the design space feasible at all.
- **What I built.** I benchmarked Gaussian processes, random forests, and gradient boosting in scikit-learn, then built the PyTorch neural network that beat all of them, with custom early stopping and a physics-informed loss. I own the data cleaning, feature encoding, cross-validation, and reproducible training code end to end.
- **What it changed.** My study decided the team's input representation by showing one shape family needs **6 to 8 times less** simulation data for the same accuracy. I also showed that an energy-conservation shortcut the team planned to rely on would introduce up to **183 times** the label error, which changed the method before it reached the paper.
- **Where it goes.** Co-author on a paper submitting to **EuCAP 2027**, an IEEE international conference.

<br>

# &nbsp;🚀&nbsp; CrowdKick

### [CrowdKick](https://apps.apple.com/ca/app/crowdkick-sports-venue-finder/id6763609021) &nbsp;<sub>founder and CTO, built and shipped alone</sub>

<img src="https://img.shields.io/badge/React_Native-61DAFB?style=flat-square&logo=react&logoColor=black"> <img src="https://img.shields.io/badge/Supabase-3FCF8E?style=flat-square&logo=supabase&logoColor=white"> <img src="https://img.shields.io/badge/PostGIS-336791?style=flat-square&logo=postgresql&logoColor=white"> <img src="https://img.shields.io/badge/LLM_Pipeline-8A2BE2?style=flat-square"> <img src="https://img.shields.io/badge/App_Store-0D96F6?style=flat-square&logo=appstore&logoColor=white">

- **What it is.** A real-time iOS app that tells you which bars near you are showing the game you want to watch. I designed it, built it, launched it, and still run it, by myself.
- **Why it stands out.** Ten thousand users, $4,000 in monthly recurring revenue, and 150 partner venues inside six weeks. It peaked in the **top 15 of the App Store Sports category** on organic growth alone, and CP24 put me on air to demo it to roughly 70,000 viewers.
- **The machine learning inside it.** Venue coverage comes from an NLP pipeline that uses LLMs to read Reddit posts and pull out which bar is showing which match. Every extraction is validated with Pydantic, fuzzy-matched to a real venue, and scored for confidence before it reaches a user.
- **Skills shown.** Taking a production mobile app from idea to paying users, React Native, PostgreSQL and PostGIS for geospatial queries, LLM extraction with structured validation, and `pytest` suites over synthetic data for every pipeline stage.

<br>

# &nbsp;⚙️&nbsp; ML Systems &amp; Infrastructure

### [llama-cuda-runtime](https://github.com/mohamadmsalman82/llama-cuda-runtime) &nbsp;<sub>a Llama-3.2-1B inference engine written from scratch</sub>

<img src="https://img.shields.io/badge/C%2B%2B17-00599C?style=flat-square&logo=cplusplus&logoColor=white"> <img src="https://img.shields.io/badge/CUDA-76B900?style=flat-square&logo=nvidia&logoColor=white"> <img src="https://img.shields.io/badge/cuBLAS-76B900?style=flat-square"> <img src="https://img.shields.io/badge/Nsight-1A1A1A?style=flat-square&logo=nvidia&logoColor=76B900">

- **What it is.** A complete LLM inference engine in C++17 and CUDA with no PyTorch, no libtorch, and no inference library of any kind. I wrote the safetensors loader, the BPE tokenizer, the paged KV cache, and a custom kernel for every operation in the model, including grouped-query attention and RoPE. cuBLAS does the matmuls and nothing else.
- **Why it stands out.** Decoding a token means reading all 2.47 GB of weights out of HBM, so the only honest score is what fraction of the memory bus you saturate. This runtime hits 798 GB/s on an RTX 4090, which is **79.1% of the card's theoretical peak**, at 322 tokens/s. HuggingFace Transformers manages 17.7% on the same card. Every layer was validated against a PyTorch reference before any of it was timed.
- **Skills shown.** CUDA kernel engineering, GPU memory hierarchy and occupancy, roofline analysis, Nsight profiling, modern C++, and the discipline to prove numerical correctness before claiming a speedup.

<br>

### [dstraining](https://github.com/mohamadmsalman82/dstraining) &nbsp;<sub>every form of LLM parallelism, written on raw collectives</sub>

<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white"> <img src="https://img.shields.io/badge/NCCL-76B900?style=flat-square&logo=nvidia&logoColor=white"> <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white"> <img src="https://img.shields.io/badge/Distributed-8A2BE2?style=flat-square">

- **What it is.** Tensor, sequence, pipeline, and data parallelism for GPT-2, all implemented from first principles in about 3,500 lines. No DDP, no FSDP, no DeepSpeed, no Megatron code. Pipeline parallelism includes both 1F1B and interleaved schedules, and the trainer composes all four axes at once with selective recompute and bf16 weights on fp32 masters.
- **Why it stands out.** It races NVIDIA's own Megatron-LM on identical hardware and identical kernels, and wins by **37% on four-way pipeline parallel** and 16% on two-way data parallel. Where it loses, at two-way tensor parallel, the README publishes the number and the reason instead of hiding it. Five correctness suites prove each parallel model matches a single-process reference numerically.
- **Skills shown.** `torch.distributed` and NCCL at the collective level, 1F1B pipeline scheduling, gradient bucketing overlapped with backward, vocab-parallel cross-entropy, mixed-precision training, and debugging deadlocks across multiple communicators.

<br>

### [mlc-compiler](https://github.com/mohamadmsalman82/mlc-compiler) &nbsp;<sub>an ahead-of-time compiler for static-shape PyTorch inference</sub>

<img src="https://img.shields.io/badge/Triton-8A2BE2?style=flat-square"> <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white"> <img src="https://img.shields.io/badge/CUDA_Graphs-76B900?style=flat-square&logo=nvidia&logoColor=white"> <img src="https://img.shields.io/badge/Compilers-0F172A?style=flat-square">

- **What it is.** It captures a model with `torch.export`, lowers it into its own graph IR, fuses elementwise and reduction chains into single Triton kernels, packs every intermediate tensor into one planned memory arena, and replays the entire schedule as a CUDA graph.
- **Why it stands out.** At batch size 1 on BERT-base it runs 5.5 times faster than eager and **3.0 times faster than `torch.compile`**. The more interesting result is that the advantage vanishes by batch 8, and the project is built around answering exactly that question: how much is specializing all the way to static-shape inference actually worth, and where does it stop paying. Every variant is checked against eager before it is timed.
- **Skills shown.** Compiler and IR design, operator fusion, Triton kernel authoring, whole-graph memory planning, CUDA graph capture, and benchmark methodology that reports the negative result.

<br>

# &nbsp;🧠&nbsp; Applied Machine Learning

### [physsplat](https://github.com/mohamadmsalman82/physsplat) &nbsp;<sub>a photo goes in, interactive learned 3D physics comes out</sub> &nbsp;[![demo](https://img.shields.io/badge/try_it_live-2EA44F?style=flat-square&logo=vercel&logoColor=white)](https://physsplat.vercel.app)

<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white"> <img src="https://img.shields.io/badge/ONNX-005CED?style=flat-square&logo=onnx&logoColor=white"> <img src="https://img.shields.io/badge/WebGPU-005A9C?style=flat-square&logo=webgpu&logoColor=white"> <img src="https://img.shields.io/badge/three.js-000000?style=flat-square&logo=threedotjs&logoColor=white">

- **What it is.** One photograph of objects on a table becomes a scene you can orbit, grab, and knock over in the browser. The pipeline reconstructs the photo to 3D with TripoSR, decomposes it into rigid bodies with no supervision, and hands the result to a physics engine. One of the two engines is a graph neural network trained entirely on synthetic data with no hand-coded collision solver.
- **Why it stands out.** Every physics step runs client-side with no server in the loop, which meant exporting the GNN to ONNX and engineering around a race in ONNX Runtime's scatter operation. Eight blind review rounds went into measuring and fencing the learned model's failure modes, and the write-up is honest that the analytic solver still ships as the default.
- **Skills shown.** Graph neural networks, sim-to-real transfer, synthetic data generation, ONNX export and its sharp edges, WebGPU and three.js, and designing an evaluation loop that drives model iteration.

<br>

### [multi-agent-debate](https://github.com/mohamadmsalman82/multi-agent-debate) &nbsp;<sub>a fine-tuned LLaMA that replaces a GPT-4 judge</sub>

<img src="https://img.shields.io/badge/LLaMA_3.1_8B-0668E1?style=flat-square&logo=meta&logoColor=white"> <img src="https://img.shields.io/badge/QLoRA-FFD21E?style=flat-square&logo=huggingface&logoColor=black"> <img src="https://img.shields.io/badge/DeepSpeed-0F6CBD?style=flat-square"> <img src="https://img.shields.io/badge/pytest-0A9EDC?style=flat-square&logo=pytest&logoColor=white">

- **What it is.** A framework where LLM agents run structured debates, with five role-specific agents and three turn-taking protocols. Any agent can be backed by a different provider, so GPT-4 can argue against Claude while Cohere fact-checks, all behind one async interface with retries and rate-limit handling.
- **Why it stands out.** The judge is the expensive part, so I replaced it. A LoRA adapter of 4M trainable parameters on LLaMA 3.1 8B, trained on 5,000 debate transcripts at 4-bit precision with DeepSpeed ZeRO, reaches **87% agreement with GPT-4 judgements** on a held-out set and cuts serving latency by 70%. The repo carries 89 tests at 93% coverage, run against a mock provider so the suite costs nothing.
- **Skills shown.** QLoRA fine-tuning, distributed training with DeepSpeed ZeRO, 4-bit quantization, self-hosted inference deployment, async multi-provider orchestration, evaluation design, and testing an LLM system without paying per run.

<br>

# &nbsp;🛡️&nbsp; Also Worth a Look

| | |
|---|---|
| **AI infrastructure security** | Found and coordinated disclosure of a **CVSS 9.1** unauthenticated path traversal in [LocalAI](https://github.com/mudler/LocalAI), a 30k-star model serving project, through to a CVE. The method generalizes: track a project's security commits, then audit every sibling code path the fix did not touch. |
| **[codelangid](https://github.com/mohamadmsalman82/codelangid)** | A character-level CNN that identifies which of 10 programming languages a snippet is written in, from the characters alone. **90.6%** on a final test set of fresh repositories that was collected after every hyperparameter was frozen and evaluated exactly once, with leakage checked by exact hash and a MinHash near-duplicate sweep. |

<br>

# &nbsp;🧰&nbsp; Toolkit

<table width="100%">
<tr>
<td valign="top"><b>Languages</b></td>
<td>
<img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white">
<img src="https://img.shields.io/badge/C%2B%2B-00599C?style=flat-square&logo=cplusplus&logoColor=white">
<img src="https://img.shields.io/badge/CUDA-76B900?style=flat-square&logo=nvidia&logoColor=white">
<img src="https://img.shields.io/badge/C-A8B9CC?style=flat-square&logo=c&logoColor=black">
<img src="https://img.shields.io/badge/TypeScript-3178C6?style=flat-square&logo=typescript&logoColor=white">
<img src="https://img.shields.io/badge/SQL-4479A1?style=flat-square&logo=postgresql&logoColor=white">
<img src="https://img.shields.io/badge/Bash-4EAA25?style=flat-square&logo=gnubash&logoColor=white">
</td>
</tr>
<tr>
<td valign="top"><b>ML &amp; GPU</b></td>
<td>
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white">
<img src="https://img.shields.io/badge/Triton-8A2BE2?style=flat-square">
<img src="https://img.shields.io/badge/torch.distributed-EE4C2C?style=flat-square&logo=pytorch&logoColor=white">
<img src="https://img.shields.io/badge/NCCL-76B900?style=flat-square&logo=nvidia&logoColor=white">
<img src="https://img.shields.io/badge/cuBLAS-76B900?style=flat-square&logo=nvidia&logoColor=white">
<img src="https://img.shields.io/badge/Transformers-FFD21E?style=flat-square&logo=huggingface&logoColor=black">
<img src="https://img.shields.io/badge/QLoRA-FFD21E?style=flat-square&logo=huggingface&logoColor=black">
<img src="https://img.shields.io/badge/DeepSpeed-0F6CBD?style=flat-square">
<img src="https://img.shields.io/badge/ONNX-005CED?style=flat-square&logo=onnx&logoColor=white">
<img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white">
<img src="https://img.shields.io/badge/Nsight-1A1A1A?style=flat-square&logo=nvidia&logoColor=76B900">
</td>
</tr>
<tr>
<td valign="top"><b>Backend &amp; Data</b></td>
<td>
<img src="https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white">
<img src="https://img.shields.io/badge/PostgreSQL-336791?style=flat-square&logo=postgresql&logoColor=white">
<img src="https://img.shields.io/badge/PostGIS-336791?style=flat-square&logo=postgresql&logoColor=white">
<img src="https://img.shields.io/badge/Redis-DC382D?style=flat-square&logo=redis&logoColor=white">
<img src="https://img.shields.io/badge/Celery-37814A?style=flat-square&logo=celery&logoColor=white">
<img src="https://img.shields.io/badge/Supabase-3FCF8E?style=flat-square&logo=supabase&logoColor=white">
<img src="https://img.shields.io/badge/Node.js-339933?style=flat-square&logo=nodedotjs&logoColor=white">
<img src="https://img.shields.io/badge/Pydantic-E92063?style=flat-square&logo=pydantic&logoColor=white">
</td>
</tr>
<tr>
<td valign="top"><b>Frontend</b></td>
<td>
<img src="https://img.shields.io/badge/React-61DAFB?style=flat-square&logo=react&logoColor=black">
<img src="https://img.shields.io/badge/React_Native-61DAFB?style=flat-square&logo=react&logoColor=black">
<img src="https://img.shields.io/badge/Next.js-000000?style=flat-square&logo=nextdotjs&logoColor=white">
<img src="https://img.shields.io/badge/WebGPU-005A9C?style=flat-square&logo=webgpu&logoColor=white">
<img src="https://img.shields.io/badge/three.js-000000?style=flat-square&logo=threedotjs&logoColor=white">
</td>
</tr>
<tr>
<td valign="top"><b>Infra &amp; Tooling</b></td>
<td>
<img src="https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white">
<img src="https://img.shields.io/badge/Linux-FCC624?style=flat-square&logo=linux&logoColor=black">
<img src="https://img.shields.io/badge/Git-F05032?style=flat-square&logo=git&logoColor=white">
<img src="https://img.shields.io/badge/GitHub_Actions-2088FF?style=flat-square&logo=githubactions&logoColor=white">
<img src="https://img.shields.io/badge/CMake-064F8C?style=flat-square&logo=cmake&logoColor=white">
<img src="https://img.shields.io/badge/pytest-0A9EDC?style=flat-square&logo=pytest&logoColor=white">
<img src="https://img.shields.io/badge/AWS-FF9900?style=flat-square&logo=amazonaws&logoColor=white">
</td>
</tr>
</table>

<br>

<div align="center">
  <sub><b>B.A.Sc. Computer Engineering, University of Toronto</b> &nbsp;&#183;&nbsp; Minor in AI Engineering &nbsp;&#183;&nbsp; Class of 2028</sub>
  <br><br>
  <a href="https://www.linkedin.com/in/msalman06"><b>Let's build something.</b></a>
</div>
