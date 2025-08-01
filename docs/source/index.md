:::{figure} ./assets/logos/angelslim_logo.png
:align: center
:alt: AngelSlim
:class: no-scaled-link
:width: 60%
:::

:::{raw} html
<p style="text-align:center">
<strong>Efficient LLM Compression Toolkit
</strong>
</p>

<p style="text-align:center">
<script async defer src="https://buttons.github.io/buttons.js"></script>
<a class="github-button" href="https://github.com/Tencent/AngelSlim" data-show-count="true" data-size="large" aria-label="Star">Star</a>
<a class="github-button" href="https://github.com/Tencent/AngelSlim/subscription" data-icon="octicon-eye" data-size="large" aria-label="Watch">Watch</a>
<a class="github-button" href="https://github.com/Tencent/AngelSlim/fork" data-icon="octicon-repo-forked" data-size="large" aria-label="Fork">Fork</a>
</p>
:::

AngelSlim是腾讯自研的，致力于打造更易用、更全面和更高效的大语言模型压缩工具包。我们将开源量化、投机采样、稀疏化和蒸馏等压缩算法。覆盖主流最前沿的大模型，并且端到端打通从压缩到部署的全流程。

(AngelSlim, developed by Tencent, is a large language model compression toolkit engineered for enhanced usability, comprehensiveness, and efficiency. We will open-source compression algorithms including quantization, speculative decoding, pruning, and distillation. Supporting cutting-edge mainstream LLMs, the toolkit streamlines the complete end-to-end workflow from compression to deployment.)


🌟AngelSlim主要特性有：

- **高度集成化**：本工具将主流的压缩算法集成到工具，开发者可一键式调用，具有很好的易用性。
- **持续算法创新**：本工具除了集成工业界使用最广的算法，还持续自研更好的压缩算法，并且会陆续开源。
- **追求极致性能**：在模型压缩流程、压缩算法部署方面，本工具持续端到端优化，致力于用更少的成本压缩部署大模型。

目前支持的压缩策略：

- **量化**：动态INT8、静态FP8、动态FP8、INT4-GPTQ、INT4-AWQ等方法；
- **投机采样**：EAGLE2、EAGLE3等方法。


## 文档

% How to start using AngelSlim?

:::{toctree}
:caption: 入门指南
:maxdepth: 1

getting_started/installation
getting_started/quickstrat
:::

% Additional capabilities

:::{toctree}
:caption: 算法特性
:maxdepth: 1

features/quantization/index
features/speculative_decoding/index
features/cache/index
:::

% Additional capabilities

:::{toctree}
:caption: 模型支持
:maxdepth: 1

models/hunyuan/hunyuan_quant
models/deepseek/deepseek_quant
models/qwen/qwen_quant
models/flux/flux_cache
:::


% Scaling up AngelSlim for production

:::{toctree}
:caption: 部署文档
:maxdepth: 1

deployment/deploy

:::

% Making the most out of AngelSlim

:::{toctree}
:caption: 性能表现
:maxdepth: 1

performance/quantization/benchmarks
performance/speculative_decoding/benchmarks
:::

% Explanation of AngelSlim internals

:::{toctree}
:caption: 设计文档
:maxdepth: 1

design/index
:::

## 更多

想了解更多信息，可以给我们在[GitHub Issues](https://github.com/Tencent/AngelSlim/issues)上留言，也可以加入我们的[微信交流群](./docs/source/assets/angel_slim_wechat.png)讨论更多的技术问题。