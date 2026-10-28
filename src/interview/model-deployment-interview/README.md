---
title: 模型部署面试题
icon: mdi:cloud-outline
tag:
  - 模型部署面试
  - 面试题
  - 模型部署
---

# 模型部署面试题

大模型部署与推理优化相关的高频面试问题，共 12 篇文章，覆盖核心指标、推理加速、显存优化、并行策略和工程运维。

模型部署是大模型从训练到上线的最后一公里，也是工程能力的集中体现。面试中不仅要理解原理，还要讲清楚如何在生产环境中落地。

---

## 合集文章导航

**核心指标篇**
1. [大模型部署需要关注哪些核心指标？](/interview/model-deployment-interview/core-metrics.html) - 延迟、吞吐、显存和成本
2. [大模型推理中的首 Token 延迟和平均生成速度分别受什么影响？](/interview/model-deployment-interview/latency-throughput.html) - 性能瓶颈分析
3. [如何估算一个大模型部署需要多少显存？](/interview/model-deployment-interview/memory-estimation.html) - 显存估算方法

**推理加速篇**
4. [KV Cache 在模型部署中有什么作用？](/interview/model-deployment-interview/kv-cache-deploy.html) - 为什么它会占用大量显存
5. [vLLM 是什么？PagedAttention 为什么能提升推理吞吐？](/interview/model-deployment-interview/vllm-paged-attention.html) - vLLM 核心原理
6. [连续批处理 Continuous Batching 是什么？](/interview/model-deployment-interview/continuous-batching.html) - 为什么能提升并发能力
7. [模型量化是什么？](/interview/model-deployment-interview/quantization.html) - INT8、INT4、GPTQ、AWQ

**并行与分布式篇**
8. [Tensor Parallel、Pipeline Parallel 和 Data Parallel 有什么区别？](/interview/model-deployment-interview/parallel-strategies.html) - 三种并行策略对比

**工程运维篇**
9. [如何优化大模型服务的并发、吞吐和响应速度？](/interview/model-deployment-interview/optimization.html) - 性能优化实战
10. [OpenAI-Compatible API 是什么？](/interview/model-deployment-interview/openai-compatible-api.html) - 为什么很多本地模型服务要兼容它
11. [模型服务如何做限流、熔断、超时和降级？](/interview/model-deployment-interview/rate-limiting.html) - 服务稳定性保障
12. [部署大模型时如何做日志监控、调用统计和成本控制？](/interview/model-deployment-interview/monitoring-cost.html) - 运维与成本管理

