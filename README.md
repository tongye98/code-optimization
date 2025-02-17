# code-optimization
Source code for "A Problem-Oriented Perspective and Anchor Verification for Code Optimization"

## Abstract 
Large language models (LLMs) have shown remarkable capabilities in solving various programming tasks, such as code generation. However, their potential for code optimization, particularly in performance enhancement, remains largely unexplored. This paper investigates the capabilities of LLMs in optimizing code for minimal execution time, addressing a critical gap in current research. The recently proposed code optimization dataset constructs program optimization pairs based on iterative submissions from the same programmer for the same problem. However, this approach limits LLMs to local performance improvements, neglecting global algorithmic innovation. To overcome this limitation, we adopt a completely different perspective by reconstructing the optimization pairs into a problem-oriented approach. This allows for the integration of various ideas from multiple programmers tackling the same problem. Experimental results demonstrate that adapting LLMs to problem-oriented optimization pairs significantly enhances their optimization capabilities. Furthermore, recognizing the inherent trade-offs in code optimization, we introduce an anchor verification mechanism to mitigate the "optimization tax". Ultimately, our approach elevates both the optimization ratio and speedup to new levels.

## Code
```
cd src
```
###  make-pco
Python script to construct the problem-oriented (PCO) dataset and the original user-oriented (PIE) dataset.

### finetuning
config file of LlamaFactory[https://github.com/hiyouga/LLaMA-Factory] for adative finetuning on LLMs.

###  generate_process
1. Compilation and correctness check
2. Execution time benchmarking

###  metric_analysis
Calculation of all metrics

### anchor verfication
A novel anchor verification framework that leverages the original "slow code" as a reliable test case verification anchor. Unlike the code generation domain, which may rely on potentially error-prone synthetic test cases for refinement, the code optimization scenario has a unique advantage: the "slow code", despite its inefficiency, is functionally correct. This inherent characteristic positions it as an ideal test case verification anchor.


###  merge
Script for merging LLM models (Optional)
