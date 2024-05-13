# OLLAMA LLMs and Planning

This repo is forked from [PlanBench](https://github.com/harshakokel/PlanBench) and modified so diverse models from OLLAMA can be evaluated. The repo also includes an [ngrok-google_colab](ollama_collab_ngrok.ipynb) integration in order to run the different models in the Computational Units of Google.

<!-- Models -->
## Models evaluated
The models evaluated are:
| Model        | Parameters           | Quantization  | Size |
| ------------- |:-------------:| :-----:|-----:|
| llama3      | 8B | 4-bit | 4.7 GB|
|       | 70B | 4-bit | 40 GB|
| llama2      | 7B | 4-bit | 3.8 GB| 
| tinyllama      | 1B | 4-bit | 638MB |
| phind-codellama      | 34B | 4-bit | 19 GB|
| command-r-plus      | 104B | 4-bit | 59 GB|
| wizardlm2      | 7B | 4-bit | 4.1 Gb|
| gemma      | 9B | 4-bit | 5 GB|
| vicuna      | 7B | 4-bit | 3.8 GB|


<!-- ROADMAP -->
## Roadmap
- [x] Run `ollama` in `google-collab`
- [x] Generate `ngrok-tunnel` in `google-collab`
- [x] Establish connection from local to remote ngrok serve
- [x] Add `ollama.Client(host=*)` in *plan-bench*
- [x] Get responses of a *ollama model*
- [ ] Evaluate responses of *ollama model*
  - [ ] **llm_plan** file is corrupted
     
- [ ] Multiple *ollama models* in a loop:
  - [ ] Get responses
      > :warning: **ERROR:** `ollama._types.ResponseError: pull model manifest: file does not exist`
      > [Bug from `ollama`: pulling from a PROXY causes errors](https://github.com/ollama/ollama/issues/1417)

- [ ] Benchmark the models:
  - [ ] Analyze responses
  - [ ] Compare size and cpu-time consumption of each model 





