# HERO: History-awarE fRamewOrk for Automated Code Review

## Introduction

Code review helps maintain software quality. Although automated review methods increasingly use language models and broader context, the evolution of a target change across preceding commits in the same pull request (PR) remains less explored. HERO (History-awarE fRamewOrk) constructs PR-level historical context from preceding code changes and commit information for Review Necessity Prediction (RNP) and Code Review Comment Generation (CRCG).

We construct HistoryCR from 29,986 PRs, 128,897 commits, and 592,973 code changes. Across five pre-trained language models, HERO improves average RNP accuracy by 7.97 percentage points on Java and 10.10 percentage points on C++, and increases CRCG BLEU-4 by 1.45 and 3.03 points, respectively. Experiments with four large language models provide further evidence. An exploratory user study complements the automatic metrics by assessing the perceived usefulness of generated review comments; its findings indicate practical potential rather than broad generalizability.

---

## Dataset and Source Repositories

[Download HistoryCR](https://drive.google.com/file/d/1Cn_6mISIHd3QN-U3cFdjUK_vUuH8zjlQ/view?usp=sharing). The dataset preserves three levels of GitHub pull-request data: PR identifiers, dates, titles, and descriptions; commit SHAs, dates, messages, filenames, and code changes; and inline review comments associated with reviewed diff hunks. For each reviewed change, HERO uses preceding commits to the same file within the same PR as historical context. These records support the RNP and CRCG tasks.

HistoryCR was collected from 41 Java and 46 C++ GitHub repositories with substantial review activity. The source repositories are listed below.

<details>
<summary>Java repositories (41)</summary>

- [alibaba/nacos](https://github.com/alibaba/nacos)
- [Anuken/Mindustry](https://github.com/Anuken/Mindustry)
- [apache/dolphinscheduler](https://github.com/apache/dolphinscheduler)
- [apache/doris](https://github.com/apache/doris)
- [apache/druid](https://github.com/apache/druid)
- [apache/dubbo](https://github.com/apache/dubbo)
- [apache/flink](https://github.com/apache/flink)
- [apache/hadoop](https://github.com/apache/hadoop)
- [apache/incubator-seata](https://github.com/apache/incubator-seata)
- [apache/kafka](https://github.com/apache/kafka)
- [apache/pulsar](https://github.com/apache/pulsar)
- [apache/rocketmq](https://github.com/apache/rocketmq)
- [apache/shardingsphere](https://github.com/apache/shardingsphere)
- [apache/skywalking](https://github.com/apache/skywalking)
- [bazelbuild/bazel](https://github.com/bazelbuild/bazel)
- [dataease/dataease](https://github.com/dataease/dataease)
- [dbeaver/dbeaver](https://github.com/dbeaver/dbeaver)
- [deeplearning4j/deeplearning4j](https://github.com/deeplearning4j/deeplearning4j)
- [doocs/leetcode](https://github.com/doocs/leetcode)
- [elastic/elasticsearch](https://github.com/elastic/elasticsearch)
- [elastic/logstash](https://github.com/elastic/logstash)
- [eugenp/tutorials](https://github.com/eugenp/tutorials)
- [jenkinsci/jenkins](https://github.com/jenkinsci/jenkins)
- [kestra-io/kestra](https://github.com/kestra-io/kestra)
- [keycloak/keycloak](https://github.com/keycloak/keycloak)
- [libgdx/libgdx](https://github.com/libgdx/libgdx)
- [neo4j/neo4j](https://github.com/neo4j/neo4j)
- [netty/netty](https://github.com/netty/netty)
- [OpenAPITools/openapi-generator](https://github.com/OpenAPITools/openapi-generator)
- [openjdk/jdk](https://github.com/openjdk/jdk)
- [oracle/graal](https://github.com/oracle/graal)
- [prestodb/presto](https://github.com/prestodb/presto)
- [quarkusio/quarkus](https://github.com/quarkusio/quarkus)
- [questdb/questdb](https://github.com/questdb/questdb)
- [ReactiveX/RxJava](https://github.com/ReactiveX/RxJava)
- [SeleniumHQ/selenium](https://github.com/SeleniumHQ/selenium)
- [spring-projects/spring-boot](https://github.com/spring-projects/spring-boot)
- [spring-projects/spring-framework](https://github.com/spring-projects/spring-framework)
- [TheAlgorithms/Java](https://github.com/TheAlgorithms/Java)
- [thingsboard/thingsboard](https://github.com/thingsboard/thingsboard)
- [zaproxy/zaproxy](https://github.com/zaproxy/zaproxy)

</details>

<details>
<summary>C++ repositories (46)</summary>

- [apache/arrow](https://github.com/apache/arrow)
- [apache/mxnet](https://github.com/apache/mxnet)
- [ApolloAuto/apollo](https://github.com/ApolloAuto/apollo)
- [apple/foundationdb](https://github.com/apple/foundationdb)
- [bitcoin/bitcoin](https://github.com/bitcoin/bitcoin)
- [carbon-language/carbon-lang](https://github.com/carbon-language/carbon-lang)
- [ceph/ceph](https://github.com/ceph/ceph)
- [ClickHouse/ClickHouse](https://github.com/ClickHouse/ClickHouse)
- [cocos2d/cocos2d-x](https://github.com/cocos2d/cocos2d-x)
- [dmlc/xgboost](https://github.com/dmlc/xgboost)
- [dragonflydb/dragonfly](https://github.com/dragonflydb/dragonfly)
- [duckdb/duckdb](https://github.com/duckdb/duckdb)
- [electron/electron](https://github.com/electron/electron)
- [envoyproxy/envoy](https://github.com/envoyproxy/envoy)
- [esp8266/Arduino](https://github.com/esp8266/Arduino)
- [espressif/arduino-esp32](https://github.com/espressif/arduino-esp32)
- [ethereum/solidity](https://github.com/ethereum/solidity)
- [facebook/hhvm](https://github.com/facebook/hhvm)
- [facebook/react-native](https://github.com/facebook/react-native)
- [facebook/rocksdb](https://github.com/facebook/rocksdb)
- [FreeCAD/FreeCAD](https://github.com/FreeCAD/FreeCAD)
- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
- [godotengine/godot](https://github.com/godotengine/godot)
- [google/filament](https://github.com/google/filament)
- [grpc/grpc](https://github.com/grpc/grpc)
- [LadybirdBrowser/ladybird](https://github.com/LadybirdBrowser/ladybird)
- [MaaAssistantArknights/MaaAssistantArknights](https://github.com/MaaAssistantArknights/MaaAssistantArknights)
- [MarlinFirmware/Marlin](https://github.com/MarlinFirmware/Marlin)
- [microsoft/LightGBM](https://github.com/microsoft/LightGBM)
- [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime)
- [microsoft/react-native-windows](https://github.com/microsoft/react-native-windows)
- [microsoft/terminal](https://github.com/microsoft/terminal)
- [NixOS/nix](https://github.com/NixOS/nix)
- [notepad-plus-plus/notepad-plus-plus](https://github.com/notepad-plus-plus/notepad-plus-plus)
- [opencv/opencv](https://github.com/opencv/opencv)
- [OpenRCT2/OpenRCT2](https://github.com/OpenRCT2/OpenRCT2)
- [osquery/osquery](https://github.com/osquery/osquery)
- [PaddlePaddle/Paddle](https://github.com/PaddlePaddle/Paddle)
- [qbittorrent/qBittorrent](https://github.com/qbittorrent/qBittorrent)
- [RPCS3/rpcs3](https://github.com/RPCS3/rpcs3)
- [scylladb/scylladb](https://github.com/scylladb/scylladb)
- [SerenityOS/serenity](https://github.com/SerenityOS/serenity)
- [swiftlang/swift](https://github.com/swiftlang/swift)
- [taichi-dev/taichi](https://github.com/taichi-dev/taichi)
- [tensorflow/tensorflow](https://github.com/tensorflow/tensorflow)
- [xbmc/xbmc](https://github.com/xbmc/xbmc)

</details>

---

## Table of Contents

- [Project Structure](#project-structure)
- [Dataset and Source Repositories](#dataset-and-source-repositories)
- [PLM Launcher](#plm-launcher)
- [Qwen3-4B Training](#qwen3-4b-training)
- [Task 1: Review Necessity Prediction (RNP)](#task-1-review-necessity-prediction-rnp)
  - [Using Pre-trained Language Models (PLMs)](#rnp-using-pre-trained-language-models-plms)
  - [Using Large Language Models (LLMs)](#rnp-using-large-language-models-llms)
- [Task 2: Code Review Comment Generation (CRCG)](#task-2-code-review-comment-generation-crcg)
  - [Using Pre-trained Language Models (PLMs)](#crcg-using-pre-trained-language-models-plms)
  - [Using Large Language Models (LLMs)](#crcg-using-large-language-models-llms)
- [Evaluation Metrics](#evaluation-metrics)
- [User Study](#user-study)

---


## Project Structure

```
HERO/
├── README.md
├── run_plm.py                    # Unified launcher for all five PLMs
├── run_qwen.py                   # Qwen3-4B training and dev-selected testing
├── qwen/                        # Causal training, generation metrics, selection
├── RNP/                          # Review Necessity Prediction
│   ├── run_defect.py             # Training script for encoder-only PLMs (RoBERTa, CodeBERT)
│   ├── run_gen.py                # Training script for encoder-decoder PLMs (CodeT5, CodeReviewer)
│   ├── uni_run.py                # Training script for UniXcoder
│   ├── llm.py                    # Inference script for LLMs
│   ├── configs.py                # Configuration and argument parsing
│   ├── models.py                 # Model definitions
│   ├── utils.py                  # Data loading utilities
│   ├── cal_metrics.py            # Metrics calculator for PLM outputs
│   └── transform_cal_metrics.py  # Metrics calculator for LLM outputs
├── CRCG/                         # Code Review Comment Generation
│   ├── run_gen.py                # Training script for encoder-decoder PLMs
│   ├── uni_run.py                # Training script for UniXcoder
│   ├── llm.py                    # Inference script for LLMs
│   ├── configs.py                # Configuration and argument parsing
│   ├── models.py                 # Model definitions
│   ├── utils.py                  # Data loading utilities
│   ├── cal_metrics.py            # Generation metrics calculator
│   └── llm.sh                    # Batch inference script for LLMs
```

---

## PLM Launcher

Run commands from the repository root. `run_plm.py` accepts `roberta`, `codebert`, `codet5`, `codereviewer`, or `unixcoder` for `--model`, and `original`, `cc`, `msg`, or `all` for `--mode`. It trains, validates, and tests by default. Use `--no-test` to train without testing, or `--no-train` to test checkpoints already saved under the same output path. `--dry-run` prints the underlying command without starting it. Run `python run_plm.py --help` for training overrides.

The dataset layout is `<data-root>/<language>/<task>/<mode>/{train,dev,test}.json`. Outputs go to `<output-root>/<task>/<language>/<mode>/<model>`; the default output root is `outputs/` in this repository. For CodeBERT, `--tokenizer-path` can point to a separate RoBERTa tokenizer.

---

## Qwen3-4B Training

`run_qwen.py` fine-tunes Qwen3-4B for RNP or CRCG. Install a CUDA-enabled PyTorch build for your GPU server, then run `pip install -r qwen/requirements.txt`. The default configuration uses eight GPUs, BF16, and DeepSpeed ZeRO-3; `--num-gpus` and `--gpus` select the available devices. The data root must contain `<language>/<TASK>/<variant>/{train,dev,test}.json` JSONL files, or each file can be supplied explicitly with `--train-file`, `--dev-file`, and `--test-file`.

```bash
python run_qwen.py --task rnp --language java --variant hero \
  --data-root /path/to/dataset --model-name-or-path /path/to/Qwen3-4B \
  --context-length 1024 --num-gpus 8 --gpus 0,1,2,3,4,5,6,7
```

For CRCG, use `--task crcg` and the corresponding data directory. `--dry-run` prints both commands without loading the model. Training saves one candidate checkpoint per epoch. The selection stage evaluates candidates on **dev only**, choosing the highest F1 for RNP or BLEU-4 for CRCG. It exports that checkpoint to `best_model/` and runs **test once** with the exported model. After a successful test, intermediate checkpoints are removed unless `--keep-checkpoints` is set. If selection is interrupted, rerun the command with `--selection-only` to use the saved epoch checkpoints.

The output directory contains `train.log`, `selection.log`, `epoch_metrics.jsonl`, `selection.json`, `best_model/`, and `evaluation/{dev,test}_metrics.json`; test predictions are in `evaluation/test_predictions.jsonl`. The default output root is `outputs/qwen/`. These scripts are separate from the Qwen2.5-Coder inference examples below.

---

## Task 1: Review Necessity Prediction (RNP)

Review Necessity Prediction is a **binary classification** task that predicts whether a code change requires review from other developers.

### RNP: Using Pre-trained Language Models (PLMs)

Use the same launcher for RoBERTa, CodeBERT, CodeT5, CodeReviewer, and UniXcoder:

```bash
python run_plm.py --task RNP --model codebert --language cpp --mode original \
  --data-root /path/to/dataset --model-path /path/to/codebert-base \
  --tokenizer-path /path/to/roberta-base --gpu 0
```

### RNP: Using Large Language Models (LLMs)

We support **4 Large Language Models**:

| Model | Parameters |
|-------|------------|
| CodeLlama-7b-Instruct | 7B |
| DeepSeek-Coder-7b-Instruct | 7B |
| Yi-Coder-9B-Chat | 9B |
| Qwen2.5-Coder-7B-Instruct | 7B |

#### Running LLM Inference

```bash
cd RNP

python llm.py \
    --model_name "CodeLlama-7b" \
    --language "cpp" \
    --gpu_id "0" \
    --models_base /path/to/models \
    --dataset_base /path/to/dataset \
    --epochs 1
```

**Parameters:**
- `--model_name`: One of `CodeLlama-7b`, `DeepSeek-Coder-7b`, `Yi-Coder-9b`, `Qwen2.5-Coder-7b`
- `--language`: `cpp` or `java`
- `--gpu_id`: GPU device ID
- `--models_base`: Directory containing model folders
- `--dataset_base`: Directory containing dataset folders
- `--epochs`: Number of experiment runs (default: 1, with temperature=0 for deterministic output)

The script will automatically run experiments on all 4 input modes: `original`, `msg`, `cc`, `all`.

#### Calculating Metrics for LLM Outputs

```bash
cd RNP/llm_results/<model_name>

# For LLM outputs (parses YES/NO responses)
python ../../transform_cal_metrics.py
```

---

## Task 2: Code Review Comment Generation (CRCG)

Code Review Comment Generation is a **sequence-to-sequence** task that generates review comments for given code changes.

### CRCG: Using Pre-trained Language Models (PLMs)

Use the same launcher and change the task and model:

```bash
python run_plm.py --task CRCG --model unixcoder --language java --mode all \
  --data-root /path/to/dataset --model-path /path/to/unixcoder-base --gpu 0
```

### CRCG: Using Large Language Models (LLMs)

We support the same **4 Large Language Models** as RNP:

| Model | Parameters |
|-------|------------|
| CodeLlama-7b-Instruct | 7B |
| DeepSeek-Coder-7b-Instruct | 7B |
| Yi-Coder-9B-Chat | 9B |
| Qwen2.5-Coder-7B-Instruct | 7B |

#### Running LLM Inference

```bash
cd CRCG

python llm.py \
    --model_name "CodeLlama-7b" \
    --language "cpp" \
    --gpu_id "0" \
    --model_path /path/to/CodeLlama-7b-Instruct-hf \
    --data_dir /path/to/dataset \
    --epochs 1
```

**Parameters:**
- `--model_name`: One of `CodeLlama-7b`, `DeepSeek-Coder-7b`, `Yi-Coder-9b`, `Qwen2.5-Coder-7b`
- `--language`: `cpp` or `java`
- `--gpu_id`: GPU device ID
- `--model_path`: Path to the model directory
- `--data_dir`: Path to the dataset directory
- `--epochs`: Number of experiment runs (default: 1, with temperature=0 for deterministic output)

#### Batch LLM Inference

```bash
cd CRCG

# Run all LLMs in parallel on multiple GPUs
bash llm.sh
```

#### Calculating Metrics for LLM Outputs

```bash
cd CRCG/llm_results/<model_name>

# For generation outputs
python ../../cal_metrics.py
```

---

## Evaluation Metrics

### RNP Metrics (Classification)

| Metric | Description |
|--------|-------------|
| Accuracy | Overall prediction accuracy |
| Precision | Ratio of true positives to predicted positives |
| Recall | Ratio of true positives to actual positives |
| F1-Score | Harmonic mean of precision and recall |

### CRCG Metrics (Generation)

| Metric | Description |
|--------|-------------|
| BLEU | N-gram overlap between generated and reference text |
| ROUGE-L | Longest common subsequence based metric |
| Edit Similarity | Levenshtein distance based similarity |
| SacreBLEU | Standardized BLEU implementation |

---

## Input Modes

We evaluate models under 4 different input modes:

| Mode | Description |
|------|-------------|
| `original` | Only current code change |
| `msg` | Current code change + Historical commit messages |
| `cc` | Current code change + Historical code changes |
| `all` | Current code change + All historical information |

---

## Training Workflow

All training scripts follow a standard workflow:

1. **Training Phase**: Train the model on training set
2. **Validation Phase**: Evaluate on validation set after each epoch
3. **Checkpoint Saving**: Save best model based on validation metrics
4. **Early Stopping**: Stop training if no improvement for `patience` epochs
5. **Testing Phase**: Evaluate saved checkpoints on the test set, either after training or in a separate run with `--no-train`

This design ensures no data leakage from test set during training.

### Acknowledgments

The training and implementation of PLMs in this project are inspired by and partially based on the following open-source repositories:

- [CodeT5](https://github.com/salesforce/CodeT5) - Salesforce's CodeT5 model and training pipeline
- [UniXcoder](https://github.com/microsoft/CodeBERT/tree/master/UniXcoder) - Microsoft's UniXcoder model implementation

We sincerely thank the authors for making their code publicly available.

---

## User Study

The exploratory CRCG study involved 20 participants, each evaluating 20 cases randomly selected from the CRCG test set. Each case paired a target code change with comments generated by the original CodeT5 and CodeT5 augmented with HERO. The comments were anonymized and shown in randomized order. Participants rated each comment on a 1–5 scale for how meaningfully it identified an issue in the change, considering relevance, specificity, and actionability. This produced 400 paired evaluations (800 individual ratings).

The [sample questionnaire](./User_Study_Evaluation_of%20Code_Review_Comment.pdf) illustrates the question and rating format. It is a formatting example, not the fixed set of 20 cases administered to every participant; each participant received their own random sample.
