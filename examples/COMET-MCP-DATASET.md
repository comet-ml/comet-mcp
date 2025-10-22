# Comet MCP Dataset

This page describes making a dataset for testing comet-mcp.

## Setup

We'll use the files in the comet-mcp/examples folder:

```shell
git clone https://github.com/comet-ml/comet-mcp.git
```

First, we'll create the dataset:

```shell
cd comet-mcp/examples
python create_test_dataset.py
```

And then we'll make some comet_ml experiments so that we
can test the comet-mcp server tools.

```shell
python create_test_experiments.py
```

Please reach out if you have any questions!