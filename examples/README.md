# How to Use, Test, and Optimize your MCP server

These instructions describe taking an existing MCP server, and making
it better.

Note: the [`ez-mcp-tools`](https://pypi.org/project/ez-mcp-toolbox/) are built upon [Opik](https://www.comet.com/docs/opik/) functionality. These
provide quick command-line easy options. More experienced developers
will want to dive into Opik directly and the use the [Opik UI](https://www.comet.com/docs/opik/). In fact,
you can do much of what is described here in the Opik UI without
writing and code at all.

## Setup

In order to use, test, and optimize your MCP server, we'll be using `ez-mcp-tools`:

```shell
pip install ez-mcp-tools --upgrade
```

This will allow us to easily try out your MCP server interactively
using `ez-mcp-chatbot`.

For all of the following commands, we'll need a config that tells the
command how to start your MCP server. For this example, we'll use
`comet-mcp`, but it could be any MCP server. By default, this uses
"stdio" and that is the default of the ez-mcp-tools as well.

To use the `comet-mcp` server, you'll need to setup your [Comet ML](https://www.comet.com/)
environment, have some experiments logged to your Comet installation, and:

```shell
# Only needed if you are using the comet-mcp server
pip install comet-mcp --upgrade
```

One of the most important features that these tools have is that they log
the LLM processing to [Opik](https://www.comet.com/docs/opik/), an open source, fully self-contained
server and UI for managing your LLM tasks. To set it up (if you
haven't already):

```shell
opik configure
```

To use the cloud version (option 1), you'll need a [comet.com account](https://www.comet.com/)
(free). You can also use a self-hosted Opik server, or a local
deployment.

To begin to use the tools, we'll need to create the contents of
`ez-config.json` with your MCP server command and args. You can set
your model and model kwargs here as well. We're using [litellm](https://docs.litellm.ai/) so
make sure your model name is a litellm model:

```json
{
  "model": "openai/gpt-4o-mini",
  "model_kwargs": {
    "temperature": 0.0
  },
  "mcp_servers": [
    {
      "name": "comet-mcp",
      "description": "Comet ML MCP server",
      "command": "comet-mcp",
      "args": []
    }
  ]
}
```

## Use your MCP server

Does your MCP server work? Let's test it out by interacting with it via a chatbot (run in the same directory as your config file):

```shell
ez-mcp-chatbot
```

In the chatbot, you'll see some start-up messages and the `>>>` user
prompt. Ask a question that requires your tools to answer.

Note that `ez-mcp-chatbot` supports:

* command-line history
* slash commands (like "/show tools")
* Python evaluation with "!"

Try out questions that use your MCP tools, especially questions that
require multiple tool use to answer.

As you are interacting with your chatbot and MCP server, your logged
interactions are being sent to the [Opik server](https://www.comet.com/docs/opik/). Here, you can explore
and debug the exact trace of the LLM actions.

Many MCP servers may also need global instructions to help use the
tools in the MCP server in the system prompt. We can pass in short
prompts on the command line, but many system prompts may be too
verbose, so we can create a file to use.

For this demo, we'll save the system prompt in an arbitrarily named
file named `PROMPT.md`:

**Contents of PROMPT.md:**
```markdown
You are an expert AI assistant specializing in machine learning experiment analysis using Comet ML.

You help researchers analyze their experiments from named projects in the default workspace, providing insights, recommendations, and answering questions about their ML experiments.

Always provide actionable insights and concrete recommendations for improving experiment results.
```

Now, we can use the chatbot again, this time adding in the system prompt:

```shell
ez-mcp-chatbot --prompt PROMPT.md
```

## Test your MCP Server

Ok, now that we know that it at least functional, how does it perform?
We can try specific questions interactively in the chatbot, and
explore their traces in Opik. But to be scientific, we need a dataset.

The rest of this document requires a dataset. It takes a little bit of
work, but it is worth it. We can test variations of system prompt, and
tool prompts. But we can also optimize these as well automatically!

You can create a dataset interactively in the [Opik UI](https://www.comet.com/docs/opik/). Or you can
create one programmatically. You don't need more than a few questions
to get started.

For those using `comet-mcp`, you can create test datasets programmatically or through the [Opik UI](https://www.comet.com/docs/opik/).

An example dataset item looks like:

```json
{
    "question": "What learning rate was configured for the resnet50_baseline experiment in our comet_mcp-tests experiment?",
    "answer": "The resnet50_baseline experiment used a learning rate of 0.001. This is a standard learning rate for ResNet-50 training on comet mcp tests, providing good convergence while avoiding instability. The learning rate was kept constant throughout training without any scheduling."
}
```

We'll now assume that you have a dataset in [Opik](https://www.comet.com/docs/opik/). In these examples
we'll use the dataset `comet-mcp-tests-dataset` but you can substitute
that for your own.

Now that we have a dataset, we can test it with our prompt and MCP
tools. In addition to the dataset, we need to specify what field in
the dataset is the input or question.

Now we can automatically run the prompt + MCP server against the dataset:

```shell
ez-mcp-eval \
   --prompt PROMPT.md \
   --dataset comet-mcp-tests-dataset \
   --input question
```

You can look over the resulting answers in the [Opik UI](https://www.comet.com/docs/opik/).

But this isn't completely useful. It would be more useful if we knew which of the
generated answers was correct. For this, we need an [Opik Metric](https://www.comet.com/docs/opik/). There are a bunch
built in:

```python
ez-mcp-eval --list-metrics
```

Available metrics from [opik.evaluation.metrics](https://www.comet.com/docs/opik/):

   - AggregatedMetric
   - AnswerRelevance
   - BaseMetric
   - Contains
   - ContextPrecision
   - ContextRecall
   - ConversationThreadMetric
   - ConversationalCoherenceMetric
   - CorpusBLEU
   - Equals
   - GEval
   - Hallucination
   - IsJson
   - LevenshteinRatio
   - MetricComputationError
   - Moderation
   - ROUGE
   - RagasMetricWrapper
   - RegexMatch
   - SentenceBLEU
   - Sentiment
   - SessionCompletenessQuality
   - StructuredOutputCompliance
   - TrajectoryAccuracy
   - Usefulness
   - UserFrustrationMetric
```

For many metrics, we also need to know how to map the input of the
metric scoring method to the correct answer. For example, the
`LevenshteinRatio` metric's scoring method has the signature:

```python
LevenshteinRatio.score(output, reference)
```

The `output` here is the output of the LLM, and the `reference` is the
correct answer from the dataset. So we need to map the parameter
`reference` to the dataset field `answer` in my case. You dataset and
metric will probably be different.

So, we will add `--output reference=answer` to the command-line:

```shell
ez-mcp-eval \
   --prompt PROMPT.md \
   --dataset comet-mcp-tests-dataset \
   --input question \
   --output reference=answer \
   --metric LevenshteinRatio
```

Now we can see the metric values for each item in the dataset.

Note: this is a basic metric for gauging MCP answers. We'll explore
better metrics in a follow-up post. But it is good for demonstration
purposes.

You can tweak your prompt, and MCP tool descriptions to try to find
the best combination.

## Optimize your prompt

On the other hand, you can use one of [Opik's optimization algorithms](https://www.comet.com/docs/opik/) to
automatically find a better prompt. This command has a super-set of the ez-mcp-eval options, adding only one more: `--optimizer ALGORITHM`.

There are a number of algorithms you can try, including:

* [FewShotBayesianOptimizer](https://www.comet.com/docs/opik/agent_optimization/opik_optimizer/reference#fewshotbayesianoptimizer)
* [MetaPromptOptimizer](https://www.comet.com/docs/opik/agent_optimization/algorithms/metaprompt_optimizer)
* [EvolutionaryOptimizer](https://www.comet.com/docs/opik/agent_optimization/algorithms/evolutionary_optimizer)

and [many others](https://www.comet.com/docs/opik/agent_optimization/opik_optimizer/quickstart).

To find a better prompt using the `FewShotBayesianOptimizer` (using the algorithm defaults) you can:

```shell
ez-mcp-eval \
   --prompt PROMPT.md \
   --dataset comet-mcp-tests-dataset \
   --input question \
   --output reference=answer \
   --metric LevenshteinRatio \
   --optimizer FewShotBayesianOptimizer
```

This will create examples/demonstrations to better perform on this dataset.

There are many options available via the [ez-mcp-toolbox](https://pypi.org/project/ez-mcp-toolbox/) command-line tools. And much more power using the [Opik UI](https://www.comet.com/docs/opik/) and [SDK](https://www.comet.com/docs/opik/).