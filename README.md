# Particle: A General Programming Language Translator

Particle is a tool for translating your codebase from one programming language to another. The vision is that you snap your fingers and you suddenly have a codebase working with identical semantics in another language.

Achieving that dream is very much a research problem. The pared-down vision is for Particle to get deployed on enterprise codebases, where it will continuously translate code and verify it in a loop, until verification fails. When it fails, it will produce a PR with the failing translated code for a human to fix. When the code is fixed, the process will continue until translation is complete.

## Thesis

Coding assistants today often function as general helpers across broad domains, and what's interesting to me is doing a single, painful thing perfectly well (such as solving endemic types of tech debt) across a whole codebase.

I believe that by combining static analysis tools and AI, we can carve out precise interface boundaries within large enterprise codebases, and we can make targeted, verifiable improvements to the code within those boundaries that will make developers' lives better and save companies $MMs.

Enterprises frequently consider changing their programming languages for various strategic or technical reasons, yet historically, the switching costs are tremendous, often prohibibively so. Examples of past attempts illustrate the range of outcomes when companies attempt such transitions:

1. **Companies desire language changes but find it infeasible.**
   Facebook wrote their own compiler for PHP instead of switching to another language for this reason, which itself was a massive undertaking.

2. **Companies attempt large-scale rewrites with disastrous results.** Netscape decision to rewrite tfrom C to Java in the 90's was so problematic that the company never fully recovered. A [Meta TransCoder paper](https://scontent-lga3-1.xx.fbcdn.net/v/t39.8562-6/240880953_206259498226407_8788091695010975874_n.pdf?_nc_cat=103&ccb=1-7&_nc_sid=e280be&_nc_ohc=jsFAd9RlHiAQ7kNvgH-Jdk6&_nc_oc=AdmbYJTuZRh65Q3Vq8E5mT79WONxTlxO7MIzE8tHAParyWbGYYAeV6vTAIyEcfuVPrujkvGKMXUk2LPiW5nhF2fC&_nc_zt=14&_nc_ht=scontent-lga3-1.xx&_nc_gid=lrTpeTUZnM-r_09yeFa78g&oh=00_AYHK8hgEVgVGYRnBodO-76xnymMTsnnTwN_Lz7lm9qaglg&oe=67E34DC4) cites an Australian bank spending $750MM to translate from COBOL to Java. Wowzers.

3. **Companies successfully transition but remain dissatisfied.** Slack switched from native app development to a shared C++ lib (LibSlack) only to revert to their original strategy a year later, making net neutral progress and incurring massive costs.

Each of these examples underscores how riskly, costly, and unsatisfactory switching may be. Recent developments in AI may ease us into a future where these sorts of undertakings are viable.

## Architecture

Particle's goal is to decompose a codebase into its smallest translation subproblem until we find a verified translation. We do this by using static analysis to derive the codebase's call graph, working our way up the graph recursively as translations succeed. We maintain a translated codebase, which starts small, and we add each successful translation as it gets verified.

Although our call graph is by definition cyclic, we assume that a majority of a codebase's subgraphs are acyclic. Therefore, we identify the leaf nodes in the graph and start translation there, because they're the most independent and portable part of our codebase. These serve as atoms that we build upon, as demonstrated by the function format_price() and its corresponding unit tests in the following diagram:

![Call Graph](https://res.cloudinary.com/dwt45tvzy/image/upload/v1742510160/call_graph_jokbqy.svg)

Translation will naturally fail throughout the process, and we'll have humans fix it. I see having human "checkpoints" to fix bugs or otherwise improve code as being a feature, not a bug. I figure that the more a human is involved, the more they can course correct throughout the translation process, and the better the resulting codebase will be—with the goal being that nonetheless, this AI-assisted process reduces the time needed to translate a codebase from years to months. As such, the resulting human-in-the-loop control flow looks like this:

![Control Flow](https://res.cloudinary.com/dwt45tvzy/image/upload/v1742510160/architecture_tab3ir.svg)

To successfully translate all subtrees, we need individual components (e.g., the translation step and the apply step) to work very reliably. So to understand how much of a codebase we'd be able to auto-translate without interruption, we need comprehensive benchmarks. I've begun benchmarking using exercises in Exercism.

[Exercism](https://exercism.org/) is a collection of practice problems across programming languages, with sample answers and most importantly, unit tests in each language.

![Benchmark Results on Exercism](https://res.cloudinary.com/dwt45tvzy/image/upload/v1742569243/benchmark_results_chart_gpf6cp.png)

Amazingly, Claude 3.7 gets almost all the translations correct. o3-mini and deepseek-v3 perform poorly primarily due to their output formatting, and I just need to write a robust parser to handle their quirks better. I don't have a sense of how well they'll perform once I do that.

A caveat is that this benchmark bakes in the Exercism unit tests written in the target language. A challenge I've found with debugging LLM-generated code is that if we're translating groups of functions and unit tests, and tests fail, we won't necessarily know whether the tests or code (or both) failed to translate correctly. I wanted to control for that first and understand how much of a gap we'd have between translated tests and having tests where we know they're correct.

The next step is to benchmark actual production code.

## Components

`particle/translator.py`: The `Translator` class acts as the entry point and orchestrates the control flow described above, primarily using the help of `CallGraphAnalyzer`, `LLMTranslator`, and `FileManager`.

`particle/call_graph_analyzer.py`: CallGraphAnalyzer processes a codebase's AST to construct the call graph and stores the nodes so that they're accessible from `get_leaf_nodes()`.

`particle/file_manager.py`: Responsible for mapping files from the source codebase to the target codebase. This contains knowledge about the directory structure differences in each codebase.

`particle/llm_translator.py`: Responsible for interfacing with an LLM. Supports multi-turn structured output conversations for retrying when translations fail.

`particle/code_editor.py`: Responsible for applying new translations to the target codebase using LLMs, a la Cursor.

`particle/llm_results_parser.py`: Responsible for parsing code from LLM output.

`benchmark/main.py`: Entry point for benchmarking script.

`benchmark/visualize.py`: Tool to visualize the benchmark results, which produced the chart above.

## Limitations

### Architecture of translated codebase

I assume that a production-ready Particle would allow engineering teams to specify the architecture, directory structure, style guides, etc that they want in their target codebase and that we'd bake these into our prompts.

### Cyclical call graphs

A more advanced version of Particle would need to do cycle detection within the call graph and likely translate the whole cycle as a unit. This will be difficult to get right if the cycle spans a large enough subgraph.

### 3rd-party libraries

Particle's call graph sets a boundary between 1st-pary and 3rd-party code. An LLM won't necessarily know what to do with a third-party library call in the target language. Translating a codebase would require identifying key libraries anyway, and this would be work that would have to be research and likely integrated into Particles' prompts.

### Built-ins

LLMs seem to understand built-ins across programming languages reasonably well, likely due a density of training data for built-ins in each language. However, there are primitive across languages that won't be well represented. The [DARPA TRACTOR project](https://www.darpa.mil/research/programs/translating-all-c-to-rust) is tackling the gap in concurrency semantics across C and Rust, for instance.

## Alternative Architectures

### Parse less and prompt with more code

My architecture pieces together a lot of deterministic building blocks to translate and verify atomic units of code. With the size of context windows being what they are—e.g., Gemini models have 1-2MM token context windows—it might not necessary to be so precise in constructing an AST, and rather graph the file dependencies and shove whole files in to the prompt.

### Agentic approach

Recent examples like DeepSeek-R1 and ChatGPT Deep Research have demonstrated the power of RL use on frontier models. There could be a world where we fine-tune an LLM using RL, where the reward function is translating code and getting tests to pass. Then we use the agent to autonomously navigate the control flow I describe in my image above and translate the maximumally large subgraphs of the codebase that it's able to before getting stuck and deferring to a human.

## Alternative Applications

I believe we can apply the same core thesis—leveraging static analysis and AI to carve out interface boundaries within codebases and manipulate them—to increase automation with many other tasks, such as monolith decomposition or migrating major platforms, like an ORM.
