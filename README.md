<h1>Inductive Bias Under Weak Signal</h1>

<p>
This experiment probes how different temporal architectures behave
when trained on sequences with minimal predictive structure.
Rather than optimizing for real-world accuracy,
the goal is to observe representation formation,
failure modes, and inductive bias under extreme uncertainty.
</p>

<p>
The task is simple.
The environment is hostile.
The signal is weak by design.
</p>

<h2>Task Formulation</h2>
<p>
A univariate sequence of discrete symbols (digits 0–9) is transformed
into a supervised classification problem:
</p>

<ul>
  <li>Input: a fixed-length sliding window of past symbols</li>
  <li>Target: the next symbol in the sequence</li>
</ul>

<p>
No explicit causal structure is assumed.
Any performance must emerge from the model’s inductive bias alone.
</p>

<h2>Why This Data?</h2>
<p>
The data source is intentionally adversarial.
Digit streams derived from real-world processes with low or no predictive signal
are used to stress-test learning dynamics.
</p>

<ul>
  <li>Extremely low signal-to-noise ratio</li>
  <li>High entropy transitions</li>
  <li>Strong temptation to hallucinate patterns</li>
</ul>

<p>
This setup is designed to expose what models <em>want</em> to believe,
not what the data guarantees.
</p>

<h2>Architectures Explored</h2>
<p>
Multiple hybrid architectures are implemented to probe different biases:
</p>

<ul>
  <li>Attention + CNN blocks</li>
  <li>Attention + LSTM blocks</li>
  <li>Attention + GRU blocks</li>
</ul>

<p>
All models use stacked self-attention with residual connections,
followed by temporal processing layers.
</p>

<h2>Core Building Block</h2>
<pre>
Self-Attention →
Residual Connection →
Temporal Mixing (CNN / LSTM / GRU) →
Residual Connection
</pre>

<p>
This structure intentionally blurs the line between
Transformer-style context aggregation
and recurrent or convolutional inductive bias.
</p>

<h2>Feature Design</h2>
<p>
In addition to raw symbol windows,
optional derived features are explored:
</p>

<ul>
  <li>Distance to next occurrence</li>
  <li>Lag since last occurrence</li>
  <li>Rolling frequency statistics</li>
</ul>

<p>
These features encode weak memory signals
without injecting explicit domain knowledge.
</p>

<h2>Training Setup</h2>
<ul>
  <li>Mixed-precision training</li>
  <li>Flash attention enabled</li>
  <li>Class-balanced loss</li>
  <li>Early stopping on validation accuracy</li>
  <li>Learning-rate reduction on plateau</li>
</ul>

<p>
The objective is not maximum accuracy,
but stable training behavior under pressure.
</p>

<h2>What This Pushes Against</h2>
<ul>
  <li>Assumptions that deeper models always extract signal</li>
  <li>Benchmark-driven architecture selection</li>
  <li>Accuracy as the sole indicator of learning</li>
</ul>

<h2>Observed Failure Modes</h2>
<ul>
  <li>Rapid confidence saturation</li>
  <li>Class-frequency collapse</li>
  <li>Attention focusing on noise</li>
  <li>Overconfident misclassification</li>
</ul>

<p>
Misclassifications are analyzed explicitly to understand
how and where the model fails.
</p>

<h2>What This Is Not</h2>
<ul>
  <li>A claim that these sequences are predictable</li>
  <li>A production forecasting system</li>
  <li>A benchmark-optimized architecture</li>
</ul>

<h2>Why Explore This?</h2>
<p>
In many real systems, signal is weak, delayed, or deceptive.
Understanding how architectures behave in such regimes
is more valuable than optimizing clean benchmarks.
</p>

<p>
This experiment treats failure as data.
</p>

<blockquote>
When signal disappears, inductive bias takes over.
</blockquote>
