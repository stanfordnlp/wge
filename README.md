**Workflow-Guided Exploration (WGE)**
is a framework for exploring action sequences more efficiently
when a small amount of demonstrations are available.
It helps a reinforcement learning (RL) agent discover reward more quickly
even when the reward is sparse.

# Motivation and Setup

Our motivating task is learning an RL agent to use the Internet
by controlling the web browser.
Here is a simple example where the agent has to forward Bob's email to Alice:

![email-inbox task](img/email.png)

The input goal is different for each episode,
and there can be multiple subtasks (e.g., forward an email, reply to an email, delete an email).
The agent receives a **sparse binary reward** at the end of the episode.

To aid learning,
suppose we also have access to a few (e.g., 10) human **demonstrations**
of how to complete the tasks.

# Framework Overview

Instead of directly training a model on the demonstrations
(which would lead to overfitting),
we use the demonstrations to **constrain exploration**.

![email-inbox task](img/workflow-lattice.png)

* From each demonstration, we induce high-level **workflows**.
  A workflow (a path through the workflow lattice, like the one in bold)
  constrains the set of possible actions at each time step
  (e.g., the workflow step `Click(Tag("span"))` only allows actions
  that click a `<span>` element).
  We generate many workflows with different levels of restrictiveness.

* When we perform exploration, we follow one of the workflows
  and sample actions that are within the constraints of the workflow.
  This way, the explored actions look similar to the demonstrated actions,
  so they are more likely to get positive reward.

* But there are many possible workflows, many of which are too generic or too restrictive.
  We use a simple **workflow policy** to learn which workflows are good to use.

# Demonstrations

Evaluation is done on OpenAI [Mini World of Bits](http://alpha.openai.com/miniwob/) benchmark.
To aid further research, we have [packaged and augmented the benchmark (**MiniWoB++**)](https://github.com/stanfordnlp/miniwob-plusplus)
and [released crowdsourced demonstrations](https://github.com/stanfordnlp/miniwob-plusplus-demos).

Some results of the models learned using WGE, compared with models that use behavioral cloning + RL, are shown here:

| WGE (10 demos) | BC+RL (100 demos) | BC+RL (300 demos) | BC+RL (1000 demos) |
| -- | -- | -- | -- |
| ![social-media_wge](workflows/social-media_wge.gif) | ![social-media_100](workflows/social-media_100.gif) | ![social-media_300](workflows/social-media_300.gif) | ![social-media_1000](workflows/social-media_1000.gif) |
| ![enter-time_wge](workflows/enter-time_wge.gif) | ![enter-time_100](workflows/enter-time_100.gif) | ![enter-time_300](workflows/enter-time_300.gif) | ![enter-time_1000](workflows/enter-time_1000.gif) |
| ![click-checkboxes-large_wge](workflows/click-checkboxes-large_wge.gif) | ![click-checkboxes-large_100](workflows/click-checkboxes-large_100.gif) | ![click-checkboxes-large_300](workflows/click-checkboxes-large_300.gif) | ![click-checkboxes-large_1000](workflows/click-checkboxes-large_1000.gif) |
| ![click-checkboxes-soft_wge](workflows/click-tab-2-hard_wge.gif) | ![click-checkboxes-soft_100](workflows/click-tab-2-hard_100.gif) | ![click-checkboxes-soft_300](workflows/click-checkboxes-soft_300.gif) | ![click-checkboxes-soft_1000](workflows/click-checkboxes-soft_1000.gif) |
| ![email-inbox-nl-turk_wge](workflows/email-inbox-nl-turk_wge.gif) | ![email-inbox-nl-turk_100](workflows/email-inbox-nl-turk_100.gif) | ![email-inbox-nl-turk_300](workflows/email-inbox-nl-turk_300.gif) | ![email-inbox-nl-turk_1000](workflows/email-inbox-nl-turk_1000.gif) |


# References

Evan Zheran Liu\*, Kelvin Guu\*, Panupong (Ice) Pasupat\*, Tianlin Shi, Percy Liang.
[**Reinforcement Learning on Web Interfaces using Workflow-Guided Exploration**](https://arxiv.org/abs/1802.08802).
ICLR 2018.
