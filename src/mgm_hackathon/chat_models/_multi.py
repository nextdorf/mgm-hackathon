from typing import Dict, Literal
from langchain.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic


from ..util import LazyVal
from ._openrouter import ChatOpenRouter
from ._cerebras import ChatCerebras



class MultiChatModel:
  llm_ids = (
    'gpt-5-mini-low', 'gpt-5-medium', 'gpt-4o-mini', 'gpt-4.1-nano',
    'gemini-lite', 'gemini', 'gemini-exp',
    'claude-haiku', 'claude-sonnet', 'claude-opus',
    'router-gemini-lite', 'router-gpt-oss-120b',
    'cerebras-gpt-oss-120b', 'cerebras-llama', 'cerebras-qwen',
  )
  def __init__(self, llm_id = 'gemini'):
    self._llms: Dict[str, LazyVal[BaseChatModel]] = {
      'gpt-5-mini-low': LazyVal(lambda: ChatOpenAI(model='gpt-5-mini', reasoning_effort='low')),
      'gpt-5-medium': LazyVal(lambda: ChatOpenAI(model='gpt-5', reasoning_effort='medium')),
      'gpt-4o-mini': LazyVal(lambda: ChatOpenAI(model='gpt-4o-mini')),
      'gpt-4.1-nano': LazyVal(lambda: ChatOpenAI(model='gpt-4.1-nano')),

      'gemini-lite': LazyVal(lambda: ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite')),
      'gemini': LazyVal(lambda: ChatGoogleGenerativeAI(model='gemini-2.5-flash')),
      'gemini-exp': LazyVal(lambda: ChatGoogleGenerativeAI(model='gemini-2.0-flash-exp')),

      'claude-haiku': LazyVal(lambda: ChatAnthropic(model='claude-haiku-4-5')), # pyright: ignore[reportCallIssue]
      'claude-sonnet': LazyVal(lambda: ChatAnthropic(model='claude-sonnet-4-5')), # pyright: ignore[reportCallIssue]
      'claude-opus': LazyVal(lambda: ChatAnthropic(model='claude-opus-4-1')), # pyright: ignore[reportCallIssue]

      'router-gemini-lite': LazyVal(lambda: ChatOpenRouter(model='google/gemini-2.5-flash-lite')),
      'router-gpt-oss-120b': LazyVal(lambda: ChatOpenRouter(model='openai/gpt-oss-120b')),

      'cerebras-gpt-oss-120b': LazyVal(lambda: ChatCerebras(model='gpt-oss-120b', extra_body=dict(reasoning_effort='low'))),
      'cerebras-llama': LazyVal(lambda: ChatCerebras(model='llama3.1-8b')),
      'cerebras-qwen': LazyVal(lambda: ChatCerebras(model='qwen-3-32b')),
    }
    diff_set = set(
        [x for x in MultiChatModel.llm_ids if x not in self._llms]
      + [x for x in self._llms if x not in MultiChatModel.llm_ids]
    )
    assert not diff_set, f'Unaccounted ids: {diff_set}'
    self.llm_id = llm_id

  @property
  def llm(self):
    return self._llms[self.llm_id].val

  @property
  def model_provider(self) -> Literal['gpt', 'gemini', 'claude', 'router', 'cerebras']:
    return self.llm_id.split('-')[0] # pyright: ignore[reportReturnType]

  @property
  def model_name(self):
    provider = self.model_provider
    if provider == 'gpt':
      return self.llm.model_name
    elif provider == 'gemini':
      return self.llm.model.split('/', 1)[-1]
    else:
      try:
        return self.llm.model
      except:
        return 'UNKNOWN'

  def with_id(self, llm_id):
    res = MultiChatModel(llm_id)
    res._llms = { k: v.copy() for k,v in self._llms.items() } # pylint: disable=protected-access
    return res

  def __setitem__(self, llm_id, llm):
    self._llms[llm_id].val = llm
  def __getitem__(self, llm_id):
    return self._llms[llm_id].val
  def __delitem__(self, llm_id):
    del self._llms[llm_id].val
  def __iter__(self):
    return (x.val for x in self._llms.values())

  def items(self):
    return ((k, v.val) for k,v in self._llms.items())

  def __repr__(self):
    return f'{MultiChatModel.__qualname__}(id={self.llm_id}, model={self.model_name})'


if __name__ == '__main__':
  import time
  import numpy as np
  from langgraph.graph import add_messages


  multi_llm = MultiChatModel()
  responses = {}
  hist = add_messages([], [
    ('system', 'You are being benchmarked. Follow the instructions strictly and answer with 1000 words.'),
    ('user', 'Explain in abuot 1000 words (the word does not have to be exact) how a LLM works. Don\'t talk back and just do what I am asking immediatly'),
  ])
  hist

  for k, llm in multi_llm.items():
    # if k.split('-')[0] not in 'claude'.split():
    #   continue
    # print(k)
    ts, rs = [], []
    ts.append(time.time_ns())
    for ch in llm.stream(hist):
      rs.append(ch.content)
      ts.append(time.time_ns())
    ts = (np.asarray(ts[1:]) - np.asarray(ts[:-1])) / 1e9
    responses[k] = dict(
      chunks = rs,
      full = ''.join(rs),
      ts = ts,
      tft = ts[0],
      t_tot = np.sum(ts),
      dt_mean = np.mean(ts[1:]),
      dt_std = np.std(ts[1:], ddof=1),
    )

    print('{k}: ({tft:.3f} + {dt_mean:.3f} ± {dt_std:.3f})s -> {t_tot:.3f}s'.format(k=k, **responses[k]))

    # s_full = responses[k]['full']
    # s = s_full if len(s_full) <= 150 else f'{s_full[:72]} ... {s_full[-72:]}'
    words = responses[k]['full'].split(' ')
    s = ' '.join(words) if len(words) <= 150 else f'{' '.join(words[:74])} ... {' '.join(words[-74:])}'
    print('\n  | '.join(f'  | {s}'.splitlines()))


  # gpt-5-mini-low: (3.938 + 0.017 ± 0.032)s -> 27.986s
  #   | A large language model (LLM) is a type of neural network designed to process and generate human-like text by learning statistical patterns from vast amounts of language data. Understanding how an LLM works involves several connected pieces: data and tokens, model architecture (primarily the transformer), training objectives and optimization, inference and decoding, and practical limitations and safety concerns. Below is a coherent explanation that ties these parts together.
  #   | 
  #   | Data and tokenization
  #   | LLMs are trained on large ... generate language by tokenizing text into embeddings, processing sequences through transformer layers that use attention to model context, and being trained with next-token prediction and variants like RLHF to align behavior. During inference, decoding strategies translate token probabilities into coherent text. The result is a powerful but imperfect statistical model of language capable of many tasks yet susceptible to hallucination, bias, and other limitations; addressing these issues requires careful training, evaluation, and deployment practices.
  # gpt-5-medium: (78.751 + 0.010 ± 0.042)s -> 92.171s
  #   | Large language models (LLMs) are probabilistic programs that learn to predict the next token in a sequence and, through that simple objective, acquire broad linguistic and world knowledge. At their core, they are neural networks based on the Transformer architecture, composed of layers that repeatedly mix information across positions (self-attention) and transform it (feed-forward networks), guided by billions of learned parameters.
  #   | 
  #   | Text must first be converted into tokens, which are subword units created by methods ... efficient attention, tighter integration with tools, and deeper interpretability—LLMs will become more capable and controllable, while responsible design is needed to ensure safety, privacy, and societal benefit.
  #   | 
  #   | Understanding how LLMs work clarifies both their promise and their pitfalls: they are scalable statistical learners, not omniscient oracles. With careful data practices, transparent evaluation, and thoughtful integration with retrieval and tools, they can augment human work while keeping humans in the loop and in control and accountable.
  # gpt-4o-mini: (1.112 + 0.019 ± 0.021)s -> 26.470s
  #   | Large Language Models (LLMs) have revolutionized the way we interact with technology and process language. By combining advanced machine learning techniques with extensive datasets, LLMs like OpenAI’s GPT-3 and others can generate human-like text, answer questions, translate languages, and much more. This explanation aims to provide a comprehensive overview of how LLMs work, delving into their architecture, training processes, and the underlying mechanisms that enable them to understand and generate language.
  #   | 
  #   | ### Architecture of Large ... requires substantial computational power and energy, contributing to environmental concerns.
  #   | 
  #   | ### Conclusion
  #   | 
  #   | Large Language Models represent a significant leap in understanding and generating human language, powered by advanced architectures, extensive training on diverse data, and sophisticated mechanisms of context processing. As research and technology progress, their capabilities will continue to expand, opening new possibilities in artificial intelligence and human-computer interaction. However, addressing ethical considerations and refining their application remains crucial to harnessing their full potential responsibly.
  # gpt-4.1-nano: (1.664 + 0.010 ± 0.017)s -> 14.522s
  #   | A Large Language Model (LLM) is a sophisticated type of artificial intelligence designed to understand, generate, and manipulate human language with remarkable proficiency. The core idea behind an LLM revolves around processing vast amounts of textual data to learn patterns, relationships, and structures inherent in language, enabling it to perform a wide array of language-related tasks with impressive accuracy. To understand how an LLM works, it is essential to delve into its underlying architecture, ... language understanding. However, despite their sophistication, they remain tools that operate based on statistical associations rather than true comprehension. As research progresses, continual improvements aim to make LLMs more accurate, less biased, and more aligned with human values and expectations, so that they can serve as reliable and valuable assistants across countless domains. Their ongoing evolution promises to reshape how humans interact with machines, access knowledge, and automate complex language-based tasks in the future.
  # gemini-lite: (0.758 + 0.178 ± 0.079)s -> 6.643s
  #   | Large Language Models (LLMs) are a fascinating and rapidly evolving area of artificial intelligence. At their core, they are sophisticated computer programs designed to understand, generate, and manipulate human language. The "Large" in their name refers to two key aspects: the sheer size of the datasets they are trained on and the enormous number of parameters (variables) within their architecture.
  #   | 
  #   | The fundamental process by which LLMs work can be broken down into several key stages: ... shorter versions.
  #   | *   **Translation:** Converting text from one language to another.
  #   | *   **Code Generation:** Writing code based on descriptions.
  #   | *   **Sentiment Analysis:** Determining the emotional tone of text.
  #   | 
  #   | In essence, LLMs learn to predict the probability of sequences of tokens. By doing so, they develop a sophisticated internal representation of language that allows them to perform an astonishing array of tasks, making them powerful tools for communication, information retrieval, and creative endeavors.
  # gemini: (5.569 + 0.375 ± 0.118)s -> 22.056s
  #   | A Large Language Model (LLM) is a sophisticated type of artificial intelligence designed to understand, generate, and manipulate human language. At its core, an LLM is a neural network, specifically a transformer-based architecture, trained on an enormous corpus of text data. This training allows it to learn the statistical relationships, patterns, grammar, and even some semantic meaning within language, enabling it to perform a wide array of language-related tasks, from translation and summarization to ... training data, and lack true common-sense reasoning or consciousness. They are sophisticated pattern matchers and predictors, not sentient beings.
  #   | 
  #   | In summary, an LLM functions by tokenizing input, processing it through layers of self-attention and feed-forward networks within a Transformer architecture, learning to predict the next token based on billions of parameters trained on vast text datasets, and then refining this capability through fine-tuning and human feedback to generate coherent, contextually relevant, and increasingly helpful language.
  # gemini-exp: (0.658 + 0.360 ± 0.166)s -> 11.465s
  #   | Okay, here's an explanation of how a Large Language Model (LLM) works, aiming for approximately 1000 words.
  #   | 
  #   | **What is a Large Language Model (LLM)?**
  #   | 
  #   | At its core, a Large Language Model is a type of artificial intelligence (AI) model designed to understand and generate human language. It's "large" because it's trained on a massive dataset of text and code, often comprising billions or even trillions of words. This vast training allows the model to learn intricate ... computationally expensive.
  #   | 
  #   | **Conclusion:**
  #   | 
  #   | Large Language Models are powerful AI models that have revolutionized the field of natural language processing. They are based on the transformer architecture and are trained on massive datasets of text and code. While they have limitations, LLMs are capable of performing a wide range of language-based tasks, from text generation to question answering to code completion. Continued research and development are addressing the limitations and further enhancing the capabilities of these models.
  # claude-haiku: (0.931 + 0.058 ± 0.055)s -> 16.247s
  #   | # How a Large Language Model Works
  #   | 
  #   | A Large Language Model (LLM) is a sophisticated artificial intelligence system designed to understand and generate human language. The mechanisms that power these systems involve complex mathematics, neural networks, and statistical learning principles. Understanding how an LLM works requires examining everything from the fundamental architecture to the training process and inference mechanisms that allow these systems to produce coherent, contextually relevant text.
  #   | 
  #   | ## The Transformer Architecture
  #   | 
  #   | At the heart of ... an intricate combination of mathematical operations, learned parameters, and architectural innovations. By tokenizing text, converting tokens to embeddings, processing information through multiple layers of attention and feed-forward networks, and training on massive amounts of data to predict the next token, these models learn to generate remarkably coherent and contextually appropriate text. While they operate through statistical pattern recognition rather than true understanding, their performance across diverse tasks demonstrates the power of scaling neural networks.
  # claude-sonnet: (3.501 + 0.056 ± 0.095)s -> 32.801s
  #   | Large Language Models (LLMs) are sophisticated artificial intelligence systems designed to understand and generate human language. These models represent one of the most significant breakthroughs in natural language processing and have transformed how machines interact with text.
  #   | 
  #   | At their foundation, LLMs are built on neural network architectures, most commonly the transformer architecture introduced in 2017. The transformer revolutionized language processing by introducing a mechanism called "attention," which allows the model to weigh the importance of ... for human knowledge—by learning language patterns, these models indirectly learn about the world those patterns describe.
  #   | 
  #   | The development of LLMs continues rapidly, with researchers exploring more efficient architectures, better training methods, improved alignment techniques, and ways to integrate external knowledge and tools. As these systems become more capable, they raise important questions about their appropriate use, potential risks, and societal impact, making the understanding of how they work increasingly important for users and policymakers alike.
  # claude-opus: (1.408 + 0.090 ± 0.141)s -> 36.866s
  #   | A Large Language Model (LLM) is a sophisticated artificial intelligence system designed to understand and generate human-like text by processing vast amounts of textual data through deep neural network architectures. At its core, an LLM operates using a transformer architecture, which revolutionized natural language processing when introduced in 2017.
  #   | 
  #   | The fundamental building block of an LLM is the transformer, which employs a mechanism called self-attention. This mechanism allows the model to weigh the importance of ... constitutional AI approaches, and careful dataset curation to reduce biases and harmful outputs.
  #   | 
  #   | Understanding LLMs requires appreciating their statistical nature. They operate by identifying and reproducing patterns from training data, creating an implicit model of language that enables seemingly intelligent behavior. While they don't truly understand text in the human sense, their ability to process and generate language has proven remarkably useful across countless applications, from writing assistance to code generation to complex reasoning tasks.
  # router-gemini-lite: (0.579 + 0.150 ± 0.087)s -> 5.086s
  #   | Large Language Models (LLMs) are a type of artificial intelligence designed to understand, generate, and manipulate human language. They are trained on massive datasets of text and code, allowing them to learn patterns, grammar, facts, reasoning abilities, and even stylistic nuances present in the data. At their core, LLMs are sophisticated pattern-matching and prediction machines that operate through a complex interplay of mathematical operations and statistical probabilities.
  #   | 
  #   | The foundational architecture for most modern LLMs is ... is reached).
  #   | 
  #   | In essence, an LLM works by transforming input text into numerical representations, processing these representations through a complex neural network (the Transformer) that uses self-attention to understand context and relationships, and then generating new text by probabilistically predicting the most likely next token in a sequence, iteratively building a coherent and contextually relevant response. The immense scale of their training data and parameters is what enables their remarkable fluency and breadth of knowledge.
  # router-gpt-oss-120b: (0.408 + 0.002 ± 0.016)s -> 5.779s
  #   | Large language models (LLMs) are statistical systems that generate or analyse natural‑language text by learning patterns from massive collections of written material.  At a high level, an LLM takes a sequence of symbols—usually words or sub‑word pieces—converts them into numerical vectors, runs those vectors through a deep neural network (most commonly a transformer architecture), and finally produces a probability distribution over the next possible symbols.  By repeating this step, the model can ... architectural variants (sparse attention, MoE, retrieval) and multimodal extensions to tackle new challenges.
  #   | 
  #   | The combined effect of massive data, deep transformer stacks, and sophisticated training regimes enables LLMs to capture rich linguistic regularities, perform reasoning, answer questions, write code, and even exhibit emergent behaviours that surprise researchers.  As compute and data continue to grow, and as alignment techniques mature, LLMs are poised to become ever more capable foundations for a variety of AI‑driven applications.
  # cerebras-gpt-oss-120b: (0.429 + 0.001 ± 0.002)s -> 1.992s
  #   | **How Large Language Models Work: An Approximate 1,000‑Word Overview**
  #   | 
  #   | ---
  #   | 
  #   | ### 1. Introduction  
  #   | 
  #   | Large language models (LLMs) are a class of artificial‑intelligence systems that generate or understand human language by learning statistical patterns from massive text corpora. Though they can appear magical, their operation can be broken down into a series of well‑understood components: data collection, tokenization, model architecture, training objectives, optimization, inference, and safety mechanisms. This overview walks through each stage, emphasizing the core ... returns the detokenized answer. Safety mechanisms and optional retrieval components help steer outputs toward useful and responsible behavior.
  #   | 
  #   | Although LLMs can generate impressively coherent and knowledgeable text, they remain statistical machines that inherit the strengths and flaws of their training data and architecture. Ongoing research continues to push the frontier toward more efficient, trustworthy, and multimodal systems, but the core principles—large-scale data, tokenization, Transformer self‑attention, and next‑token prediction—remain the foundation of how modern LLMs work.
  # cerebras-llama: (0.469 + 0.000 ± 0.000)s -> 0.806s
  #   | A Large Language Model (LLM) is a type of artificial intelligence designed to process and generate human-like language. These models are trained on vast amounts of text data, enabling them to learn patterns, relationships, and structures of language. In this explanation, we'll delve into the inner workings of LLMs, including their architecture, training process, and capabilities.
  #   | 
  #   | **Architecture of LLMs**
  #   | 
  #   | LLMs typically employ a transformer architecture, which is a type of neural network designed specifically for handling ... challenging due to the subjective nature of language understanding.
  #   | 
  #   | **Conclusion**
  #   | 
  #   | In conclusion, LLMs have made significant progress in recent years and have been shown to possess various capabilities, including language translation, text summarization, question answering, and sentiment analysis. However, there are still several challenges and limitations to consider, including computational resources, training data, adversarial examples, and evaluation metrics. As the field continues to evolve, it's likely that we'll see further advancements in LLM capabilities and challenges.
  # cerebras-qwen: (0.371 + 0.001 ± 0.004)s -> 2.037s
  #   | <think>
  #   | Okay, the user wants a 1000-word explanation on how LLMs work, and they want it without me talking back. Let me start by outlining the key components. I need to cover the basics of neural networks, transformers, training processes, and how they generate text.
  #   | 
  #   | First, I should explain the general concept of a language model. Maybe start with the shift from traditional models to deep learning. Mention the role of training data, especially large corpora. ... on massive datasets to capture linguistic patterns. They generate text through token prediction, guided by probabilities learned during training and refined by decoding strategies. While transformative, their outputs are shaped by the data they were trained on, and their limitations—like biases and hallucinations—highlight the need for careful oversight. As research advances, LLMs are likely to become even more capable, with narrower models tailored to specific domains and improved methods for ensuring accuracy and fairness.

