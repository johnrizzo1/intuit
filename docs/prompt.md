# Intuit: A personal assistant

Today we are building an agentic assistant.  This assistant should be incredibly flexible.  We should be able to easily add new tools.  At a minimum the agent should be able to search the internet, read my gmail and tell me the weather.  The agent should be able to be invoked on the cli and it will drop you into a curses like interface where you can converse with the agent.  You should also be able to pass the query to the agent via a simple command like argument.  Finally and possibly most importantly you should be able to converse with the agent in realtime by voice.

Requirements

* Build a vector database of all the content of the files on my filesystem and use them as part of the query
* Uses devenv to manage dependencies
* Written in python
* Model the application after https://openai.github.io/openai-agents-python/voice/quickstart/
* Multi-Process to help improve latency
* Full-Duplex
