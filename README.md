# qa ai process project


# Setup Instructions

1. Create a virtual environment:
	```sh
	python3 -m venv venv
	```

2. Activate the virtual environment:
	```sh
	source venv/bin/activate
	```

3. Install dependencies from requirements.txt:
	```sh
	pip install -r requirements.txt
	```

4. Download and install Ollama:
	```sh
	curl -fsSL https://ollama.com/install.sh | sh
	```

5. Start the Ollama server:
	```sh
	ollama serve
	```

6. Run a model (example with llama3.2):
	```sh
	ollama run llama3.2
	```
