# Emoji Searcher App

This app lets you search emojis using sparse vector search powered by MiniCoil-v1 and [Qdrant](https://qdrant.tech).

### Get it up and running!

Get the GitHub repository:

```bash
git clone https://github.com/AstraBert/emoji-searcher-app
```

Install dependencies:

```bash
uv sync
```

And then modify the `.env.example` file with your API keys and move it to `.env`.

Now, you will have to execute the following scripts:

```bash
uv run tools/load_data_to_qdrant.py
```

You're ready to set up the app!

Last, run the Gradio frontend, and start exploring at http://localhost:7860:

```bash
uv run src/emoji_searcher_app/main.py
```

### Contributing

Contribute to this project following the [guidelines](./CONTRIBUTING.md).

### License

This project is provided under an [MIT License](LICENSE).
