{
  pkgs,
  lib,
  config,
  inputs,
  ...
}:

{
  packages = with pkgs; [
    git
    portaudio
    # soundevice
    (python312.withPackages (
      ps: with ps; [
        aiohttp
        beautifulsoup4
        chromadb
        google-api-python-client
        gtts
        langchain
        langchain-community
        langchain-openai
        langgraph
        numpy
        openai
        openai-whisper
        pillow
        prompt-toolkit
        # pyautogui
        pydantic
        pypdf2
        pyperclip
        pyside6
        pytest
        pytest-asyncio
        python-dotenv
        requests
        rich
        soundfile
        speechrecognition
        textual
        typer
        yaspin
      ]
    ))
    # Example: Build the local Python package using Nix
    # Uncomment and adjust the following block as needed:
    (python312Packages.buildPythonPackage {
      pname = "intuit";
      version = "1.0.0";
      src = ./.;
      pyproject = true;
      # Ensure build dependencies for pyproject.toml are available
      propagatedBuildInputs = [
        python312Packages.setuptools
        python312Packages.wheel
      ];
      # Optionally add meta or other attributes
    })
  ];

  languages.python.enable = true;
  languages.python.venv.enable = true;
  languages.python.venv.quiet = true;
  languages.python.venv.requirements = ''
    fastmcp>=2.3.0
    langmem>=0.1.0
    mcp-client>=0.2.0
    mcp>=1.8.0
  '';

  enterShell = ''
    git --version
  '';

  # claude.code.enable = true;

  git-hooks.hooks = {
    rustfmt.enable = true;
    nixfmt-rfc-style.enable = true;
    black.enable = true;
    prettier.enable = true;
  };

  cachix.enable = true;
  cachix.pull = [
    "pre-commit-hooks"
    "nix-community"
  ];
  cachix.push = "cv-ml-cache";

  dotenv.enable = true;
  difftastic.enable = true;
  delta.enable = true;
  # See full reference at https://devenv.sh/reference/options/
}
