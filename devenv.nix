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
    (python3.withPackages (
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
        psutil
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
        sounddevice
        soundfile
        speechrecognition
        textual
        tkinter
        typer
        yaspin
      ]
    ))
    # Example: Build the local Python package using Nix
    # Uncomment and adjust the following block as needed:
    (python3Packages.buildPythonPackage {
      pname = "intuit";
      version = "1.0.0";
      src = ./.;
      pyproject = true;
      # Ensure build dependencies for pyproject.toml are available
      nativeBuildInputs = [
        python3Packages.setuptools
        python3Packages.wheel
      ];
      propagatedBuildInputs = with python3Packages; [
        aiohttp
        beautifulsoup4
        chromadb
        google-api-python-client
        google-auth-oauthlib
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
        psutil
        pyautogui
        pydantic
        pypdf2
        pyperclip
        pyside6
        pytest
        pytest-asyncio
        python-dotenv
        requests
        rich
        sounddevice
        soundfile
        speechrecognition
        textual
        tkinter
        typer
        yaspin
      ];
      # Handle packages not available in nixpkgs through pip in venv
      doCheck = false; # Skip tests during build to avoid dependency issues
      dontCheckRuntimeDeps = true; # Skip runtime dependency checking for packages handled by venv
      # Optionally add meta or other attributes
    })
  ];

  languages.python = {
    enable = true;
    venv = {
      enable = true;
      quiet = true;
      requirements = ''
        fastmcp>=2.3.0
        langmem>=0.1.0
        mcp-client>=0.2.0
        mcp>=1.8.0
      '';
    };
  };

  # tasks = {
  #   "intuit:chat" = {
  #     exec = "python -m intuit.main chat";
  #   };
  # };

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
