{ pkgs, lib, config, inputs, ... }:

{
  packages = with pkgs; [ 
    git
    portaudio
    (python3.withPackages (ps: with ps; [
      openai
      pydantic
      typer
      rich
      chromadb
    ]))
  ];

  languages.python.enable = true;
  languages.python.venv.enable = true;
  languages.python.uv.enable = true;

  enterShell = ''
    git --version
  '';
  
  difftastic.enable = true;
  dotenv.enable = true;
  # See full reference at https://devenv.sh/reference/options/
}
