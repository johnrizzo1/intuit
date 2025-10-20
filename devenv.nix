{ pkgs, lib, config, inputs, ... }:

{
  packages = with pkgs; [
    git
    portaudio
  ];

  languages.python.enable = true;
  languages.python.venv.enable = true;
  languages.python.uv.enable = true;
  languages.python.version = "3.12";

  enterShell = ''
    git --version
    echo "Python environment ready. Use 'uv pip install -e .' to install dependencies."
  '';
  
  difftastic.enable = true;
  dotenv.enable = true;
  
  # See full reference at https://devenv.sh/reference/options/
}
