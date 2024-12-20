{ pkgs, ... }: {
  channel = "stable-24.05"; # Define el canal de Nix
  packages = [
    pkgs.python310
    pkgs.python310Packages.pip 
    pkgs.python310Packages.numpy
    #pkgs.python310Packages.scikit-learn
    #pkgs.python310Packages.pdfplumber
    #pkgs.python310Packages.flask
    #pkgs.python310Packages.transformers
    #pkgs.python310Packages.torch
    #pkgs.python310Packages.torchaudio
    #pkgs.python310Packages.torchvision
    #pkgs.python310Packages.tokenizers
  ];
}