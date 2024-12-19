# To learn more about how to use Nix to configure your environment
# see: https://developers.google.com/idx/guides/customize-idx-env
{ pkgs, ... }: {
  # Which nixpkgs channel to use.
  channel = "stable-24.05"; # or "unstable"

  # Use https://search.nixos.org/packages to find packages
  packages = [
    pkgs.python310 # Python 3.10
    pkgs.python310Packages.pip # pip package manager

    # Python dependencies for your project
    pkgs.python310Packages.numpy # Numerical operations
    pkgs.python310Packages.scikit-learn # Machine learning and TF-IDF
    pkgs.python310Packages.pdfplumber # PDF text extraction
    pkgs.python310Packages.flask # Flask web framework
    pkgs.python310Packages.transformers # Hugging Face Transformers library
    pkgs.python310Packages.torch # PyTorch for machine learning models

    # Optional: Node.js if required for frontend development (uncomment if needed)
    # pkgs.nodejs_20
    # pkgs.nodePackages.nodemon
  ];

  # Sets environment variables in the workspace
  env = {};
  idx = {
    # Search for the extensions you want on https://open-vsx.org/ and use "publisher.id"
    extensions = [
      # Example: Uncomment if you use Vim for editing
      # "vscodevim.vim"
    ];

    # Enable previews
    previews = {
      enable = true;
      previews = {
        # Uncomment and configure if your project uses a development server
        # web = {
        #   command = ["npm" "run" "dev"];
        #   manager = "web";
        #   env = {
        #     PORT = "$PORT";
        #   };
        # };
      };
    };

    # Workspace lifecycle hooks
    workspace = {
      # Runs when a workspace is first created
      onCreate = {
        # Example: Install Python dependencies
        pip-install = "pip install -r requirements.txt";
      };
      # Runs when the workspace is (re)started
      onStart = {
        # Example: Run Flask development server
        flask-server = "python main.py";
      };
    };
  };
}