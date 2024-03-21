{
  description = "NixOS environment";

  inputs = { nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable"; };

  outputs = { self, nixpkgs, }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };

    in
    {
      devShell.${system} = with pkgs;

        mkShell rec {
          packages = [
            python311
            pipenv
            nodejs
            python311Packages.transformers
            python311Packages.torch
            python311Packages.numpy
            python311Packages.pandas
            (pkgs.python311Packages.opencv4.override { enableGtk2 = true; })
            facedetect
          ];
        };
    };
}
