{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = [
    pkgs.python3
    pkgs.python3Packages.torch
    pkgs.python3Packages.torchvision
    pkgs.python3Packages.numpy
  ];
}
