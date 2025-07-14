{ pkgs ? import <nixpkgs> {} }:
pkgs.mkShell {
  buildInputs = [
    pkgs.imagemagick  # Для обработки изображений в CLI/скриптах
    pkgs.nodejs       # Для Next.js
  ];

  # Отключаем X11, если не нужен GUI
  imagemagick = pkgs.imagemagick.override { enableX11 = false; };

  # Настройки для Node.js
  shellHook = ''
    export PATH="$PATH:$(pwd)/node_modules/.bin"
  '';
}