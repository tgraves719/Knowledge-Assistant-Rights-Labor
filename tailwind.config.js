/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: "class",
  content: [
    "./frontend/modular/**/*.html",
    "./frontend/modular/**/*.js"
  ],
  theme: {
    extend: {
      colors: {
        ufcw: {
          blue: "#0D3B54",
          "blue-mid": "#14506E",
          "blue-light": "#1B6B8A",
          gold: "#D4A029",
          "gold-light": "#E8B84A"
        }
      }
    }
  }
};
